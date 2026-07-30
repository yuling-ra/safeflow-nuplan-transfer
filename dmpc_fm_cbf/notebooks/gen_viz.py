"""Standalone visualization script — uses cached traces/metrics, no model needed."""
import math, random, sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from pyproj import Transformer
from shapely.geometry import (GeometryCollection, LineString,
                               MultiLineString, MultiPolygon, Polygon)

# ── paths ──────────────────────────────────────────────────────────────
ROOT      = Path('/new_world/cockatiel/ra').resolve()
DATA_ROOT = ROOT / 'data' / 'cache' / 'mini'
MAP_ROOT  = ROOT / 'maps'
CACHE_DIR = ROOT / 'safeflow-nuplan-transfer/dmpc_fm_cbf/notebooks/cache/nuplan_transfer'
OUT_PATH  = CACHE_DIR / 'trajectory_fig3_map.png'

# ── load cached results ────────────────────────────────────────────────
traces_df  = pd.read_csv(CACHE_DIR / 'traces.csv')
metrics_df = pd.read_csv(CACHE_DIR / 'metrics.csv')
print(f'Loaded {len(traces_df)} trace rows, {len(metrics_df)} metric rows')

# ── helpers ────────────────────────────────────────────────────────────
def yaw_from_quaternion(qw, qx, qy, qz):
    return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz)))

def blob_hex(b): return None if b is None else b.hex()


@dataclass
class ScenarioRecord:
    db_path: Path; scene_token: bytes; anchor_lidar_pc_token: bytes
    goal_ego_pose_token: Optional[bytes]; scenario_tag: str
    paper_bucket: str; scene_name: str; location: str; map_version: str


def discover_scenarios():
    records = []
    for db_path in sorted(DATA_ROOT.glob('*.db')):
        try:
            with sqlite3.connect(str(db_path)) as con:
                cur = con.cursor()
                row = cur.execute('SELECT location,map_version FROM log LIMIT 1').fetchone()
                if not row: continue
                loc, mv = row
                for bucket, tag in [('lane_following','following_lane_without_lead'),
                                     ('left_turn','starting_left_turn')]:
                    for st,at,gt,sn,sc in cur.execute('''
                        SELECT DISTINCT lp.scene_token,st2.lidar_pc_token,
                               s.goal_ego_pose_token,s.name,st2.type
                        FROM scenario_tag st2
                        JOIN lidar_pc lp ON st2.lidar_pc_token=lp.token
                        JOIN scene s ON lp.scene_token=s.token
                        WHERE st2.type=? ORDER BY s.name''',(tag,)).fetchall():
                        records.append(ScenarioRecord(
                            db_path=db_path, scene_token=st, anchor_lidar_pc_token=at,
                            goal_ego_pose_token=gt, scenario_tag=sc,
                            paper_bucket=bucket, scene_name=sn, location=loc, map_version=mv))
        except sqlite3.DatabaseError:
            pass
    return records


def load_bundle(record):
    con = sqlite3.connect(str(record.db_path)); cur = con.cursor()
    frame_rows = cur.execute('''
        SELECT lp.token,lp.timestamp,ep.x,ep.y,ep.qw,ep.qx,ep.qy,ep.qz,ep.vx,ep.vy
        FROM lidar_pc lp JOIN ego_pose ep ON lp.ego_pose_token=ep.token
        WHERE lp.scene_token=? ORDER BY lp.timestamp''',(record.scene_token,)).fetchall()
    goal_xy = None
    if record.goal_ego_pose_token is not None:
        r = cur.execute('SELECT x,y FROM ego_pose WHERE token=?',
                        (record.goal_ego_pose_token,)).fetchone()
        if r: goal_xy = np.array(r, np.float32)
    frames = []; last_ts = None
    for token,ts,x,y,qw,qx,qy,qz,vx,vy in frame_rows:
        dt = 0.1 if last_ts is None else max((ts-last_ts)*1e-6, 1e-3)
        last_ts = ts
        agents = [{'xy':np.array([ax,ay],np.float32),'yaw':float(ayaw),
                   'width':float(w),'length':float(l)}
                  for _,ax,ay,_,_,ayaw,w,l,_ in cur.execute('''
                      SELECT lb.track_token,lb.x,lb.y,lb.vx,lb.vy,
                             lb.yaw,lb.width,lb.length,c.name
                      FROM lidar_box lb JOIN track t ON lb.track_token=t.token
                      JOIN category c ON t.category_token=c.token
                      WHERE lb.lidar_pc_token=? AND c.name='vehicle' ''',(token,)).fetchall()]
        frames.append({'lidar_pc_token':token,'dt':dt,
                       'ego_x':float(x),'ego_y':float(y),
                       'ego_yaw':yaw_from_quaternion(qw,qx,qy,qz),'agents':agents})
    con.close()
    anchor_idx = next(i for i,f in enumerate(frames)
                      if f['lidar_pc_token'] == record.anchor_lidar_pc_token)
    if goal_xy is None:
        gf = min(anchor_idx+80, len(frames)-1)
        goal_xy = np.array([frames[gf]['ego_x'], frames[gf]['ego_y']], np.float32)
    corridor = np.array([[f['ego_x'],f['ego_y']] for f in frames[anchor_idx:]], np.float32)
    history  = np.array([[f['ego_x'],f['ego_y']] for f in frames[:anchor_idx+1]], np.float32)
    return {'record':record, 'frames':frames, 'anchor_index':anchor_idx,
            'goal_xy':goal_xy, 'corridor_xy':corridor, 'history_xy':history}


# ── select representative scene for each bucket ────────────────────────
all_records = discover_scenarios()
rng = random.Random(7)
selected = []
for bucket in ['lane_following', 'left_turn']:
    pool = [r for r in all_records if r.paper_bucket == bucket]
    rng.shuffle(pool)
    selected.extend(pool[:8])
print(f'Selected {len(selected)} scenarios')

BUNDLE_CACHE = {}
def get_bundle(record):
    key = (str(record.db_path), blob_hex(record.anchor_lidar_pc_token))
    if key not in BUNDLE_CACHE:
        BUNDLE_CACHE[key] = load_bundle(record)
    return BUNDLE_CACHE[key]

# ── map drawing ────────────────────────────────────────────────────────
MAP_LAYER_STYLES = [
    ('generic_drivable_areas', dict(facecolor='#eef1ec',edgecolor='none',linewidth=0.0,alpha=0.95,zorder=0)),
    ('road_segments',          dict(facecolor='#f4f5f1',edgecolor='#d4d8cf',linewidth=0.35,alpha=0.95,zorder=1)),
    ('lanes_polygons',         dict(facecolor='#fbfbf7',edgecolor='#b8beb6',linewidth=0.45,alpha=0.96,zorder=2)),
    ('gen_lane_connectors_scaled_width_polygons',
                               dict(facecolor='#f7f7f2',edgecolor='#c7ccc3',linewidth=0.35,alpha=0.90,zorder=2)),
    ('intersections',          dict(facecolor='#ede4d7',edgecolor='#cbbba6',linewidth=0.35,alpha=0.70,zorder=3)),
    ('crosswalks',             dict(facecolor='#f1d59c',edgecolor='#bd9448',linewidth=0.40,alpha=0.78,zorder=4)),
    ('baseline_paths',         dict(color='#6f7782',linewidth=0.55,alpha=0.75,linestyle='--',zorder=5)),
    ('boundaries',             dict(color='#666666',linewidth=0.55,alpha=0.55,linestyle='-',zorder=6)),
]
_MAP_CACHE = {}

def map_gpkg(record):
    cands = (sorted((MAP_ROOT/record.map_version).glob('*/map.gpkg')) +
             sorted((MAP_ROOT/record.location).glob('*/map.gpkg')))
    if not cands:
        raise FileNotFoundError(f'No map for {record.location}/{record.map_version}')
    return cands[0]

def get_map_crs(gpkg_path):
    key = ('crs', str(gpkg_path))
    if key not in _MAP_CACHE:
        meta = gpd.read_file(gpkg_path, layer='meta')
        d = dict(zip(meta['key'], meta['value']))
        _MAP_CACHE[key] = (d.get('projectedCoordSystem','epsg:32617'),
                           d.get('geographicCoordSystem','epsg:4326'))
    return _MAP_CACHE[key]

def bounds_proj_to_geo(bounds, proj_crs, geo_crs='epsg:4326'):
    minx,miny,maxx,maxy = [float(v) for v in bounds]
    t = Transformer.from_crs(proj_crs, geo_crs, always_xy=True)
    pts = [(minx,miny),(minx,maxy),(maxx,miny),(maxx,maxy)]
    ll = np.array([t.transform(x,y) for x,y in pts])
    return float(ll[:,0].min()), float(ll[:,1].min()), float(ll[:,0].max()), float(ll[:,1].max())

def load_layer(gpkg_path, layer, bounds_proj):
    proj_crs, geo_crs = get_map_crs(gpkg_path)
    bbox_geo = bounds_proj_to_geo(bounds_proj, proj_crs, geo_crs)
    key = ('lyr', str(gpkg_path), layer, tuple(round(v,1) for v in bounds_proj))
    if key in _MAP_CACHE: return _MAP_CACHE[key]
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer, bbox=bbox_geo)
    except Exception as e:
        gdf = gpd.GeoDataFrame(geometry=[], crs=geo_crs)
    if len(gdf) and gdf.crs is None: gdf = gdf.set_crs(geo_crs)
    if len(gdf): gdf = gdf.to_crs(proj_crs)
    _MAP_CACHE[key] = gdf
    return gdf

def _draw_poly(ax, poly, facecolor, edgecolor, linewidth, alpha, zorder):
    xy = np.asarray(poly.exterior.coords)
    ax.fill(xy[:,0], xy[:,1], facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, alpha=alpha, zorder=zorder)
    for interior in poly.interiors:
        h = np.asarray(interior.coords)
        ax.fill(h[:,0], h[:,1], facecolor=ax.get_facecolor(), edgecolor='none', zorder=zorder+0.01)

def _draw_line(ax, line, color, linewidth, alpha, linestyle, zorder):
    xy = np.asarray(line.coords)
    if len(xy) > 1:
        ax.plot(xy[:,0], xy[:,1], color=color, linewidth=linewidth,
                alpha=alpha, linestyle=linestyle, zorder=zorder)

def draw_geom(ax, geom, style):
    if geom is None or geom.is_empty: return
    poly_style = {k:v for k,v in style.items() if k in ('facecolor','edgecolor','linewidth','alpha','zorder')}
    line_style = {k:v for k,v in style.items() if k in ('color','linewidth','alpha','linestyle','zorder')}
    if isinstance(geom, Polygon): _draw_poly(ax, geom, **poly_style)
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms: _draw_poly(ax, p, **poly_style)
    elif isinstance(geom, LineString): _draw_line(ax, geom, **line_style)
    elif isinstance(geom, MultiLineString):
        for l in geom.geoms: _draw_line(ax, l, **line_style)
    elif isinstance(geom, GeometryCollection):
        for c in geom.geoms: draw_geom(ax, c, style)

def draw_map(ax, record, bounds):
    try:
        gpkg = map_gpkg(record)
    except FileNotFoundError as e:
        print(f'  [map missing] {e}')
        ax.set_facecolor('#f0f0ee'); return
    ax.set_facecolor('#f8faf8')
    total = 0
    for layer, style in MAP_LAYER_STYLES:
        gdf = load_layer(gpkg, layer, bounds)
        total += len(gdf)
        for geom in gdf.geometry:
            draw_geom(ax, geom, style)
    print(f'  [map] {record.location}: {total} features')

# ── fixed viewport centred on anchor ──────────────────────────────────
def make_bounds(bundle, view_half=45.0):
    af = bundle['frames'][bundle['anchor_index']]
    cx, cy = float(af['ego_x']), float(af['ego_y'])
    # shift slightly toward near corridor
    cor = np.asarray(bundle['corridor_xy'], np.float64)
    if cor.ndim == 2 and len(cor) > 5:
        near = cor[min(15, len(cor)-1)]
        d = near - np.array([cx, cy]); dist = float(np.linalg.norm(d))
        if dist > 1e-3:
            shift = min(dist * 0.35, 12.0)
            cx += d[0]/dist*shift; cy += d[1]/dist*shift
    return (cx-view_half, cy-view_half, cx+view_half, cy+view_half)

# ── agent boxes ────────────────────────────────────────────────────────
def draw_agent(ax, agent):
    x, y = float(agent['xy'][0]), float(agent['xy'][1])
    L = float(agent.get('length', 4.6)); W = float(agent.get('width', 1.9))
    rect = Rectangle((-L/2, -W/2), L, W,
                     facecolor='#f2a03a', edgecolor='#2d2d2d',
                     linewidth=0.55, alpha=0.88, zorder=28)
    rect.set_transform(Affine2D().rotate(float(agent['yaw'])).translate(x, y) + ax.transData)
    ax.add_patch(rect)

# ── gradient trace ─────────────────────────────────────────────────────
def draw_trace(ax, xy, color):
    xy = np.asarray(xy, np.float64)
    if len(xy) < 2: return
    for i in range(len(xy)-1):
        a = 0.25 + 0.75*(i/max(1, len(xy)-2))
        ax.plot(xy[i:i+2,0], xy[i:i+2,1], color=color, linewidth=2.8,
                alpha=a, solid_capstyle='round', zorder=30)
    ax.annotate('', xy=xy[-1], xytext=xy[-2],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.8, alpha=0.95), zorder=31)

# ── main figure ────────────────────────────────────────────────────────
METHOD_COLORS = {'fm_cbf':'#e74c3c', 'pure_fm':'#3498db', 'idm_proxy':'#2ecc71'}
METHOD_LABELS = {'fm_cbf':'FM+CBF (Ours)', 'pure_fm':'Pure FM', 'idm_proxy':'IDM-proxy'}
METHODS = ['fm_cbf', 'pure_fm', 'idm_proxy']
BUCKET_TITLES = {'lane_following':'(a) Lane Change', 'left_turn':'(b) Unprotected Left Turn'}
BUCKETS = ['lane_following', 'left_turn']

fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.0),
                          gridspec_kw=dict(hspace=0.26, wspace=0.04))

for row, bucket in enumerate(BUCKETS):
    bt = traces_df[traces_df['paper_bucket'] == bucket]
    # pick scene with most methods in traces
    scene_name = bt.groupby('scene_name')['method'].nunique().sort_values(ascending=False).index[0]
    record = next(r for r in selected if r.scene_name == scene_name and r.paper_bucket == bucket)
    bundle = get_bundle(record)
    print(f'\n[{bucket}] scene={scene_name}  loc={record.location}')

    corridor = np.asarray(bundle['corridor_xy'], np.float64)
    history  = np.asarray(bundle['history_xy'],  np.float64)
    af       = bundle['frames'][bundle['anchor_index']]
    start_xy = np.array([af['ego_x'], af['ego_y']], np.float64)
    agents   = af['agents']
    bounds   = make_bounds(bundle)
    print(f'  bounds={tuple(round(b,0) for b in bounds)}')

    near_goal = corridor[min(30, len(corridor)-1)] if len(corridor) > 1 else start_xy
    cor_vis   = corridor[:min(40, len(corridor))]

    for col, method in enumerate(METHODS):
        ax = axes[row, col]
        ax.set_aspect('equal'); ax.axis('off')
        color = METHOD_COLORS[method]

        draw_map(ax, record, bounds)

        if len(cor_vis) > 1:
            ax.plot(cor_vis[:,0], cor_vis[:,1], color='#1f2933', linestyle='--',
                    linewidth=1.1, alpha=0.55, zorder=24)
        if len(history) > 1:
            ax.plot(history[:,0], history[:,1], color='#d62728', linewidth=2.8,
                    alpha=0.95, solid_capstyle='round', zorder=25)
            ax.scatter(history[-1,0], history[-1,1], s=30, color='#d62728',
                       edgecolors='white', linewidths=0.7, zorder=26)

        sc = bt[(bt['scene_name']==scene_name) & (bt['method']==method)].sort_values('step')
        if len(sc) > 1:
            # cap at 50 steps to avoid large wandering blobs
            draw_trace(ax, sc[['x','y']].values[:50], color)

        for agent in agents[:10]:
            draw_agent(ax, agent)

        ax.scatter(start_xy[0], start_xy[1], s=85, marker='o', color=color,
                   edgecolors='black', linewidths=0.8, zorder=36)
        ax.scatter(near_goal[0], near_goal[1], s=180, marker='*', color=color,
                   edgecolors='black', linewidths=0.8, zorder=36)
        ax.set_xlim(bounds[0], bounds[2]); ax.set_ylim(bounds[1], bounds[3])

        m = metrics_df[(metrics_df.scene_name==scene_name) & (metrics_df.method==method)]
        safe  = 'OK'   if (len(m)==0 or m['collision'].values[0]==0) else 'COLL'
        goal  = 'GOAL' if (len(m)>0 and m['goal_reached'].values[0]==1) else 'MISS'
        ttg   = f"{m['time_to_goal_s'].values[0]:.1f}s" if len(m)>0 else 'N/A'
        ax.set_title(f'{METHOD_LABELS[method]}\n{safe}/{goal}/{ttg}', fontsize=9, pad=4)

    axes[row,0].text(0.0, 1.02, BUCKET_TITLES[bucket],
                     transform=axes[row,0].transAxes, fontsize=11,
                     fontweight='bold', ha='left', va='bottom')

legend_handles = [
    plt.Line2D([],[],color='#d62728',lw=3,label='Ego history'),
    plt.Line2D([],[],color='#1f2933',lw=1.2,ls='--',label='Expert future'),
    plt.Line2D([],[],color='#3498db',lw=3,label='Planner rollout'),
    mpatches.Patch(facecolor='#f2a03a',edgecolor='#2d2d2d',label='Neighbour vehicle'),
    mpatches.Patch(facecolor='#fbfbf7',edgecolor='#b8beb6',label='nuPlan lane map'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.01), fontsize=9, frameon=False)
plt.suptitle('nuPlan Mini — Map-Aware Trajectory Visualization',
             fontsize=14, fontweight='bold', y=1.05)
fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
print(f'\nSaved: {OUT_PATH}')
