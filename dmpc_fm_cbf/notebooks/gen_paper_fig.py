"""Generate paper-quality 2×2 figure: method rows × scenario cols, with compact map."""
import math, random, sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from pyproj import Transformer
from shapely.geometry import (GeometryCollection, LineString,
                               MultiLineString, MultiPolygon, Polygon)

ROOT      = Path('/new_world/cockatiel/ra').resolve()
DATA_ROOT = ROOT / 'data' / 'cache' / 'mini'
MAP_ROOT  = ROOT / 'maps'
CACHE_DIR = ROOT / 'safeflow-nuplan-transfer/dmpc_fm_cbf/notebooks/cache/nuplan_transfer'

traces_df  = pd.read_csv(CACHE_DIR / 'traces.csv')
print(f'Loaded {len(traces_df)} trace rows')

# ── scenario data loading ──────────────────────────────────────────────
def yaw_q(qw, qx, qy, qz):
    return float(np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))

@dataclass
class Rec:
    db_path: Path; scene_token: bytes; anchor_token: bytes
    goal_token: Optional[bytes]; paper_bucket: str; scene_name: str
    location: str; map_version: str

def discover():
    out = []
    for db in sorted(DATA_ROOT.glob('*.db')):
        try:
            with sqlite3.connect(str(db)) as con:
                cur = con.cursor()
                row = cur.execute('SELECT location,map_version FROM log LIMIT 1').fetchone()
                if not row: continue
                loc, mv = row
                for bucket, tag in [('lane_following','following_lane_without_lead'),
                                     ('left_turn','starting_left_turn')]:
                    for st,at,gt,sn,_ in cur.execute('''
                        SELECT DISTINCT lp.scene_token,st2.lidar_pc_token,
                               s.goal_ego_pose_token,s.name,st2.type
                        FROM scenario_tag st2 JOIN lidar_pc lp ON st2.lidar_pc_token=lp.token
                        JOIN scene s ON lp.scene_token=s.token
                        WHERE st2.type=? ORDER BY s.name''',(tag,)).fetchall():
                        out.append(Rec(db,st,at,gt,bucket,sn,loc,mv))
        except: pass
    return out

def load_bundle(r):
    con = sqlite3.connect(str(r.db_path)); cur = con.cursor()
    rows = cur.execute('''SELECT lp.token,lp.timestamp,ep.x,ep.y,
            ep.qw,ep.qx,ep.qy,ep.qz,ep.vx,ep.vy
        FROM lidar_pc lp JOIN ego_pose ep ON lp.ego_pose_token=ep.token
        WHERE lp.scene_token=? ORDER BY lp.timestamp''', (r.scene_token,)).fetchall()
    gxy = None
    if r.goal_token:
        row = cur.execute('SELECT x,y FROM ego_pose WHERE token=?',(r.goal_token,)).fetchone()
        if row: gxy = np.array(row, np.float32)
    frames = []; last_ts = None
    for tok,ts,x,y,qw,qx,qy,qz,vx,vy in rows:
        dt = 0.1 if last_ts is None else max((ts-last_ts)*1e-6, 1e-3); last_ts = ts
        agents = [{'xy':np.array([ax,ay],np.float32),'yaw':float(ay2),
                   'width':float(w),'length':float(l)}
                  for _,ax,ay,_,_,ay2,w,l,_ in cur.execute('''
                      SELECT lb.track_token,lb.x,lb.y,lb.vx,lb.vy,lb.yaw,lb.width,lb.length,c.name
                      FROM lidar_box lb JOIN track t ON lb.track_token=t.token
                      JOIN category c ON t.category_token=c.token
                      WHERE lb.lidar_pc_token=? AND c.name='vehicle' ''',(tok,)).fetchall()]
        frames.append({'tok':tok,'dt':dt,'ego_x':float(x),'ego_y':float(y),
                       'ego_yaw':yaw_q(qw,qx,qy,qz),'agents':agents})
    con.close()
    aidx = next(i for i,f in enumerate(frames) if f['tok']==r.anchor_token)
    if gxy is None:
        gf = min(aidx+80, len(frames)-1)
        gxy = np.array([frames[gf]['ego_x'],frames[gf]['ego_y']],np.float32)
    corridor = np.array([[f['ego_x'],f['ego_y']] for f in frames[aidx:]],np.float32)
    history  = np.array([[f['ego_x'],f['ego_y']] for f in frames[:aidx+1]],np.float32)
    return dict(frames=frames, aidx=aidx, gxy=gxy, corridor=corridor, history=history)

rng = random.Random(7)
all_recs = discover()
selected = []
for b in ['lane_following','left_turn']:
    pool = [r for r in all_recs if r.paper_bucket==b]; rng.shuffle(pool); selected.extend(pool[:8])

BCACHE = {}
def get_bundle(r):
    k = (str(r.db_path), r.anchor_token)
    if k not in BCACHE: BCACHE[k] = load_bundle(r)
    return BCACHE[k]

# ── map rendering ──────────────────────────────────────────────────────
CLEAN_POLY_LAYERS = [
    ('generic_drivable_areas',
     dict(facecolor='#dfe8de', edgecolor='none', linewidth=0, alpha=1.0, zorder=0)),
    ('road_segments',
     dict(facecolor='#eef0eb', edgecolor='#c2c8bf', linewidth=0.3, alpha=1.0, zorder=1)),
    ('lanes_polygons',
     dict(facecolor='#f8f8f4', edgecolor='#adb5ab', linewidth=0.45, alpha=1.0, zorder=2)),
    ('gen_lane_connectors_scaled_width_polygons',
     dict(facecolor='#f4f4ef', edgecolor='#c0c5be', linewidth=0.3, alpha=0.9, zorder=2)),
    ('intersections',
     dict(facecolor='#e8dfd0', edgecolor='none', linewidth=0, alpha=0.75, zorder=3)),
    ('crosswalks',
     dict(facecolor='#edd898', edgecolor='#b89040', linewidth=0.35, alpha=0.7, zorder=4)),
]
CLEAN_LINE_LAYERS = [
    ('baseline_paths',
     dict(color='#808898', linewidth=0.45, alpha=0.55, linestyle='--', zorder=5)),
    ('boundaries',
     dict(color='#606870', linewidth=0.45, alpha=0.45, linestyle='-',  zorder=6)),
]
MC = {}

def _gpkg(r):
    for base in [MAP_ROOT/r.map_version, MAP_ROOT/r.location]:
        c = sorted(base.glob('*/map.gpkg'))
        if c: return c[0]
    raise FileNotFoundError

def _meta(g):
    k = ('m',str(g))
    if k not in MC:
        m = gpd.read_file(g, layer='meta')
        MC[k] = dict(zip(m['key'],m['value']))
    return MC[k]

def _bbox_geo(bounds, proj, geo='epsg:4326'):
    t = Transformer.from_crs(proj, geo, always_xy=True)
    pts = [(bounds[0],bounds[1]),(bounds[0],bounds[3]),
           (bounds[2],bounds[1]),(bounds[2],bounds[3])]
    ll = np.array([t.transform(x,y) for x,y in pts])
    return float(ll[:,0].min()),float(ll[:,1].min()),float(ll[:,0].max()),float(ll[:,1].max())

def _lyr(g, layer, bounds):
    m = _meta(g); proj = m.get('projectedCoordSystem','epsg:32617'); geo = m.get('geographicCoordSystem','epsg:4326')
    bbox = _bbox_geo(bounds, proj, geo)
    k = ('l',str(g),layer,tuple(round(v,1) for v in bounds))
    if k in MC: return MC[k]
    try: gdf = gpd.read_file(g, layer=layer, bbox=bbox)
    except: gdf = gpd.GeoDataFrame(geometry=[], crs=geo)
    if len(gdf):
        if gdf.crs is None: gdf = gdf.set_crs(geo)
        gdf = gdf.to_crs(proj)
    MC[k] = gdf; return gdf

def _dpoly(ax, geom, style):
    if isinstance(geom, Polygon):
        xy = np.asarray(geom.exterior.coords)
        ax.fill(xy[:,0],xy[:,1],**{k:style[k] for k in style if k!='linestyle'})
        for h in geom.interiors:
            hxy=np.asarray(h.coords); ax.fill(hxy[:,0],hxy[:,1],fc=ax.get_facecolor(),ec='none',zorder=style['zorder']+0.01)
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms: _dpoly(ax, p, style)
    elif isinstance(geom, GeometryCollection):
        for c in geom.geoms:
            if isinstance(c,(Polygon,MultiPolygon)): _dpoly(ax,c,style)

def _dline(ax, geom, style):
    kw = {k:style[k] for k in ('color','linewidth','alpha','linestyle','zorder') if k in style}
    if isinstance(geom, LineString):
        xy = np.asarray(geom.coords)
        if len(xy)>1: ax.plot(xy[:,0],xy[:,1],**kw)
    elif isinstance(geom, MultiLineString):
        for l in geom.geoms: _dline(ax,l,style)
    elif isinstance(geom, GeometryCollection):
        for c in geom.geoms: _dline(ax,c,style)

def draw_map(ax, r, bounds):
    try: g = _gpkg(r)
    except FileNotFoundError: ax.set_facecolor('#e8e8e8'); return
    ax.set_facecolor('#dfe8de')
    for layer, style in CLEAN_POLY_LAYERS:
        for geom in _lyr(g, layer, bounds).geometry: _dpoly(ax, geom, style)
    for layer, style in CLEAN_LINE_LAYERS:
        for geom in _lyr(g, layer, bounds).geometry: _dline(ax, geom, style)

def fig_bounds(bundle, vh=42.0):
    af = bundle['frames'][bundle['aidx']]
    cx, cy = float(af['ego_x']), float(af['ego_y'])
    cor = np.asarray(bundle['corridor'], np.float64)
    if cor.ndim==2 and len(cor)>5:
        near = cor[min(12,len(cor)-1)]; d = near-np.array([cx,cy]); dist = float(np.linalg.norm(d))
        if dist>1e-3: s=min(dist*0.3,10.0); cx+=d[0]/dist*s; cy+=d[1]/dist*s
    return (cx-vh, cy-vh, cx+vh, cy+vh)

# ── figure ─────────────────────────────────────────────────────────────
METHOD_ORDER = ['idm_proxy', 'fm_cbf']
METHOD_LABEL = {'idm_proxy':'IDM-proxy', 'fm_cbf':'Flow Planner (ours)'}
BUCKET_ORDER = ['lane_following', 'left_turn']
BUCKET_LABEL = {'lane_following':'(a) Lane Change', 'left_turn':'(b) Unprotected Left Turn'}
COLOR_MAP    = {'idm_proxy':'#27ae60', 'fm_cbf':'#e74c3c'}
HIST_COLOR   = '#8b1a1a'

fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.5),
                          gridspec_kw=dict(hspace=0.08, wspace=0.06))

for col, bucket in enumerate(BUCKET_ORDER):
    bt = traces_df[traces_df['paper_bucket']==bucket]
    scene_name = bt.groupby('scene_name')['method'].nunique().sort_values(ascending=False).index[0]
    record = next(r for r in selected if r.scene_name==scene_name and r.paper_bucket==bucket)
    bundle = get_bundle(record)
    bounds = fig_bounds(bundle)
    print(f'[{bucket}] scene={scene_name}  loc={record.location}  bounds={tuple(round(b,0) for b in bounds)}')

    corridor = np.asarray(bundle['corridor'], np.float64)
    history  = np.asarray(bundle['history'],  np.float64)
    af       = bundle['frames'][bundle['aidx']]
    start_xy = np.array([af['ego_x'], af['ego_y']], np.float64)
    agents   = af['agents']
    near_goal= corridor[min(28,len(corridor)-1)] if len(corridor)>1 else start_xy
    cor_vis  = corridor[:min(35,len(corridor))]

    for row, method in enumerate(METHOD_ORDER):
        ax = axes[row,col]; ax.set_aspect('equal'); ax.axis('off')
        color = COLOR_MAP[method]

        draw_map(ax, record, bounds)

        if len(cor_vis)>1:
            ax.plot(cor_vis[:,0],cor_vis[:,1],color='#3a4a5a',lw=1.1,ls='--',alpha=0.45,zorder=20)

        if len(history)>1:
            hv = history[-min(15,len(history)):]
            ax.plot(hv[:,0],hv[:,1],color=HIST_COLOR,lw=2.2,alpha=0.88,solid_capstyle='round',zorder=25)
            ax.scatter(hv[-1,0],hv[-1,1],s=24,color=HIST_COLOR,edgecolors='white',linewidths=0.6,zorder=26)

        sc = bt[(bt['scene_name']==scene_name)&(bt['method']==method)].sort_values('step')
        if len(sc)>1:
            txy = sc[['x','y']].values[:50]
            for i in range(len(txy)-1):
                a = 0.28+0.72*(i/max(1,len(txy)-2))
                ax.plot(txy[i:i+2,0],txy[i:i+2,1],color=color,lw=2.6,alpha=a,
                        solid_capstyle='round',zorder=30)
            ax.annotate('',xy=txy[-1],xytext=txy[-2],
                        arrowprops=dict(arrowstyle='-|>',color=color,lw=2.6,alpha=0.96),zorder=31)

        for agent in agents[:10]:
            x,y = float(agent['xy'][0]),float(agent['xy'][1])
            L=float(agent.get('length',4.6)); W=float(agent.get('width',1.9))
            rect=Rectangle((-L/2,-W/2),L,W,facecolor='#f0a028',edgecolor='#282828',
                            linewidth=0.55,alpha=0.88,zorder=28)
            rect.set_transform(Affine2D().rotate(float(agent.get('yaw',0))).translate(x,y)+ax.transData)
            ax.add_patch(rect)

        ax.scatter(start_xy[0],start_xy[1],s=75,marker='o',color=color,edgecolors='k',linewidths=0.8,zorder=36)
        ax.scatter(near_goal[0],near_goal[1],s=165,marker='*',color=color,edgecolors='k',linewidths=0.7,zorder=36)
        ax.set_xlim(bounds[0],bounds[2]); ax.set_ylim(bounds[1],bounds[3])

        if row==0:
            ax.set_title(BUCKET_LABEL[bucket], fontsize=11, fontweight='bold', pad=7)

for row, method in enumerate(METHOD_ORDER):
    axes[row,0].annotate(METHOD_LABEL[method], xy=(0,0.5), xycoords='axes fraction',
        xytext=(-8,0), textcoords='offset points',
        fontsize=10, fontweight='bold', ha='right', va='center', rotation=90)

legend_handles = [
    plt.Line2D([],[],color=HIST_COLOR,lw=2.2,           label='Ego history'),
    plt.Line2D([],[],color='#3a4a5a', lw=1.1, ls='--',  label='Expert future'),
    plt.Line2D([],[],color='#e74c3c', lw=2.6,            label='Flow Planner rollout'),
    plt.Line2D([],[],color='#27ae60', lw=2.6,            label='IDM-proxy rollout'),
    mpatches.Patch(fc='#f0a028', ec='#282828',           label='Neighbour vehicle'),
    mpatches.Patch(fc='#f8f8f4', ec='#adb5ab',           label='nuPlan lane map'),
]
fig.legend(handles=legend_handles, loc='upper center', ncol=3,
           bbox_to_anchor=(0.5,1.03), fontsize=8.5, frameon=False, columnspacing=1.2)
plt.suptitle('nuPlan Mini — Closed-Loop Trajectory Visualization',
             fontsize=13, fontweight='bold', y=1.08)

out = CACHE_DIR / 'trajectory_paper_fig.png'
fig.savefig(out, dpi=220, bbox_inches='tight')
print('saved:', out)
