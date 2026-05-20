from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from fmtorch.models.simple1d_unet import SimpleUNet1D

from .canonical import build_can_tf_5d


def _get_piecewise_sampler():
    from .sampler_5h import piecewise_sample_fm_5ch_with_cbf

    return piecewise_sample_fm_5ch_with_cbf


@dataclass
class BenchmarkConfig:
    root: Path
    seed: int = 7
    scenarios_per_type: int = 8
    episode_steps: int = 40
    horizon_steps: int = 16
    replay_dt_fallback: float = 0.1
    max_agents: int = 8
    goal_tol_m: float = 8.0
    safe_radius_m: float = 3.0
    road_half_width_m: float = 4.5
    road_margin_m: float = 0.5
    idm_desired_speed_mps: float = 8.0
    fm_noise_scale: float = 1.0
    fm_num_segments: int = 20
    fm_tau0: float = 0.50
    fm_tau1: float = 0.90
    fm_goal_guidance_weight: float = 1.20
    fm_start_guidance_weight: float = 0.40
    cbf_extra_margin_m: float = 0.25
    cbf_alpha: float = 8.0
    cbf_max_corr_norm: float = 2.0
    methods: Tuple[str, ...] = ("fm_cbf", "pure_fm", "idm_proxy")
    scenario_type_map: Optional[Dict[str, str]] = None
    model_ckpt: Optional[Path] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if self.scenario_type_map is None:
            self.scenario_type_map = {
                "lane_following": "following_lane_without_lead",
                "left_turn": "starting_left_turn",
            }
        if self.model_ckpt is None:
            self.model_ckpt = (
                self.root
                / "safeflow-nuplan-transfer"
                / "dmpc_fm_cbf"
                / "notebooks"
                / "cache"
                / "model_vel.pt"
            )

    @property
    def data_root(self) -> Path:
        return self.root / "data" / "cache" / "mini"

    @property
    def project_root(self) -> Path:
        return self.root / "safeflow-nuplan-transfer" / "dmpc_fm_cbf"

    @property
    def output_dir(self) -> Path:
        out = self.project_root / "notebooks" / "cache" / "nuplan_transfer"
        out.mkdir(parents=True, exist_ok=True)
        return out


@dataclass
class ScenarioRecord:
    db_path: Path
    scene_token: bytes
    anchor_lidar_pc_token: bytes
    goal_ego_pose_token: Optional[bytes]
    scenario_tag: str
    paper_bucket: str
    scene_name: str
    location: str
    map_version: str


@dataclass
class EpisodeMetrics:
    method: str
    paper_bucket: str
    scenario_tag: str
    scene_name: str
    time_to_goal_s: float
    goal_reached: int
    collision: int
    min_inter_agent_dist_m: float
    road_boundary_violations: int
    final_dist_to_goal_m: float
    num_steps: int


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def yaw_from_quaternion(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def polyline_distance(point_xy: np.ndarray, polyline_xy: np.ndarray) -> float:
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    pts = np.asarray(polyline_xy, dtype=np.float64)
    if len(pts) == 0:
        return float("inf")
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))

    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-9:
            d = np.linalg.norm(p - a)
        else:
            t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
            proj = a + t * ab
            d = np.linalg.norm(p - proj)
        best = min(best, float(d))
    return best


def nearest_agent_distance(ego_xy: np.ndarray, agents: Sequence[dict]) -> float:
    if not agents:
        return float("inf")
    dists = [float(np.linalg.norm(ego_xy - a["xy"])) for a in agents]
    return float(min(dists))


def select_nearest_agents(ego_xy: np.ndarray, agents: Sequence[dict], k: int) -> List[dict]:
    return sorted(agents, key=lambda a: np.linalg.norm(a["xy"] - ego_xy))[:k]


def make_runtime_state(frame: dict, prev_state: Optional[np.ndarray] = None) -> np.ndarray:
    x = float(frame["ego_x"])
    y = float(frame["ego_y"])
    theta = float(frame["ego_yaw"])
    v = float(math.hypot(frame["ego_vx"], frame["ego_vy"]))
    yaw_rate = 0.0 if prev_state is None else wrap_pi(theta - float(prev_state[2])) / max(frame["dt"], 1e-3)
    return np.array([x, y, theta, v, yaw_rate], dtype=np.float32)


def rollout_point_mass_to_target(state: np.ndarray, target_xy: np.ndarray, dt: float, speed_cap: float) -> np.ndarray:
    xy = state[:2].astype(np.float64)
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    delta = target - xy
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return state.copy()
    direction = delta / dist
    speed = min(speed_cap, dist / max(dt, 1e-3))
    next_xy = xy + direction * speed * dt
    theta = float(np.arctan2(direction[1], direction[0]))
    yaw_rate = wrap_pi(theta - float(state[2])) / max(dt, 1e-3)
    return np.array([next_xy[0], next_xy[1], theta, speed, yaw_rate], dtype=np.float32)


def load_model_or_none(cfg: BenchmarkConfig, device: str):
    if not cfg.model_ckpt or not cfg.model_ckpt.exists():
        print(f"[WARN] checkpoint not found: {cfg.model_ckpt}")
        return None

    model = SimpleUNet1D(
        time_emb_dim=128,
        hidden_dim=128,
        cond_dim=5,
        in_channels=5,
        out_channels=5,
    )
    ckpt = torch.load(cfg.model_ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def discover_scenarios(cfg: BenchmarkConfig) -> List[ScenarioRecord]:
    records: List[ScenarioRecord] = []
    for db_path in sorted(cfg.data_root.glob("*.db")):
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        log_row = cur.execute("SELECT location, map_version FROM log LIMIT 1").fetchone()
        location, map_version = log_row if log_row is not None else ("unknown", "unknown")

        for paper_bucket, nuplan_tag in cfg.scenario_type_map.items():
            rows = cur.execute(
                """
                SELECT DISTINCT lp.scene_token, st.lidar_pc_token, s.goal_ego_pose_token, s.name, st.type
                FROM scenario_tag st
                JOIN lidar_pc lp ON st.lidar_pc_token = lp.token
                JOIN scene s ON lp.scene_token = s.token
                WHERE st.type = ?
                ORDER BY s.name
                """,
                (nuplan_tag,),
            ).fetchall()
            for scene_token, anchor_token, goal_pose_token, scene_name, scenario_tag in rows:
                records.append(
                    ScenarioRecord(
                        db_path=db_path,
                        scene_token=scene_token,
                        anchor_lidar_pc_token=anchor_token,
                        goal_ego_pose_token=goal_pose_token,
                        scenario_tag=scenario_tag,
                        paper_bucket=paper_bucket,
                        scene_name=scene_name,
                        location=location,
                        map_version=map_version,
                    )
                )
        con.close()
    return records


def sample_balanced_scenarios(records: Sequence[ScenarioRecord], cfg: BenchmarkConfig) -> List[ScenarioRecord]:
    rng = random.Random(cfg.seed)
    chosen: List[ScenarioRecord] = []
    for bucket in cfg.scenario_type_map.keys():
        bucket_records = [r for r in records if r.paper_bucket == bucket]
        rng.shuffle(bucket_records)
        chosen.extend(bucket_records[: cfg.scenarios_per_type])
    return chosen


def load_scene_bundle(record: ScenarioRecord, cfg: BenchmarkConfig) -> dict:
    con = sqlite3.connect(str(record.db_path))
    cur = con.cursor()

    frame_rows = cur.execute(
        """
        SELECT lp.token, lp.timestamp,
               ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz,
               ep.vx, ep.vy
        FROM lidar_pc lp
        JOIN ego_pose ep ON lp.ego_pose_token = ep.token
        WHERE lp.scene_token = ?
        ORDER BY lp.timestamp
        """,
        (record.scene_token,),
    ).fetchall()

    goal_xy = None
    if record.goal_ego_pose_token is not None:
        row = cur.execute("SELECT x, y FROM ego_pose WHERE token = ?", (record.goal_ego_pose_token,)).fetchone()
        if row is not None:
            goal_xy = np.array(row, dtype=np.float32)

    frames = []
    last_ts = None
    for token, ts, x, y, qw, qx, qy, qz, vx, vy in frame_rows:
        dt = cfg.replay_dt_fallback if last_ts is None else max((ts - last_ts) * 1e-6, 1e-3)
        last_ts = ts
        yaw = yaw_from_quaternion(qw, qx, qy, qz)

        box_rows = cur.execute(
            """
            SELECT lb.track_token, lb.x, lb.y, lb.vx, lb.vy, lb.yaw, lb.width, lb.length, c.name
            FROM lidar_box lb
            JOIN track t ON lb.track_token = t.token
            JOIN category c ON t.category_token = c.token
            WHERE lb.lidar_pc_token = ? AND c.name = 'vehicle'
            """,
            (token,),
        ).fetchall()

        agents = [
            {
                "track_token": track_token,
                "xy": np.array([ax, ay], dtype=np.float32),
                "vx": float(avx),
                "vy": float(avy),
                "yaw": float(ayaw),
                "width": float(width),
                "length": float(length),
                "category": category,
            }
            for track_token, ax, ay, avx, avy, ayaw, width, length, category in box_rows
        ]

        frames.append(
            {
                "lidar_pc_token": token,
                "timestamp": ts,
                "dt": dt,
                "ego_x": float(x),
                "ego_y": float(y),
                "ego_yaw": yaw,
                "ego_vx": float(vx),
                "ego_vy": float(vy),
                "agents": agents,
            }
        )

    con.close()
    anchor_index = next(i for i, f in enumerate(frames) if f["lidar_pc_token"] == record.anchor_lidar_pc_token)
    if goal_xy is None:
        goal_xy = np.array([frames[-1]["ego_x"], frames[-1]["ego_y"]], dtype=np.float32)

    corridor = np.array([[f["ego_x"], f["ego_y"]] for f in frames[anchor_index:]], dtype=np.float32)
    return {
        "record": record,
        "frames": frames,
        "anchor_index": anchor_index,
        "goal_xy": goal_xy,
        "corridor_xy": corridor,
    }


class BasePlannerAdapter:
    planner_name = "base"

    def step(self, state: np.ndarray, frame: dict, bundle: dict, cfg: BenchmarkConfig) -> np.ndarray:
        raise NotImplementedError


class FMPlannerAdapter(BasePlannerAdapter):
    def __init__(self, model, device: str, use_cbf: bool):
        self.model = model
        self.device = device
        self.use_cbf = use_cbf
        self.planner_name = "fm_cbf" if use_cbf else "pure_fm"

    def _sample_path(self, state: np.ndarray, goal_xy: np.ndarray, agents: List[dict], cfg: BenchmarkConfig) -> np.ndarray:
        if self.model is None:
            raise FileNotFoundError(f"Model checkpoint missing: {cfg.model_ckpt}")
        piecewise_sample_fm_5ch_with_cbf = _get_piecewise_sampler()

        start_xy = state[:2].astype(np.float32)
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(2)
        tf = build_can_tf_5d(start_xy, goal_xy)
        goal_dist = float(tf["dist"])
        theta_can = wrap_pi(float(state[2]) - float(tf["phi"]))

        cond = np.array(
            [goal_dist, float(state[3]), math.cos(theta_can), math.sin(theta_can), float(state[4])],
            dtype=np.float32,
        )

        ellipses = []
        for agent in select_nearest_agents(start_xy, agents, cfg.max_agents):
            center_can = tf["w2c"](agent["xy"].reshape(1, 2))[0].astype(np.float32)
            radius = 0.5 * max(agent["width"], agent["length"]) + cfg.safe_radius_m * 0.25
            ellipses.append({"center": center_can, "radius": float(radius)})

        x_can, _, _ = piecewise_sample_fm_5ch_with_cbf(
            model_5ch=self.model,
            T_steps=cfg.horizon_steps,
            device=self.device,
            num_segments=cfg.fm_num_segments,
            noise_scale=cfg.fm_noise_scale,
            cond=cond,
            start_xy_norm=np.array([0.0, 0.0], dtype=np.float32),
            lock_start=True,
            goal_xy_norm=np.array([goal_dist, 0.0], dtype=np.float32),
            start_guidance_weight=cfg.fm_start_guidance_weight,
            goal_guidance_weight=cfg.fm_goal_guidance_weight,
            use_cbf=self.use_cbf,
            ellipses=ellipses,
            tau0=cfg.fm_tau0,
            tau1=cfg.fm_tau1,
            cbf_kwargs={
                "passes": 8,
                "extra_margin": cfg.cbf_extra_margin_m,
                "alpha": cfg.cbf_alpha,
                "max_corr_norm": cfg.cbf_max_corr_norm,
            },
        )
        path_can = np.asarray(x_can[:2, :].T, dtype=np.float32)
        return tf["c2w"](path_can).astype(np.float32)

    def step(self, state: np.ndarray, frame: dict, bundle: dict, cfg: BenchmarkConfig) -> np.ndarray:
        path_xy = self._sample_path(state, bundle["goal_xy"], frame["agents"], cfg)
        target_xy = path_xy[-1]
        for pt in path_xy[1:]:
            if np.linalg.norm(pt - state[:2]) > 0.5:
                target_xy = pt
                break
        return rollout_point_mass_to_target(state, target_xy, dt=float(frame["dt"]), speed_cap=cfg.idm_desired_speed_mps)


class IDMProxyPlannerAdapter(BasePlannerAdapter):
    planner_name = "idm_proxy"

    def step(self, state: np.ndarray, frame: dict, bundle: dict, cfg: BenchmarkConfig) -> np.ndarray:
        ego_xy = state[:2].astype(np.float32)
        corridor_xy = bundle["corridor_xy"]
        if len(corridor_xy) == 0:
            return state.copy()

        idx = int(np.argmin(np.linalg.norm(corridor_xy - ego_xy[None, :], axis=1)))
        lookahead_idx = min(idx + 5, len(corridor_xy) - 1)
        target_xy = corridor_xy[lookahead_idx]

        desired_speed = cfg.idm_desired_speed_mps
        nearest = select_nearest_agents(ego_xy, frame["agents"], 1)
        if nearest:
            lead_dist = float(np.linalg.norm(nearest[0]["xy"] - ego_xy))
            if lead_dist < 20.0:
                desired_speed = min(desired_speed, max(0.0, lead_dist - cfg.safe_radius_m))
        return rollout_point_mass_to_target(state, target_xy, dt=float(frame["dt"]), speed_cap=desired_speed)


class OfficialNuPlanPlannerAdapter(BasePlannerAdapter):
    def __init__(self, planner_name: str):
        if planner_name not in {"official_idm", "official_pdm_closed"}:
            raise ValueError(f"Unsupported official planner: {planner_name}")
        self.planner_name = planner_name

    def build_nuplan_planner(self, **kwargs):
        from .nuplan_sdk_adapter import build_idm_planner, build_pdm_closed_planner

        if self.planner_name == "official_idm":
            return build_idm_planner(**kwargs)
        if self.planner_name == "official_pdm_closed":
            return build_pdm_closed_planner(**kwargs)
        raise ValueError(self.planner_name)

    def build_simulation_command(self, simulation_cfg):
        from .nuplan_sdk_adapter import build_run_simulation_command

        return build_run_simulation_command(simulation_cfg)

    def step(self, state: np.ndarray, frame: dict, bundle: dict, cfg: BenchmarkConfig) -> np.ndarray:
        raise NotImplementedError(
            f"{self.planner_name} is a package-level hook only. Instantiate the official nuPlan planner via "
            "build_nuplan_planner() and route step() through the SDK runtime."
        )


def build_planner(method: str, cfg: BenchmarkConfig, model=None, device: str = "cpu") -> BasePlannerAdapter:
    if method == "fm_cbf":
        return FMPlannerAdapter(model=model, device=device, use_cbf=True)
    if method == "pure_fm":
        return FMPlannerAdapter(model=model, device=device, use_cbf=False)
    if method == "idm_proxy":
        return IDMProxyPlannerAdapter()
    if method in {"official_idm", "official_pdm_closed"}:
        return OfficialNuPlanPlannerAdapter(method)
    raise ValueError(f"Unknown method: {method}")


def run_episode(bundle: dict, planner: BasePlannerAdapter, cfg: BenchmarkConfig) -> Tuple[EpisodeMetrics, pd.DataFrame]:
    frames = bundle["frames"]
    anchor_idx = bundle["anchor_index"]
    goal_xy = bundle["goal_xy"].astype(np.float32)
    corridor_xy = bundle["corridor_xy"].astype(np.float32)
    record = bundle["record"]

    state = make_runtime_state(frames[anchor_idx])
    traces = []
    min_dist = float("inf")
    road_violations = 0
    collision = 0
    reached = 0
    time_to_goal = cfg.episode_steps * cfg.replay_dt_fallback

    for step in range(cfg.episode_steps):
        frame_idx = min(anchor_idx + step, len(frames) - 1)
        frame = dict(frames[frame_idx])
        frame["agents"] = select_nearest_agents(state[:2], frame["agents"], cfg.max_agents)
        dt = float(frame["dt"])
        state = planner.step(state, frame, bundle, cfg)

        dist_goal = float(np.linalg.norm(state[:2] - goal_xy))
        agent_dist = nearest_agent_distance(state[:2], frame["agents"])
        min_dist = min(min_dist, agent_dist)
        road_err = polyline_distance(state[:2], corridor_xy)
        road_violations += int(road_err > (cfg.road_half_width_m + cfg.road_margin_m))
        if agent_dist < cfg.safe_radius_m:
            collision = 1

        traces.append(
            {
                "step": step,
                "x": float(state[0]),
                "y": float(state[1]),
                "theta": float(state[2]),
                "speed": float(state[3]),
                "dist_to_goal_m": dist_goal,
                "nearest_agent_dist_m": agent_dist,
                "road_err_m": road_err,
            }
        )

        if dist_goal < cfg.goal_tol_m:
            reached = 1
            time_to_goal = float(step + 1) * dt
            break

    trace_df = pd.DataFrame(traces)
    final_dist = float(trace_df["dist_to_goal_m"].iloc[-1]) if len(trace_df) else float(np.linalg.norm(state[:2] - goal_xy))
    metrics = EpisodeMetrics(
        method=planner.planner_name,
        paper_bucket=record.paper_bucket,
        scenario_tag=record.scenario_tag,
        scene_name=record.scene_name,
        time_to_goal_s=float(time_to_goal),
        goal_reached=int(reached),
        collision=int(collision),
        min_inter_agent_dist_m=float(min_dist),
        road_boundary_violations=int(road_violations),
        final_dist_to_goal_m=float(final_dist),
        num_steps=int(len(trace_df)),
    )
    return metrics, trace_df


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(["paper_bucket", "method"], as_index=False)
        .agg(
            time_to_goal_s=("time_to_goal_s", "mean"),
            collision_rate=("collision", "mean"),
            goal_reached_rate=("goal_reached", "mean"),
            min_inter_agent_dist_m=("min_inter_agent_dist_m", "mean"),
            road_boundary_violations=("road_boundary_violations", "mean"),
            final_dist_to_goal_m=("final_dist_to_goal_m", "mean"),
        )
        .sort_values(["paper_bucket", "method"])
    )


def run_benchmark(cfg: BenchmarkConfig, methods: Optional[Sequence[str]] = None, device: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = tuple(methods or cfg.methods)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = load_model_or_none(cfg, device)
    scenarios = sample_balanced_scenarios(discover_scenarios(cfg), cfg)

    metrics_rows = []
    trace_rows = []
    scene_cache: Dict[Tuple[str, bytes], dict] = {}
    planner_cache: Dict[str, BasePlannerAdapter] = {}

    for record in scenarios:
        key = (str(record.db_path), record.anchor_lidar_pc_token)
        if key not in scene_cache:
            scene_cache[key] = load_scene_bundle(record, cfg)
        bundle = scene_cache[key]

        for method in methods:
            if method not in planner_cache:
                planner_cache[method] = build_planner(method, cfg, model=model, device=device)
            planner = planner_cache[method]
            metrics, trace_df = run_episode(bundle, planner, cfg)
            metrics_rows.append(asdict(metrics))
            if len(trace_df):
                trace_df = trace_df.copy()
                trace_df["method"] = planner.planner_name
                trace_df["scene_name"] = record.scene_name
                trace_df["paper_bucket"] = record.paper_bucket
                trace_rows.append(trace_df)

    metrics_df = pd.DataFrame(metrics_rows)
    traces_df = pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()
    return metrics_df, traces_df
