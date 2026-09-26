#!/usr/bin/env python3
"""List simple nuPlan mini scenarios with at least one nearby vehicle."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional


DEFAULT_MINI_ROOT = (
    "/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/"
    "nuplan-v1.1_mini/data/cache/mini"
)


@dataclass(frozen=True)
class ScenarioCandidate:
    token: str
    scenario_type: str
    map_name: str
    db_file: str
    nearby_vehicles: int
    path_length: float
    direct_distance: float
    path_ratio: float
    yaw_change: float
    x: float
    y: float
    lead_longitudinal: Optional[float]
    lead_lateral: Optional[float]


def _yaw_from_quaternion(qw: float, qz: float) -> float:
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _scenario_points(con: sqlite3.Connection, lidar_pc_token: bytes, steps: int) -> List[tuple[float, float, float]]:
    points: List[tuple[float, float, float]] = []
    lidar_pc = con.execute(
        "select token, next_token, ego_pose_token from lidar_pc where token=?",
        (lidar_pc_token,),
    ).fetchone()
    for _ in range(steps):
        if lidar_pc is None:
            break
        ego_pose = con.execute(
            "select x, y, qw, qz from ego_pose where token=?",
            (lidar_pc["ego_pose_token"],),
        ).fetchone()
        if ego_pose is None:
            break
        points.append(
            (
                float(ego_pose["x"]),
                float(ego_pose["y"]),
                _yaw_from_quaternion(float(ego_pose["qw"]), float(ego_pose["qz"])),
            )
        )
        if lidar_pc["next_token"] is None:
            break
        lidar_pc = con.execute(
            "select token, next_token, ego_pose_token from lidar_pc where token=?",
            (lidar_pc["next_token"],),
        ).fetchone()
    return points


def _nearby_vehicle_count(
    con: sqlite3.Connection, lidar_pc_token: bytes, ego_x: float, ego_y: float, radius_m: float
) -> int:
    radius_sq = radius_m * radius_m
    return int(
        con.execute(
            """
            select count(*) as count
            from lidar_box lb
            join track tr on tr.token = lb.track_token
            join category ca on ca.token = tr.category_token
            where lb.lidar_pc_token = ?
              and ca.name = 'vehicle'
              and ((lb.x - ?) * (lb.x - ?) + (lb.y - ?) * (lb.y - ?)) <= ?
            """,
            (lidar_pc_token, ego_x, ego_x, ego_y, ego_y, radius_sq),
        ).fetchone()["count"]
    )


def _lead_vehicle(
    con: sqlite3.Connection, lidar_pc_token: bytes, ego_x: float, ego_y: float, ego_yaw: float
) -> Optional[tuple[float, float]]:
    forward = (math.cos(ego_yaw), math.sin(ego_yaw))
    left = (-math.sin(ego_yaw), math.cos(ego_yaw))
    best: Optional[tuple[float, float]] = None
    for row in con.execute(
        """
        select lb.x, lb.y
        from lidar_box lb
        join track tr on tr.token = lb.track_token
        join category ca on ca.token = tr.category_token
        where lb.lidar_pc_token = ?
          and ca.name = 'vehicle'
        """,
        (lidar_pc_token,),
    ):
        dx = float(row["x"]) - ego_x
        dy = float(row["y"]) - ego_y
        longitudinal = dx * forward[0] + dy * forward[1]
        lateral = dx * left[0] + dy * left[1]
        if 4.0 <= longitudinal <= 30.0 and abs(lateral) <= 3.0:
            candidate = (longitudinal, lateral)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best


def _iter_db_candidates(
    db_path: str,
    steps: int,
    radius_m: float,
    map_name: Optional[str],
    scenario_types: Optional[set[str]],
    require_lead: bool,
) -> Iterable[ScenarioCandidate]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select lower(hex(st.lidar_pc_token)) as token,
                   st.lidar_pc_token as raw_token,
                   st.type as scenario_type,
                   l.map_version as map_name
            from scenario_tag st
            join lidar_pc lp on lp.token = st.lidar_pc_token
            join ego_pose ep on ep.token = lp.ego_pose_token
            join log l on l.token = ep.log_token
            """
        ).fetchall()
        for row in rows:
            if map_name and row["map_name"] != map_name:
                continue
            if scenario_types and row["scenario_type"] not in scenario_types:
                continue

            points = _scenario_points(con, row["raw_token"], steps)
            if len(points) < 5:
                continue

            path_length = sum(
                math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
                for index in range(1, len(points))
            )
            direct_distance = math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1])
            yaw_change = sum(abs(_wrap_pi(points[index][2] - points[index - 1][2])) for index in range(1, len(points)))
            ego_x, ego_y, _ = points[0]
            nearby_vehicles = _nearby_vehicle_count(con, row["raw_token"], ego_x, ego_y, radius_m)
            if nearby_vehicles < 1:
                continue
            lead = _lead_vehicle(con, row["raw_token"], ego_x, ego_y, points[0][2])
            if require_lead and lead is None:
                continue

            yield ScenarioCandidate(
                token=row["token"],
                scenario_type=row["scenario_type"],
                map_name=row["map_name"],
                db_file=os.path.basename(db_path),
                nearby_vehicles=nearby_vehicles,
                path_length=path_length,
                direct_distance=direct_distance,
                path_ratio=path_length / max(direct_distance, 1e-6),
                yaw_change=yaw_change,
                x=ego_x,
                y=ego_y,
                lead_longitudinal=lead[0] if lead else None,
                lead_lateral=lead[1] if lead else None,
            )
    finally:
        con.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini-root", default=os.environ.get("NUPLAN_MINI_DB_ROOT", DEFAULT_MINI_ROOT))
    parser.add_argument("--db-file", action="append", default=[])
    parser.add_argument("--map-name", default=None)
    parser.add_argument("--scenario-types", default="medium_magnitude_speed,following_lane_with_lead,low_magnitude_speed")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--radius-m", type=float, default=40.0)
    parser.add_argument("--require-lead", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db_paths = args.db_file or sorted(glob.glob(os.path.join(args.mini_root, "*.db")))
    scenario_types = {item.strip() for item in args.scenario_types.split(",") if item.strip()}
    candidates: List[ScenarioCandidate] = []
    skipped: List[tuple[str, str]] = []

    for db_path in db_paths:
        try:
            candidates.extend(
                _iter_db_candidates(
                    db_path,
                    args.steps,
                    args.radius_m,
                    args.map_name,
                    scenario_types,
                    args.require_lead,
                )
            )
        except sqlite3.DatabaseError as exc:
            skipped.append((os.path.basename(db_path), str(exc)))

    candidates.sort(
        key=lambda candidate: (
            candidate.path_ratio,
            candidate.yaw_change,
            abs(candidate.nearby_vehicles - 1),
            -candidate.direct_distance,
        )
    )

    if skipped:
        print("Skipped malformed/unreadable databases:")
        for db_file, reason in skipped:
            print(f"  {db_file}: {reason}")
        print()

    for candidate in candidates[: args.limit]:
        print(
            f"{candidate.token} | {candidate.scenario_type:<30} | {candidate.map_name:<28} "
            f"| veh={candidate.nearby_vehicles:2d} | ratio={candidate.path_ratio:.2f} "
            f"| yaw={candidate.yaw_change:.2f} | path={candidate.path_length:.1f} "
            f"| direct={candidate.direct_distance:.1f} | {candidate.db_file} "
            f"| xy=({candidate.x:.1f},{candidate.y:.1f})"
            + (
                f" | lead=({candidate.lead_longitudinal:.1f},{candidate.lead_lateral:.1f})"
                if candidate.lead_longitudinal is not None and candidate.lead_lateral is not None
                else ""
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
