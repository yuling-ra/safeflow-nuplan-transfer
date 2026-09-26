#!/usr/bin/env python3
"""Remove serialized planners from nuBoard simulation logs.

nuBoard only consumes the scenario and simulation history.  Serializing a
PyTorch planner embeds its model in every log and can make a CPU nuBoard
process unable to deserialize a log produced on CUDA.
"""

from __future__ import annotations

import argparse
import io
import lzma
import os
import pathlib
import pickle
import shutil
import sys
import warnings
from typing import Any

import msgpack
import torch

from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.simulation.simulation_log import SimulationLog


class _MapsDbPlaceholder:
    """Avoid eagerly loading all map layers from a planner that will be discarded."""

    def __init__(self, *_: Any) -> None:
        pass


class _PlannerSkippingUnpickler(pickle.Unpickler):
    """Replace heavyweight planner dependencies while reading legacy logs."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "nuplan.database.maps_db.gpkg_mapsdb" and name == "GPKGMapsDB":
            return _MapsDbPlaceholder
        return super().find_class(module, name)


def _simulation_directory(nuboard_path: pathlib.Path) -> pathlib.Path:
    nuboard_file = NuBoardFile.load_nuboard_file(nuboard_path)
    if nuboard_file.simulation_folder is None:
        raise RuntimeError(f"Simulation logging is disabled in {nuboard_path}")

    local_directory = nuboard_path.parent / nuboard_file.simulation_folder
    if local_directory.is_dir():
        return local_directory

    return pathlib.Path(nuboard_file.simulation_main_path) / nuboard_file.simulation_folder


def _unpack_pickle(file_path: pathlib.Path) -> bytes:
    with lzma.open(file_path, "rb") as stream:
        data = stream.read()

    log_type = SimulationLog.simulation_log_type(file_path)
    return msgpack.unpackb(data) if log_type == "msgpack" else data


def _load_legacy_log(file_path: pathlib.Path) -> SimulationLog:
    # These tensors belong to the planner that is removed immediately after
    # loading.  A shared dummy storage avoids restoring hundreds of CUDA model
    # tensors merely to discard them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dummy_storage = torch.FloatStorage(10_000_000)
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda _: dummy_storage
    try:
        serialized = _unpack_pickle(file_path)
        return _PlannerSkippingUnpickler(io.BytesIO(serialized)).load()
    finally:
        torch.storage._load_from_bytes = original_loader


def _write_log(log: SimulationLog, destination: pathlib.Path) -> None:
    serialized = pickle.dumps(log, protocol=pickle.HIGHEST_PROTOCOL)
    if SimulationLog.simulation_log_type(destination) == "msgpack":
        serialized = msgpack.packb(serialized)
    destination.write_bytes(lzma.compress(serialized, preset=0))


def _log_files(simulation_directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in simulation_directory.rglob("*")
        if path.is_file() and tuple(path.suffixes[-2:]) in ((".msgpack", ".xz"), (".pkl", ".xz"))
    )


def sanitize_log(file_path: pathlib.Path, simulation_directory: pathlib.Path, backup_root: pathlib.Path) -> bool:
    relative_path = file_path.relative_to(simulation_directory)
    backup_path = backup_root / relative_path

    source_path = file_path
    if backup_path.exists():
        try:
            current_log = SimulationLog.load_data(file_path)
            if current_log.planner is None and current_log.simulation_history.map_api is None:
                print(f"Already sanitized: {file_path}")
                return False
        except (AttributeError, RuntimeError):
            pass
        source_path = backup_path

    print(f"Sanitizing: {file_path}", flush=True)
    log = _load_legacy_log(source_path)
    if getattr(log, "planner", None) is None and log.simulation_history.map_api is None:
        print(f"Planner already omitted: {file_path}")
        return False

    history_size = len(log.simulation_history.data)
    log.planner = None
    log.simulation_history.map_api = None

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)

    temporary_path = backup_path.parent / f"sanitized-{file_path.name}"
    try:
        _write_log(log, temporary_path)
        validated = SimulationLog.load_data(temporary_path)
        if (
            validated.planner is not None
            or validated.simulation_history.map_api is not None
            or len(validated.simulation_history.data) != history_size
        ):
            raise RuntimeError(f"Sanitized log validation failed: {file_path}")
        os.replace(temporary_path, file_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Sanitized log: {file_path} ({backup_path.stat().st_size} -> {file_path.stat().st_size} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nuboard_file", type=pathlib.Path)
    args = parser.parse_args()

    nuboard_path = args.nuboard_file.expanduser().resolve()
    if not nuboard_path.is_file():
        parser.error(f"NuBoard file does not exist: {nuboard_path}")

    simulation_directory = _simulation_directory(nuboard_path)
    if not simulation_directory.is_dir():
        parser.error(f"Simulation directory does not exist: {simulation_directory}")

    backup_root = simulation_directory.parent / "simulation_log_planner_backups"
    files = _log_files(simulation_directory)
    if not files:
        parser.error(f"No simulation logs found under: {simulation_directory}")

    changed = sum(sanitize_log(path, simulation_directory, backup_root) for path in files)
    print(f"Processed {len(files)} log(s); sanitized {changed}. Backups: {backup_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
