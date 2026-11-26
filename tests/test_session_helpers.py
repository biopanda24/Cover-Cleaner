"""Trim that follows rotation, the scratch folder, and the worker pool."""
import os
import threading

import numpy as np

from covercleaner import cli, scratch
from covercleaner.geometry import apply_geometry
from covercleaner.params import trim_from_view, trim_to_view


def test_trim_boxes_follow_rotation():
    img = np.random.rand(60, 100, 3).astype(np.float32)
    for k in range(4):
        for view in ([7, 0, 0, 0], [0, 7, 0, 0], [0, 0, 7, 0], [0, 0, 0, 7], [3, 5, 11, 13]):
            stored = trim_from_view(view, k)
            assert trim_to_view(stored, k) == view
            shown = np.rot90(img, -k)                    # what is on screen after k turns
            l, t, r, b = view
            h, w = shown.shape[:2]
            assert np.array_equal(apply_geometry(img, None, trim=stored, rotate90=k),
                                  shown[t:h - b, l:w - r])


def test_scratch_round_trip_is_exact_for_saving(tmp_path):
    s = scratch.Scratch(str(tmp_path))
    img = np.random.rand(40, 30, 3).astype(np.float32)
    back = scratch.load_result(scratch.save_result(s.new_path(), img))
    u8 = lambda a: np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
    u16 = lambda a: np.clip(a * 65535 + 0.5, 0, 65535).astype(np.uint16)
    assert np.array_equal(u8(back), u8(img)) and np.array_equal(u16(back), u16(img))
    s.close()
    assert not os.path.exists(s.dir)


def test_stale_session_folders_are_removed(tmp_path):
    dead = tmp_path / "session-999999-20200101000000"     # no such process
    mine = tmp_path / f"session-{os.getpid()}-20200101000000"
    dead.mkdir()
    mine.mkdir()
    scratch.clean_stale(str(tmp_path))
    assert not dead.exists() and mine.exists()


def test_run_pool_in_process_and_in_workers():
    for jobs in (1, 2):
        got = {}
        cli.run_pool(abs, [(-1,), (-2,), (-3,)], jobs, lambda t, r: got.__setitem__(t[0], r))
        assert got == {-1: 1, -2: 2, -3: 3}


def test_run_pool_stop():
    stop = threading.Event()
    stop.set()
    got = []
    cli.run_pool(abs, [(-1,), (-2,)], 1, lambda t, r: got.append(r), stop)
    assert got == []
