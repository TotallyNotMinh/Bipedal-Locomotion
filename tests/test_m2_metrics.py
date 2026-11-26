"""Unit test metric của M2 trên quỹ đạo giả (plan Phase 7: mỗi metric phải có test)."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "flat_walking"))
from m2_metrics import (check_thresholds, command_in_range, curve_trend,  # noqa: E402
                        fall_stats, tracking_metrics)

N, T = 4, 200


def _traj(cmd=(0.5, 0.0, 0.0)):
    c = np.tile(np.array(cmd, float), (N, T, 1))
    return c, c.copy(), np.ones((N, T), bool)


def test_perfect_tracking_zero_error():
    c, v, a = _traj()
    m = tracking_metrics(c, v, a)
    assert m["rmse_vx"] == 0 and m["rmse_vy"] == 0 and m["rmse_wz"] == 0
    assert np.isclose(m["mean_vx"], 0.5)


def test_constant_offset_rmse():
    c, v, a = _traj()
    v[..., 0] += 0.1
    v[..., 2] -= 0.2
    m = tracking_metrics(c, v, a)
    assert np.isclose(m["rmse_vx"], 0.1) and np.isclose(m["rmse_wz"], 0.2) and m["rmse_vy"] == 0


def test_warmup_is_ignored():
    c, v, a = _traj()
    v[:, :50, 0] = 5.0                       # sai lớn nhưng chỉ trong warmup
    assert tracking_metrics(c, v, a, warmup=50)["rmse_vx"] == 0
    assert tracking_metrics(c, v, a, warmup=0)["rmse_vx"] > 1


def test_dead_steps_are_excluded():
    c, v, a = _traj()
    a[0, 100:] = False
    v[0, 100:, 0] = 99.0                     # rác sau khi ngã không được tính
    assert tracking_metrics(c, v, a)["rmse_vx"] == 0


def test_all_fell_before_warmup_gives_nan_not_crash():
    c, v, a = _traj()
    a[:, 10:] = False
    m = tracking_metrics(c, v, a, warmup=50)
    assert m["n_samples"] == 0 and np.isnan(m["rmse_vx"])


def test_no_fall():
    s = fall_stats(np.ones((N, T), bool))
    assert s["n_fell"] == 0 and s["fall_rate"] == 0 and s["success_rate"] == 1
    assert s["mean_episode_len"] == T and s["fall_steps"] == []


def test_fall_recorded_as_failure_with_step():
    a = np.ones((N, T), bool)
    a[1, 120:] = False
    a[3, 30:] = False
    s = fall_stats(a)
    assert s["n_fell"] == 2 and np.isclose(s["fall_rate"], 0.5)
    assert sorted(s["fall_steps"]) == [30, 120]
    assert np.isclose(s["mean_episode_len"], (T + 120 + T + 30) / 4)


def test_all_fall():
    s = fall_stats(np.zeros((N, T), bool))
    assert s["success_rate"] == 0 and s["mean_episode_len"] == 0


def test_command_range():
    vx, vy, wz = (0.2, 1.0), (-0.3, 0.3), (-0.5, 0.5)
    assert command_in_range((0.5, 0.0, 0.0), vx, vy, wz)
    assert command_in_range((0.2, 0.3, -0.5), vx, vy, wz)        # sát biên
    assert not command_in_range((0.0, 0.0, 0.0), vx, vy, wz)     # vx=0 ngoài dải
    assert not command_in_range((0.5, 0.4, 0.0), vx, vy, wz)


def test_thresholds():
    th = dict(min_success_rate=0.95, max_rmse_vx=0.25, max_rmse_vy=0.25, max_rmse_wz=0.3)
    good = dict(success_rate=1.0, rmse_vx=0.1, rmse_vy=0.1, rmse_wz=0.1)
    assert all(check_thresholds(good, th).values())
    bad = dict(good, success_rate=0.9, rmse_wz=0.5)
    r = check_thresholds(bad, th)
    assert sum(not v for v in r.values()) == 2


def test_curve_trend():
    g, rel, conv = curve_trend([1, 5, 10, 20, 30, 30.5])
    assert not conv and g > 0
    assert curve_trend([10] * 10)[2] is True


def test_scenarios_file_is_inside_minh_range():
    """Mọi kịch bản cố định phải nằm trong dải lệnh của Minh (vx 0.2-1.0, vy +-0.3, wz +-0.5)."""
    cfg = yaml.safe_load((ROOT / "flat_walking" / "eval_scenarios.yaml").read_text())
    for name, cmd in cfg["scenarios"].items():
        if cmd is not None:
            assert command_in_range(cmd, (0.2, 1.0), (-0.3, 0.3), (-0.5, 0.5)), name
