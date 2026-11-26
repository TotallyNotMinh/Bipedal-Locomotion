"""Metric đánh giá thuần numpy (không cần JAX) để test được trên quỹ đạo giả.

Quy ước dữ liệu: N env, T bước.
  cmd  [N,T,3]  lệnh (vx, vy, wz)
  vel  [N,T,3]  vận tốc thực (vx, vy trong hệ thân; wz quanh trục z)
  alive[N,T]    True nếu env CHƯA bị termination trước bước t (kể cả bước t)

Định nghĩa:
  - FALL: env bị termination (done) trong khoảng T bước. fall_rate = tỉ lệ env ngã.
  - SUCCESS: không ngã trong T bước (success_rate = 1 - fall_rate).
  - Sai số vận tốc: RMSE trên các bước còn sống, bỏ `warmup` bước đầu (khởi động).
"""
import numpy as np


def fall_stats(alive):
    alive = np.asarray(alive, dtype=bool)
    n, t = alive.shape
    fell = ~alive[:, -1]
    first_dead = np.where(fell, np.argmax(~alive, axis=1), -1)   # bước đầu tiên đã chết
    return {"n_envs": int(n), "n_fell": int(fell.sum()), "fall_rate": float(fell.mean()),
            "success_rate": float(1 - fell.mean()), "mean_episode_len": float(alive.sum(1).mean()),
            "fall_steps": [int(x) for x in first_dead[fell]]}


def tracking_metrics(cmd, vel, alive, warmup=50):
    cmd, vel = np.asarray(cmd, float), np.asarray(vel, float)
    m = np.asarray(alive, dtype=bool).copy()
    m[:, :warmup] = False
    out = {"n_samples": int(m.sum())}
    if m.sum() == 0:
        out.update({k: float("nan") for k in
                    ["rmse_vx", "rmse_vy", "rmse_wz", "mean_vx", "mean_vy", "mean_wz"]})
        return out
    err = (vel - cmd)[m]
    for i, k in enumerate(["vx", "vy", "wz"]):
        out[f"rmse_{k}"] = float(np.sqrt((err[:, i] ** 2).mean()))
        out[f"mean_{k}"] = float(vel[..., i][m].mean())
    return out


def command_in_range(cmd, vx, vy, wz):
    return bool(vx[0] <= cmd[0] <= vx[1] and vy[0] <= cmd[1] <= vy[1] and wz[0] <= cmd[2] <= wz[1])


def check_thresholds(m, th):
    """m: dict gộp fall_stats + tracking_metrics. th: dict ngưỡng. Trả về {tên: bool}."""
    return {
        f"success_rate >= {th['min_success_rate']}": m["success_rate"] >= th["min_success_rate"],
        f"rmse_vx < {th['max_rmse_vx']}": m["rmse_vx"] < th["max_rmse_vx"],
        f"rmse_vy < {th['max_rmse_vy']}": m["rmse_vy"] < th["max_rmse_vy"],
        f"rmse_wz < {th['max_rmse_wz']}": m["rmse_wz"] < th["max_rmse_wz"],
    }


def curve_trend(reward, tol=0.05):
    """Reward nửa sau tăng bao nhiêu so với giữa train. Trả (gain, rel_gain, converged)."""
    r = np.asarray(reward, float)
    half = len(r) // 2
    gain = float(r[-1] - r[half])
    rel = gain / max(abs(r[half]), 1e-9)
    return gain, float(rel), bool(rel < tol)
