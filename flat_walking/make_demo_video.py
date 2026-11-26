"""Quay video demo M2: robot G1 đi theo chuỗi lệnh vận tốc, kèm đồ thị vận tốc lệnh vs thực tế.

Dùng (từ thư mục gốc repo, cần GPU/EGL như trên Kaggle):
    python flat_walking/make_demo_video.py [--run <tên run>] [--out demo_m2.mp4]
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, yaml

# (vx, vy, wz, mô tả) — mỗi pha 3 giây, nằm trong dải lệnh của Minh
PHASES = [(0.5, 0.0, 0.0, "tien cham"), (1.0, 0.0, 0.0, "tien nhanh"), (0.5, 0.0, 0.5, "re trai"),
          (0.5, 0.0, -0.5, "re phai"), (0.4, 0.3, 0.0, "di cheo trai"), (0.4, -0.3, 0.0, "di cheo phai")]
PHASE_STEPS = 150  # 3 s @ 50 Hz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "flat"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import jax, jax.numpy as jnp, mujoco, imageio.v2 as imageio
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from eval_m2_flat import latest_run, ckpt_steps, load_policy, body_vel
    from train_m2_flat import build_minh_env, _patch_jax
    from g1_minh_env import G1MinhTerrainEnv
    _patch_jax()

    run = Path(a.results_dir) / a.run if a.run else latest_run(a.results_dir)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    env = G1MinhTerrainEnv(build_minh_env(cfg), impl="jax")
    policy = load_policy(cfg, env, ckpt_steps(run)[-1])
    step_fn = jax.jit(env.step)
    vel_fn = jax.jit(body_vel)

    state = jax.jit(env.reset)(jax.random.PRNGKey(a.seed))
    key = jax.random.PRNGKey(a.seed + 1)
    qpos, qvel, cmds, acts = [], [], [], []
    for t in range(PHASE_STEPS * len(PHASES)):
        vx, vy, wz, _ = PHASES[t // PHASE_STEPS]
        c = jnp.array([vx, vy, wz], dtype=jnp.float32)
        info = dict(state.info); info["command"] = c
        if "steps_until_next_cmd" in info:
            info["steps_until_next_cmd"] = jnp.array(10**6, dtype=info["steps_until_next_cmd"].dtype)
        state = state.replace(info=info)
        key, k = jax.random.split(key)
        act, _ = policy(state.obs, k)
        state = step_fn(state, act)
        lin, gyro = vel_fn(state.data)
        qpos.append(np.asarray(state.data.qpos)); qvel.append(np.asarray(state.data.qvel))
        cmds.append([vx, vy, wz]); acts.append([float(lin[0]), float(lin[1]), float(gyro[2])])
        if float(state.done) > 0.5:
            print(f"[!] Robot ngã ở bước {t}, dừng quay."); break
    n = len(qpos); cmds, acts = np.array(cmds), np.array(acts)
    print(f"[+] Mô phỏng xong {n} bước ({n/50:.1f}s)")

    # ---- render bằng MuJoCo CPU
    m = env.mj_model; d = mujoco.MjData(m)
    W, H = 640, 480
    renderer = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = 3.5, 135, -15

    fig, axs = plt.subplots(3, 1, figsize=(W / 100, H / 100), dpi=100, sharex=True)
    tt = np.arange(n) / 50
    names = ["vx (m/s)", "vy (m/s)", "wz (rad/s)"]
    cur, ttl = [], fig.suptitle("", fontsize=9)
    for i, ax in enumerate(axs):
        ax.plot(tt, cmds[:, i], "k--", lw=1, label="lệnh")
        ax.plot(tt, acts[:, i], "C0", lw=1, label="thực tế")
        cur.append(ax.axvline(0, c="r", lw=1)); ax.set_ylabel(names[i], fontsize=8); ax.grid(alpha=.3)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, loc="upper left"); axs[-1].set_xlabel("thời gian (s)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    frames = []
    for t in range(0, n, 2):                       # 25 fps
        d.qpos[:], d.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        img = renderer.render()
        for c_ in cur: c_.set_xdata([tt[t], tt[t]])
        ttl.set_text(f"pha: {PHASES[t // PHASE_STEPS][3]} | lenh vx={cmds[t,0]:.2f} vy={cmds[t,1]:.2f} wz={cmds[t,2]:.2f}"
                     f" | thuc te vx={acts[t,0]:.2f} vy={acts[t,1]:.2f} wz={acts[t,2]:.2f}")
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plot = plot[:H, :W]
        frames.append(np.hstack([img, plot]))
    out = a.out or str(run / "demo_m2.mp4")
    imageio.mimsave(out, frames, fps=25, macro_block_size=1)
    print(f"[OK] Đã lưu video: {out}")
    rm = np.sqrt(((cmds - acts) ** 2).mean(0))
    print(f"RMSE theo lệnh: vx={rm[0]:.3f} vy={rm[1]:.3f} wz={rm[2]:.3f}")


if __name__ == "__main__":
    main()
