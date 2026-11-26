"""Đánh giá M2 sau train: (1) kiểm tra checkpoint, (2) phân tích đường cong train,
(3) rollout policy đã lưu trên env của Minh, đo khả năng bám lệnh vận tốc.

Dùng (từ thư mục gốc repo):
    python flat_walking/eval_m2_flat.py --check            # chỉ kiểm tra/ lưu checkpoint
    python flat_walking/eval_m2_flat.py --curves           # chỉ phân tích metrics.jsonl
    python flat_walking/eval_m2_flat.py --rollout          # chỉ rollout
    python flat_walking/eval_m2_flat.py                    # cả ba
"""
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import yaml


def latest_run(results_dir):
    runs = [d for d in Path(results_dir).glob("*") if (d / "checkpoints").is_dir()]
    if not runs:
        raise FileNotFoundError(f"Không có run nào trong {results_dir}")
    return max(runs, key=lambda d: d.stat().st_mtime)


def ckpt_steps(run):
    return sorted((d for d in (run / "checkpoints").iterdir() if d.is_dir() and d.name.isdigit()),
                  key=lambda d: int(d.name))


def du(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


# ---------------------------------------------------------------- 1. checkpoint
def check_checkpoint(run, backup_dir=None):
    print(f"=== CHECKPOINT: {run}")
    steps = ckpt_steps(run)
    ok = True
    if not steps:
        print("[X] KHÔNG có thư mục checkpoint nào (<step>/) -> chưa lưu!"); return False
    for d in steps:
        n = sum(1 for f in d.rglob("*") if f.is_file())
        print(f"  [OK] step {int(d.name):>11,}: {n} file, {du(d)/1e6:.2f} MB")
        if n == 0 or du(d) == 0:
            ok = False; print("       [X] rỗng!")
    for name in ["resolved_config.yaml", "metrics.jsonl", "run_metadata.json"]:
        f = run / name
        print(f"  [{'OK' if f.exists() and f.stat().st_size else 'X '}] {name}")
        ok &= f.exists() and f.stat().st_size > 0
    if (run / "run_metadata.json").exists():
        meta = json.loads((run / "run_metadata.json").read_text())
        print(f"  Thời gian train: {meta['total_wall_time_s']:.0f}s | params: {meta['param_counts']}")
    if backup_dir:
        dst = Path(backup_dir) / run.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(run, dst)
        z = shutil.make_archive(str(Path(backup_dir) / run.name), "zip", run.parent, run.name)
        print(f"  [OK] Đã sao lưu: {dst}\n  [OK] Zip để tải về: {z} ({Path(z).stat().st_size/1e6:.1f} MB)")
    print("=>", "CHECKPOINT HỢP LỆ, ĐÃ LƯU" if ok else "CÓ VẤN ĐỀ")
    return ok


# ---------------------------------------------------------------- 2. đường cong
def analyze_curves(run, save_png=None):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
    step = np.array([r["step"] for r in rows])
    rew = np.array([r.get("eval/episode_reward", np.nan) for r in rows])
    ln = np.array([r.get("eval/avg_episode_length", np.nan) for r in rows])
    print(f"=== ĐƯỜNG CONG TRAIN ({len(rows)} lần eval, tới {step[-1]:,} bước)")
    best = int(np.nanargmax(rew))
    last3 = rew[-3:]
    print(f"  reward cuối: {rew[-1]:.2f} | tốt nhất: {rew[best]:.2f} @ {step[best]:,}")
    print(f"  trung bình 3 eval cuối: {last3.mean():.2f} ± {last3.std():.2f}")
    from m2_metrics import curve_trend
    gain, rel, converged = curve_trend(rew)
    print(f"  tăng trong nửa sau train: {gain:+.2f} ({rel*100:+.1f}%)  -> "
          f"{'đã hội tụ (plateau, <5%)' if converged else 'còn đang cải thiện (>=5%)'}")
    fell = np.argmax(ln >= 0.95 * ln.max()) if (ln >= 0.95 * ln.max()).any() else None
    print(f"  độ dài episode cuối: {ln[-1]:.0f}/1000 | đạt >=95% mức max từ {step[fell]:,} bước" if fell is not None else "")
    keys = sorted({k for r in rows for k in r if k.startswith("eval/episode_") and k not in
                   ("eval/episode_reward",) and "std" not in k})
    if keys:
        print("  Thành phần reward (eval cuối):")
        for k in sorted(keys, key=lambda k: -abs(rows[-1].get(k, 0)))[:15]:
            print(f"    {k.replace('eval/episode_',''):34s} {rows[-1].get(k, float('nan')):9.3f}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(step / 1e6, rew, "o-"); ax[0].set(title="eval reward", xlabel="triệu bước")
        ax[1].plot(step / 1e6, ln, "o-", c="C1"); ax[1].set(title="eval episode length", xlabel="triệu bước")
        for a in ax: a.grid(alpha=.3)
        plt.tight_layout()
        if save_png:
            plt.savefig(save_png, dpi=120); print(f"  [OK] Đã lưu biểu đồ: {save_png}")
        plt.show()
    except Exception as e:
        print("  (bỏ qua vẽ biểu đồ:", e, ")")
    return {"final_reward": float(rew[-1]), "best_reward": float(rew[best]),
            "final_len": float(ln[-1]), "gain_second_half": gain,
            "rel_gain": rel, "converged": converged}


# ---------------------------------------------------------------- 3. rollout
def body_vel(data):
    """Vận tốc dài (hệ toạ độ thân) và vận tốc góc của pelvis, tính trực tiếp từ qpos/qvel
    (không phụ thuộc tên sensor/frame của Playground)."""
    import jax.numpy as jnp
    q, v = data.qpos[3:7], data.qvel[0:3]          # quat (w,x,y,z), vận tốc dài hệ thế giới
    w, u = q[0], -q[1:]                            # quay ngược bằng quaternion liên hợp
    t = 2.0 * jnp.cross(u, v)
    lin = v + w * t + jnp.cross(u, t)
    return lin, data.qvel[3:6]                     # khớp tự do: vận tốc góc đã ở hệ thân


def load_policy(cfg, env, ckpt_dir):
    import jax
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    from brax.training.agents.ppo import networks as ppo_networks
    from train_m2_flat import _network_factory, _patch_jax
    _patch_jax()
    net = _network_factory(cfg, env.observation_size)(env.observation_size, env.action_size)
    params = ppo_ckpt.load(str(ckpt_dir))
    return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))


def _make_env_policy(cfg, run):
    from train_m2_flat import build_minh_env, _patch_jax
    from g1_minh_env import G1MinhTerrainEnv
    _patch_jax()
    ckpt = ckpt_steps(run)[-1]
    env = G1MinhTerrainEnv(build_minh_env(cfg), impl="jax")
    return env, load_policy(cfg, env, ckpt), ckpt


def rollout(cfg, run, n_envs=None, scen_file=None):
    """Chạy các kịch bản cố định trong eval_scenarios.yaml; trả về dict kết quả mỗi kịch bản."""
    import jax, jax.numpy as jnp
    from m2_metrics import fall_stats, tracking_metrics, command_in_range, check_thresholds
    sc = yaml.safe_load(Path(scen_file or Path(__file__).parent / "eval_scenarios.yaml").read_text())
    n_envs = n_envs or sc["n_envs"]; max_steps = sc["max_steps"]; seed = sc["seed"]
    env, policy, ckpt = _make_env_policy(cfg, run)
    print(f"=== ROLLOUT checkpoint {ckpt.name} | {n_envs} env x {max_steps} bước | seed {seed} (deterministic)")
    vx, vy, wz = env._cmd_vx, env._cmd_vy, env._cmd_wz
    print(f"  Dải lệnh: vx={vx} vy={vy} wz={wz}")
    orig = env.sample_command
    out = {}
    for name, cmd in sc["scenarios"].items():
        if cmd is not None:
            assert command_in_range(cmd, vx, vy, wz), f"kịch bản {name} ngoài dải lệnh"
            c = jnp.array(cmd, dtype=jnp.float32); env.sample_command = lambda rng, c=c: c
        else:
            env.sample_command = orig

        def one_step(carry, _):
            st, alive, key = carry
            key, k = jax.random.split(key)
            act, _ = policy(st.obs, k)
            st = env.step(st, act)
            lin, gyro = body_vel(st.data)
            q = st.data.qpos[3:7]                                 # độ thẳng đứng: z của trục z thân
            up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
            alive = alive & (st.done < 0.5)          # alive[t]=True: còn sống SAU bước t
            rec = (st.info["command"], jnp.concatenate([lin[:2], gyro[2:3]]), alive,
                   st.data.qpos[2], up_z)
            return (st, alive, key), rec

        @jax.jit
        def run_all(keys):
            st = jax.vmap(env.reset)(keys)
            ks = jax.random.split(jax.random.PRNGKey(seed + 1), keys.shape[0])
            return jax.vmap(lambda s0, k: jax.lax.scan(
                one_step, (s0, jnp.array(True), k), None, max_steps)[1])(st, ks)

        cmd_t, vel_t, alive_t, z_t, up_t = map(np.asarray, run_all(
            jax.random.split(jax.random.PRNGKey(seed), n_envs)))
        fs = fall_stats(alive_t)
        tm = tracking_metrics(cmd_t, vel_t, alive_t, sc["warmup_steps"])
        r = {**fs, **tm, "command": cmd}
        # chẩn đoán ngã: tư thế ngay trước khi chết
        diag = []
        for i in np.where(~alive_t[:, -1])[0]:
            t0 = int(np.argmax(~alive_t[i]))
            tp = t0
            diag.append({"env": int(i), "step": t0, "z": float(z_t[i, tp]), "up_z": float(up_t[i, tp]),
                         "command": [float(x) for x in cmd_t[i, tp]]})
        r["fall_diagnostics"] = diag
        r["thresholds_ok"] = check_thresholds(r, sc["thresholds"])
        out[name] = r
        print(f"  - {name:12s} ngã {fs['n_fell']:2d}/{fs['n_envs']} | len {fs['mean_episode_len']:6.1f} | "
              f"RMSE vx {tm['rmse_vx']:.3f} vy {tm['rmse_vy']:.3f} wz {tm['rmse_wz']:.3f} | "
              f"TB vx {tm['mean_vx']:+.2f} vy {tm['mean_vy']:+.2f} wz {tm['mean_wz']:+.2f}")
        for d in diag:
            print(f"      ngã env {d['env']} ở bước {d['step']}: z={d['z']:.2f} m, up_z={d['up_z']:.2f}, "
                  f"lệnh={[round(x, 2) for x in d['command']]}")
    env.sample_command = orig
    out["_meta"] = {"checkpoint": ckpt.name, "seed": seed, "thresholds": sc["thresholds"]}
    return out


def check_reload(cfg, run):
    """Nạp checkpoint 2 lần độc lập; cùng obs + cùng PRNG phải ra cùng action."""
    import jax, jax.numpy as jnp
    from train_m2_flat import build_minh_env, _patch_jax
    from g1_minh_env import G1MinhTerrainEnv
    _patch_jax()
    ckpt = ckpt_steps(run)[-1]
    env = G1MinhTerrainEnv(build_minh_env(cfg), impl="jax")
    p1, p2 = load_policy(cfg, env, ckpt), load_policy(cfg, env, ckpt)
    st = jax.jit(env.reset)(jax.random.PRNGKey(7))
    k = jax.random.PRNGKey(1)
    a1, a2, a3 = p1(st.obs, k)[0], p2(st.obs, k)[0], p1(st.obs, jax.random.PRNGKey(2))[0]
    same_reload = bool(jnp.array_equal(a1, a2))
    same_det = bool(jnp.array_equal(a1, a3))          # deterministic: không phụ thuộc PRNG
    print(f"=== RELOAD {ckpt.name}: 2 lần nạp cho cùng action: {same_reload} | "
          f"deterministic (độc lập PRNG): {same_det} | action finite: {bool(jnp.isfinite(a1).all())}")
    return same_reload and same_det and bool(jnp.isfinite(a1).all())


def resources(cfg, run, out_json=None):
    """Phiên bản thư viện, GPU, VRAM đỉnh, độ trễ suy luận."""
    import importlib.metadata as md, subprocess, time, platform
    import jax, jax.numpy as jnp
    info = {"python": platform.python_version(), "versions": {}}
    for pkg in ["jax", "jaxlib", "brax", "flax", "optax", "orbax-checkpoint", "mujoco", "mujoco-mjx",
                "playground", "mujoco-playground", "numpy"]:
        try: info["versions"][pkg] = md.version(pkg)
        except Exception: pass
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    except Exception: pass
    env, policy, ckpt = _make_env_policy(cfg, run)
    st = jax.jit(env.reset)(jax.random.PRNGKey(0))
    lat = {}
    for b in (1, 64):
        obs = jax.tree.map(lambda x: jnp.stack([x] * b), st.obs)
        f = jax.jit(jax.vmap(policy, in_axes=(0, None)))
        k = jax.random.PRNGKey(0)
        f(obs, k)[0].block_until_ready()
        t0 = time.perf_counter()
        for _ in range(200): f(obs, k)[0].block_until_ready()
        lat[f"batch_{b}_ms"] = round((time.perf_counter() - t0) / 200 * 1e3, 3)
    info["inference_latency"] = lat
    info["checkpoint_mb"] = round(du(ckpt) / 1e6, 2)
    try:
        info["peak_vram_mb_per_device"] = [round(d.memory_stats().get("peak_bytes_in_use", 0) / 1e6, 1)
                                          for d in jax.devices()]
    except Exception: pass
    meta = run / "run_metadata.json"
    if meta.exists():
        m = json.loads(meta.read_text()); info["train_wall_time_s"] = m.get("total_wall_time_s")
        info["param_counts"] = m.get("param_counts")
    print("=== TÀI NGUYÊN"); print(json.dumps(info, indent=2, ensure_ascii=False))
    if out_json: Path(out_json).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def verdict(curves, ro):
    print("=== KẾT LUẬN TỰ ĐỘNG (ngưỡng TẠM, chưa được nhóm thống nhất)")
    if curves:
        print(f"  [{'ĐẠT' if curves['converged'] else 'CHƯA'}] Reward nửa sau tăng {curves['rel_gain']*100:+.1f}% "
              f"(hội tụ nếu < 5%)")
    for name, r in ro.items():
        if name.startswith("_"): continue
        bad = [k for k, v in r["thresholds_ok"].items() if not v]
        print(f"  [{'ĐẠT' if not bad else 'CHƯA'}] {name:12s}" + (f" -> trượt: {', '.join(bad)}" if bad else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "flat"))
    ap.add_argument("--run", default=None, help="tên thư mục run; mặc định = run mới nhất")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--curves", action="store_true")
    ap.add_argument("--rollout", action="store_true"); ap.add_argument("--reload", action="store_true")
    ap.add_argument("--resources", action="store_true")
    a = ap.parse_args()
    run = Path(a.results_dir) / a.run if a.run else latest_run(a.results_dir)
    allm = not (a.check or a.curves or a.rollout or a.reload or a.resources)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text()) if (run / "resolved_config.yaml").exists() \
        else yaml.safe_load((Path(__file__).parent / "m2_flat.yaml").read_text())
    curves = ro = None
    if allm or a.check:
        check_checkpoint(run, a.backup_dir)
    if allm or a.curves:
        curves = analyze_curves(run, str(run / "train_curves.png"))
    if allm or a.reload:
        ok = check_reload(cfg, run)
        (run / "reload_check.json").write_text(json.dumps({"ok": ok}))
    if allm or a.rollout:
        ro = rollout(cfg, run, a.n_envs)
        (run / "eval_rollout.json").write_text(json.dumps(ro, indent=2, ensure_ascii=False))
        print(f"  [OK] Đã lưu: {run/'eval_rollout.json'}")
    if allm or a.resources:
        resources(cfg, run, run / "resources.json")
    if curves or ro:
        verdict(curves, ro or {})


if __name__ == "__main__":
    main()
