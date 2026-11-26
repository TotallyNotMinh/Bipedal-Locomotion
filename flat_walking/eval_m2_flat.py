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
    half = len(rew) // 2
    gain = rew[-1] - rew[half]
    print(f"  tăng trong nửa sau train: {gain:+.2f} ({gain/abs(rew[half])*100:+.1f}%)  -> "
          f"{'còn đang cải thiện' if gain/abs(rew[half]) > 0.05 else 'đã hội tụ (plateau)'}")
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
            "final_len": float(ln[-1]), "gain_second_half": float(gain)}


# ---------------------------------------------------------------- 3. rollout
def load_policy(cfg, env, ckpt_dir):
    import jax
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    from brax.training.agents.ppo import networks as ppo_networks
    from train_m2_flat import _network_factory, _patch_jax
    _patch_jax()
    net = _network_factory(cfg, env.observation_size)(env.observation_size, env.action_size)
    params = ppo_ckpt.load(str(ckpt_dir))
    return jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))


def rollout(cfg, run, n_envs=64, max_steps=1000, seed=123):
    import jax, jax.numpy as jnp
    from train_m2_flat import build_minh_env, _patch_jax
    from g1_minh_env import G1MinhTerrainEnv
    _patch_jax()
    steps = ckpt_steps(run)
    ckpt = steps[-1]
    print(f"=== ROLLOUT checkpoint {ckpt.name} | {n_envs} env x {max_steps} bước (deterministic)")
    minh_env = build_minh_env(cfg)
    env = G1MinhTerrainEnv(minh_env, impl="jax")
    policy = load_policy(cfg, env, ckpt)
    vx, vy, wz = env._cmd_vx, env._cmd_vy, env._cmd_wz
    scenarios = {"ngẫu nhiên (dải của Minh)": None,
                 "tiến vx=0.5": (0.5, 0, 0), "tiến vx=1.0": (1.0, 0, 0),
                 "đi ngang vy=0.3": (0, 0.3, 0), "xoay wz=0.5": (0, 0, 0.5), "đứng yên": (0, 0, 0)}
    print(f"  Dải lệnh: vx={vx} vy={vy} wz={wz}")
    orig = env.sample_command
    out = {}
    for name, cmd in scenarios.items():
        if cmd is not None:
            if not (vx[0] <= cmd[0] <= vx[1] and vy[0] <= cmd[1] <= vy[1] and wz[0] <= cmd[2] <= wz[1]):
                print(f"  - {name}: bỏ qua (ngoài dải lệnh)"); continue
            c = jnp.array(cmd, dtype=jnp.float32); env.sample_command = lambda rng, c=c: c
        else:
            env.sample_command = orig

        def one_step(carry, _):
            st, alive, key = carry
            key, k = jax.random.split(key)
            act, _ = policy(st.obs, k)
            st = env.step(st, act)
            lin = env.get_local_linvel(st.data); gyro = env.get_gyro(st.data)
            rec = (st.info["command"], lin[:2], gyro[2], alive, st.data.qpos[2])
            alive = alive & (st.done < 0.5)
            return (st, alive, key), rec

        @jax.jit
        def run_all(keys):
            st = jax.vmap(env.reset)(keys)
            def one(st0, k):
                (_, alive, _), rec = jax.lax.scan(one_step, (st0, jnp.array(True), k), None, max_steps)
                return rec, alive
            ks = jax.random.split(jax.random.PRNGKey(seed + 1), keys.shape[0])
            return jax.vmap(one)(st, ks)

        (cmd_t, lin_t, wz_t, alive_t, z_t), survived = run_all(jax.random.split(jax.random.PRNGKey(seed), n_envs))
        cmd_t, lin_t, wz_t, alive_t = map(np.asarray, (cmd_t, lin_t, wz_t, alive_t))
        m = alive_t.astype(bool); m[:, :50] = False       # bỏ 1 s đầu (giai đoạn khởi động)
        ex, ey, ew = (lin_t[..., 0] - cmd_t[..., 0])[m], (lin_t[..., 1] - cmd_t[..., 1])[m], (wz_t - cmd_t[..., 2])[m]
        ep_len = alive_t.sum(1)
        r = {"survive_%": 100 * float(np.asarray(survived).mean()), "ep_len": float(ep_len.mean()),
             "rmse_vx": float(np.sqrt((ex**2).mean())), "rmse_vy": float(np.sqrt((ey**2).mean())),
             "rmse_wz": float(np.sqrt((ew**2).mean())),
             "mean_vx": float(lin_t[..., 0][m].mean()), "mean_vy": float(lin_t[..., 1][m].mean()),
             "mean_wz": float(wz_t[m].mean())}
        out[name] = r
        print(f"  - {name:26s} sống sót {r['survive_%']:5.1f}% | len {r['ep_len']:6.1f} | "
              f"RMSE vx {r['rmse_vx']:.3f} vy {r['rmse_vy']:.3f} wz {r['rmse_wz']:.3f} | "
              f"TB vx {r['mean_vx']:+.2f} vy {r['mean_vy']:+.2f} wz {r['mean_wz']:+.2f}")
    env.sample_command = orig
    return out


def verdict(curves, ro):
    print("=== KẾT LUẬN TỰ ĐỘNG")
    rand = next(iter(ro.values()), None)
    pts = []
    if curves:
        pts.append(("Train hội tụ", abs(curves["gain_second_half"]) / max(abs(curves["final_reward"]), 1e-6) < 0.1))
    if rand:
        pts.append(("Sống sót >= 95% (lệnh ngẫu nhiên)", rand["survive_%"] >= 95))
        pts.append(("RMSE vx < 0.25 m/s", rand["rmse_vx"] < 0.25))
        pts.append(("RMSE vy < 0.25 m/s", rand["rmse_vy"] < 0.25))
        pts.append(("RMSE wz < 0.30 rad/s", rand["rmse_wz"] < 0.30))
    for n, ok in pts:
        print(f"  [{'ĐẠT' if ok else 'CHƯA'}] {n}")
    print("  (Ngưỡng là mức tham khảo do mình đặt, hãy chỉnh theo yêu cầu của đồ án.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "flat"))
    ap.add_argument("--run", default=None, help="tên thư mục run; mặc định = run mới nhất")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--check", action="store_true"); ap.add_argument("--curves", action="store_true")
    ap.add_argument("--rollout", action="store_true")
    a = ap.parse_args()
    run = Path(a.results_dir) / a.run if a.run else latest_run(a.results_dir)
    allm = not (a.check or a.curves or a.rollout)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text()) if (run / "resolved_config.yaml").exists() \
        else yaml.safe_load((Path(__file__).parent / "m2_flat.yaml").read_text())
    curves = ro = None
    if allm or a.check:
        check_checkpoint(run, a.backup_dir)
    if allm or a.curves:
        curves = analyze_curves(run, str(run / "train_curves.png"))
    if allm or a.rollout:
        ro = rollout(cfg, run, a.n_envs)
        (run / "eval_rollout.json").write_text(json.dumps(ro, indent=2, ensure_ascii=False))
        print(f"  [OK] Đã lưu: {run/'eval_rollout.json'}")
    if curves or ro:
        verdict(curves, ro or {})


if __name__ == "__main__":
    main()
