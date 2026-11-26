"""Test reload checkpoint của M2: nạp 2 lần cho cùng action, deterministic.
Cần JAX/Brax/Playground và một run đã train. Đặt M2_RUN_DIR=results/flat/<run> để chạy; không thì bỏ qua.
"""
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "flat_walking"))


def test_checkpoint_reload_same_actions():
    run = os.environ.get("M2_RUN_DIR")
    try:
        import pytest
        if not run:
            pytest.skip("đặt M2_RUN_DIR để chạy")
        pytest.importorskip("brax"); pytest.importorskip("mujoco_playground")
    except ImportError:
        if not run:
            return
    from eval_m2_flat import check_reload
    run = Path(run)
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    assert check_reload(cfg, run)
