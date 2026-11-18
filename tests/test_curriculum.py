"""Unit tests for Sequential Stage Curriculum with Continuous Prerequisite Health Check & Targeted Intervention."""

import numpy as np
import yaml
from src.envs.curriculum import TerrainCurriculum


def test_initial_zero_spawn_rate():
    """Verify that at step 0, higher terrain levels have strictly 0.0% spawn probability."""
    with open("configs/terrain.yaml", "r") as f:
        config = yaml.safe_load(f)

    curriculum = TerrainCurriculum(config)

    assert curriculum.frontier_idx == 0
    assert curriculum.frontier_category == "flat"

    probs = curriculum.probabilities
    assert probs["flat"] == 1.0
    for stage in ["gentle_slopes", "rolling_hills", "rough_terrain", "steep_slopes_obstacles"]:
        assert probs[stage] == 0.0, f"Stage {stage} must start at strictly 0.0% probability"


def test_prerequisite_gated_unlocking():
    """Verify next stage unlocks only when all preceding stages sustain >= 90% success."""
    with open("configs/terrain.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Use fast step increments for test
    config["curriculum"]["patience"]["promotion_patience"] = 3
    config["curriculum"]["step_size"]["difficulty_inc"] = 0.5

    curriculum = TerrainCurriculum(config)

    # Record 20 successful episodes on flat
    for _ in range(20):
        curriculum.record_episode("flat", success=True, reward=100.0, length=500, distance=10.0, forward_vel=1.0, fall=False)

    # Step update until d=1.0 and promotion
    for _ in range(10):
        state = curriculum.update()

    assert curriculum.frontier_idx == 1
    assert curriculum.frontier_category == "gentle_slopes"
    assert curriculum.probabilities["gentle_slopes"] > 0.50
    assert curriculum.probabilities["flat"] > 0.05  # Retained
    assert curriculum.probabilities["rolling_hills"] == 0.0  # Still locked at 0.0%


def test_targeted_intervention_with_patience():
    """Verify that intervention requires sustaining failure for patience window before shifting distribution."""
    with open("configs/terrain.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["curriculum"]["patience"]["intervention_patience"] = 3
    config["curriculum"]["patience"]["recovery_patience"] = 2
    config["curriculum"]["patience"]["promotion_patience"] = 2
    config["curriculum"]["step_size"]["difficulty_inc"] = 0.5
    curriculum = TerrainCurriculum(config)

    # Master flat and gentle_slopes -> unlock rolling_hills (frontier=2)
    for _ in range(25):
        curriculum.record_episode("flat", success=True, reward=100.0, length=500, distance=10.0, forward_vel=1.0, fall=False)
        curriculum.record_episode("gentle_slopes", success=True, reward=100.0, length=500, distance=10.0, forward_vel=1.0, fall=False)
        curriculum.update()

    curriculum.frontier_idx = 2  # Stage 2: rolling_hills is active frontier
    curriculum.in_intervention = False
    curriculum.prereq_fail_streak = 0
    curriculum.update()

    # Simulate Stage 1 (gentle_slopes) failing
    for _ in range(20):
        curriculum.record_episode("gentle_slopes", success=False, reward=5.0, length=40, distance=0.5, forward_vel=0.1, fall=True)

    # Check 1: Patience streak = 1 -> should NOT be in intervention yet!
    curriculum.update()
    assert not curriculum.in_intervention, "Must wait for patience window before shifting distribution"
    assert curriculum.prereq_fail_streak == 1

    # Check 2: Patience streak = 2 -> still waiting
    curriculum.update()
    assert not curriculum.in_intervention
    assert curriculum.prereq_fail_streak == 2

    # Check 3: Patience streak = 3 -> Now patience is satisfied!
    curriculum.update()
    assert curriculum.in_intervention, "Must enter intervention once patience window expires"
    assert curriculum.failing_stage == "gentle_slopes"
    assert np.isclose(curriculum.probabilities["gentle_slopes"], 0.75, atol=0.01)

    # Now simulate recovery
    for _ in range(50):
        curriculum.record_episode("gentle_slopes", success=True, reward=100.0, length=500, distance=10.0, forward_vel=1.0, fall=False)

    # Check 1 of recovery: streak = 1 -> still in intervention
    curriculum.update()
    assert curriculum.in_intervention
    assert curriculum.recovery_streak == 1

    # Check 2 of recovery: streak = 2 (meets recovery_patience=2) -> restored!
    curriculum.update()
    assert not curriculum.in_intervention, "Must exit intervention once recovery patience is confirmed"


if __name__ == "__main__":
    test_initial_zero_spawn_rate()
    test_prerequisite_gated_unlocking()
    test_targeted_intervention_with_patience()
    print("All prerequisite health-check and targeted intervention tests with patience passed successfully!")
