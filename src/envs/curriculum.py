"""Adaptive Sequential Terrain Curriculum Manager with Continuous Prerequisite Health Check.

Features:
1. Zero Initial Spawn: Higher terrain levels strictly start at 0.0% spawn probability.
2. Sequential Frontier Advancement: Next stage unlocks ONLY when:
   - All preceding stages sustain >= 90% success rate.
   - The current frontier stage sustains >= 90% success rate at max difficulty d=1.0.
3. Targeted Intervention (Anti-Catastrophic Forgetting):
   - Prerequisites are checked continuously.
   - If ANY prerequisite drops below 90% success:
     - 75% of rollouts are immediately shifted to the failing prerequisite.
     - 25% remains distributed across the other unlocked stages as anchors to prevent forgetting them during recovery.
   - Once the failing prerequisite recovers (>= 90%), normal frontier progression resumes.
"""

from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class EpisodeRecord:
    success: bool
    reward: float
    length: int
    distance: float
    forward_vel: float
    fall: bool
    difficulty: float


class TerrainCurriculum:
    """Manages sequential stage advancement, prerequisite health checks, and targeted interventions."""

    DEFAULT_STAGES = [
        "flat",
        "gentle_slopes",
        "rolling_hills",
        "rough_terrain",
        "steep_slopes_obstacles",
    ]

    def __init__(self, config: dict):
        self.config = config
        t_cfg = config.get("terrain", {})
        c_cfg = config.get("curriculum", {})

        self.stages: List[str] = t_cfg.get("stages", self.DEFAULT_STAGES)
        self.num_stages = len(self.stages)

        # Frontier stage index (0: flat, 1: gentle_slopes, ..., 4: steep_slopes_obstacles)
        self.frontier_idx = 0

        # Continuous difficulty d in [0, 1] per stage
        self.difficulties = {cat: 0.0 for cat in self.stages}

        # Rolling history per individual terrain category
        self.window_size = int(c_cfg.get("per_stage_window_size", 50))
        self.stage_history: Dict[str, deque[EpisodeRecord]] = {
            cat: deque(maxlen=self.window_size) for cat in self.stages
        }

        # Thresholds
        thresholds = c_cfg.get("thresholds", {})
        self.prereq_thresh = float(thresholds.get("prerequisite_success_rate", 0.90))
        self.frontier_thresh = float(thresholds.get("frontier_mastery_success_rate", 0.90))
        self.min_avg_distance = float(thresholds.get("min_avg_distance", 6.0))
        self.max_fall_rate = float(thresholds.get("max_fall_rate", 0.20))

        # Intervention Settings
        interv_cfg = c_cfg.get("intervention", {})
        self.intervention_enabled = bool(interv_cfg.get("enabled", True))
        self.failing_focus_prob = float(interv_cfg.get("failing_stage_focus_prob", 0.75))
        self.anchor_prob = float(interv_cfg.get("anchor_retention_prob", 0.25))

        self.retention_decay = float(c_cfg.get("retention_decay", 0.55))
        self.min_stage_prob = float(c_cfg.get("min_stage_prob", 0.05))

        step_cfg = c_cfg.get("step_size", {})
        self.difficulty_inc = float(step_cfg.get("difficulty_inc", 0.05))
        self.difficulty_dec = float(step_cfg.get("difficulty_dec", 0.05))

        # Patience settings before shifting distributions
        patience_cfg = c_cfg.get("patience", {})
        self.intervention_patience = int(patience_cfg.get("intervention_patience", 3))
        self.recovery_patience = int(patience_cfg.get("recovery_patience", 3))
        self.promotion_patience = int(patience_cfg.get("promotion_patience", 25))

        # Patience streak counters
        self.prereq_fail_streak = 0
        self.recovery_streak = 0
        self.frontier_mastery_count = 0

        # Intervention state tracking
        self.in_intervention = False
        self.failing_stage: Optional[str] = None

        # Active spawn probability distribution
        self.probabilities: Dict[str, float] = self._compute_probabilities()

    @property
    def frontier_category(self) -> str:
        return self.stages[self.frontier_idx]

    def record_episode(
        self,
        category: str,
        success: bool,
        reward: float,
        length: int,
        distance: float,
        forward_vel: float,
        fall: bool,
    ):
        """Records episode outcome into the stage-specific history window."""
        if category in self.stage_history:
            rec = EpisodeRecord(
                success=success,
                reward=reward,
                length=length,
                distance=distance,
                forward_vel=forward_vel,
                fall=fall,
                difficulty=self.difficulties.get(category, 0.0),
            )
            self.stage_history[category].append(rec)

    def get_stage_metrics(self, category: str) -> Dict[str, float]:
        """Returns performance metrics for a specific terrain category."""
        history = self.stage_history.get(category, deque())
        if not history:
            return {
                "success_rate": 0.0,
                "fall_rate": 0.0,
                "avg_reward": 0.0,
                "avg_length": 0.0,
                "avg_distance": 0.0,
                "avg_velocity": 0.0,
                "sample_count": 0,
            }

        n = len(history)
        successes = sum(1 for e in history if e.success)
        falls = sum(1 for e in history if e.fall)
        tot_reward = sum(e.reward for e in history)
        tot_length = sum(e.length for e in history)
        tot_dist = sum(e.distance for e in history)
        tot_vel = sum(e.forward_vel for e in history)

        return {
            "success_rate": successes / n,
            "fall_rate": falls / n,
            "avg_reward": tot_reward / n,
            "avg_length": tot_length / n,
            "avg_distance": tot_dist / n,
            "avg_velocity": tot_vel / n,
            "sample_count": n,
        }

    def check_prerequisites(self) -> Tuple[bool, Optional[str]]:
        """Checks if all prerequisite stages (levels < frontier_idx) maintain >= prereq_thresh.

        Returns:
            (all_healthy: bool, failing_stage: Optional[str])
        """
        for j in range(self.frontier_idx):
            cat = self.stages[j]
            m = self.get_stage_metrics(cat)
            # Evaluate only if we have at least 10 samples
            if m["sample_count"] >= 10:
                if m["success_rate"] < self.prereq_thresh or m["fall_rate"] > self.max_fall_rate:
                    return False, cat

        return True, None

    def _compute_probabilities(self) -> Dict[str, float]:
        """Computes active spawn probabilities according to frontier and intervention status."""
        probs = {cat: 0.0 for cat in self.stages}

        # Case 1: Initial state (frontier is level 0)
        if self.frontier_idx == 0:
            probs[self.stages[0]] = 1.0
            return probs

        # Case 2: Targeted Intervention Mode
        # A prerequisite dropped below 90% -> assign 75% to failing stage, 25% to anchors
        if self.in_intervention and self.failing_stage is not None:
            unlocked = self.stages[: self.frontier_idx + 1]
            anchors = [c for c in unlocked if c != self.failing_stage]

            probs[self.failing_stage] = self.failing_focus_prob
            if anchors:
                anchor_share = self.anchor_prob / len(anchors)
                for a in anchors:
                    probs[a] = anchor_share
            return probs

        # Case 3: Normal Progression (Frontier Focus with Exponential Retention)
        # Higher locked stages (k > frontier_idx) remain strictly 0.0%
        unlocked = self.stages[: self.frontier_idx + 1]
        K = self.frontier_idx

        raw_weights = []
        for j in range(K + 1):
            w = self.retention_decay ** (K - j)
            raw_weights.append(w)

        total_w = sum(raw_weights)
        norm_weights = [w / total_w for w in raw_weights]

        # Apply minimum retention floor for all unlocked stages
        adjusted = [max(self.min_stage_prob, p) for p in norm_weights]
        tot_adj = sum(adjusted)

        for j in range(K + 1):
            probs[self.stages[j]] = adjusted[j] / tot_adj

        return probs

    def update(self) -> Dict[str, Any]:
        """Executes periodic prerequisite health check, difficulty scaling, and stage promotion."""
        frontier_cat = self.frontier_category
        frontier_metrics = self.get_stage_metrics(frontier_cat)
        f_success = frontier_metrics["success_rate"]
        f_fall = frontier_metrics["fall_rate"]

        # 1. Continuous Difficulty Scaling on Frontier
        if frontier_metrics["sample_count"] >= 10:
            if f_success >= self.frontier_thresh and f_fall <= self.max_fall_rate:
                if self.difficulties[frontier_cat] < 1.0:
                    self.difficulties[frontier_cat] = min(1.0, self.difficulties[frontier_cat] + self.difficulty_inc)
                else:
                    self.frontier_mastery_count += 1
            elif f_success < 0.60 or f_fall > self.max_fall_rate:
                self.difficulties[frontier_cat] = max(0.0, self.difficulties[frontier_cat] - self.difficulty_dec)
                self.frontier_mastery_count = max(0, self.frontier_mastery_count - 2)

        # 2. Continuous Prerequisite Health Check with Patience Hysteresis
        if self.frontier_idx > 0:
            all_prereqs_healthy, failing_cat = self.check_prerequisites()

            if not all_prereqs_healthy and self.intervention_enabled:
                self.prereq_fail_streak += 1
                self.recovery_streak = 0

                # Shift distribution ONLY after sustaining failure across patience window
                if not self.in_intervention and self.prereq_fail_streak >= self.intervention_patience:
                    self.in_intervention = True
                    self.failing_stage = failing_cat
                    print(
                        f"\n⚠️  [PREREQUISITE ALERT - PATIENCE EXPIRED] Prerequisite '{failing_cat}' dropped below {self.prereq_thresh*100:.0f}% success "
                        f"for {self.intervention_patience} consecutive checks! Entering Targeted Intervention (Allocating {self.failing_focus_prob*100:.0f}% focus)."
                    )
            else:
                self.prereq_fail_streak = 0
                if self.in_intervention:
                    self.recovery_streak += 1
                    # Restore normal distribution ONLY after sustaining recovery across patience window
                    if self.recovery_streak >= self.recovery_patience:
                        print(
                            f"\n✓  [PREREQUISITE RECOVERED - PATIENCE CONFIRMED] '{self.failing_stage}' restored to >= {self.prereq_thresh*100:.0f}% success "
                            f"for {self.recovery_patience} consecutive checks. Resuming normal frontier training on '{frontier_cat}'."
                        )
                        self.in_intervention = False
                        self.failing_stage = None
                        self.recovery_streak = 0

        # 3. Frontier Promotion (Unlocking Next Stage with Promotion Patience)
        stage_promoted = False
        all_healthy, _ = self.check_prerequisites()

        if (
            all_healthy
            and not self.in_intervention
            and self.difficulties[frontier_cat] >= 1.0
            and f_success >= self.frontier_thresh
            and self.frontier_mastery_count >= self.promotion_patience
            and self.frontier_idx < self.num_stages - 1
        ):
            old_frontier = frontier_cat
            self.frontier_idx += 1
            new_frontier = self.frontier_category
            self.difficulties[new_frontier] = 0.0
            self.frontier_mastery_count = 0
            self.prereq_fail_streak = 0
            self.recovery_streak = 0
            stage_promoted = True
            print(
                f"\n🎓 [STAGE UNLOCKED] All prerequisites maintained >= {self.prereq_thresh*100:.0f}% success "
                f"and '{old_frontier}' mastered (patience={self.promotion_patience} checks)! Unlocking Stage {self.frontier_idx}: '{new_frontier}' (d=0.0)"
            )

        # Recompute active spawn probabilities
        self.probabilities = self._compute_probabilities()

        all_stage_metrics = {cat: self.get_stage_metrics(cat) for cat in self.stages[: self.frontier_idx + 1]}

        return {
            "frontier_index": self.frontier_idx,
            "frontier_category": self.frontier_category,
            "in_intervention": self.in_intervention,
            "failing_stage": self.failing_stage,
            "stage_promoted": stage_promoted,
            "probabilities": self.probabilities.copy(),
            "difficulties": self.difficulties.copy(),
            "metrics": all_stage_metrics,
        }

    def sample_terrain(self, rng: Optional[np.random.Generator] = None) -> Tuple[str, float]:
        """Samples a singular terrain category and its difficulty according to active probabilities."""
        gen = rng if rng is not None else np.random.default_rng()
        cats = list(self.probabilities.keys())
        probs = [self.probabilities[c] for c in cats]
        chosen = str(gen.choice(cats, p=probs))
        d = self.difficulties.get(chosen, 0.0)
        return chosen, d
