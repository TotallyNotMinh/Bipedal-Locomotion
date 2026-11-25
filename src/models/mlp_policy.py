"""Model 1 (Baseline MLP) and Model 2 (Capacity-Matched Larger MLP) definitions.

Implements:
- Exact analytical parameter counting matching Brax `ppo_networks.make_ppo_networks`
- Automatic capacity matching of M2 against M3 (history stacking) or M4 (history encoder)
- Network factory builders for Brax PPO training on JAX / Dual-T4 GPUs
"""

from dataclasses import dataclass, asdict
import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class ParameterBreakdown:
    """Exact trainable parameter counts for an Actor-Critic PPO pair."""

    model_id: str
    policy_input_dim: int
    value_input_dim: int
    action_dim: int
    policy_hidden_layer_sizes: Tuple[int, ...]
    value_hidden_layer_sizes: Tuple[int, ...]
    policy_params: int
    value_params: int
    total_params: int
    encoder_params: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["policy_hidden_layer_sizes"] = list(self.policy_hidden_layer_sizes)
        d["value_hidden_layer_sizes"] = list(self.value_hidden_layer_sizes)
        return d


@dataclass
class ModelArchitectureSpec:
    """Resolved policy/value network configuration ready for training."""

    model_id: str
    policy_hidden_layer_sizes: Tuple[int, ...]
    value_hidden_layer_sizes: Tuple[int, ...]
    activation: str = "swish"
    distribution_type: str = "tanh_normal"
    policy_obs_key: str = "state"
    value_obs_key: str = "privileged_state"
    obs_dim: int = 0
    value_obs_dim: int = 0
    action_dim: int = 0
    capacity_matched: bool = False
    match_reference: str = ""
    target_policy_params: Optional[int] = None
    actual_policy_params: int = 0
    actual_value_params: int = 0
    actual_total_params: int = 0


def _resolve_obs_dims(
    observation_size: Union[int, Dict[str, Any]],
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
) -> Tuple[int, int]:
    """Extracts scalar policy & value observation dimensions from Brax/Playground obs specs."""
    if isinstance(observation_size, int):
        return int(observation_size), int(observation_size)

    if isinstance(observation_size, dict):
        if policy_obs_key in observation_size:
            p_dim = observation_size[policy_obs_key]
        else:
            p_dim = next(iter(observation_size.values()))

        if value_obs_key in observation_size:
            v_dim = observation_size[value_obs_key]
        elif policy_obs_key in observation_size:
            v_dim = observation_size[policy_obs_key]
        else:
            v_dim = next(iter(observation_size.values()))

        p_size = int(p_dim[0]) if isinstance(p_dim, (tuple, list)) else int(p_dim)
        v_size = int(v_dim[0]) if isinstance(v_dim, (tuple, list)) else int(v_dim)
        return p_size, v_size

    if isinstance(observation_size, (tuple, list)):
        dim = int(observation_size[0])
        return dim, dim

    raise TypeError(f"Unsupported observation_size type: {type(observation_size)}")


def count_mlp_params(
    input_dim: int,
    hidden_sizes: Sequence[int],
    output_dim: int,
    bias: bool = True,
) -> int:
    """Counts trainable parameters (weights + biases) of a Dense MLP."""
    layer_dims = [int(input_dim), *[int(h) for h in hidden_sizes], int(output_dim)]
    total = 0
    for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:]):
        total += in_d * out_d
        if bias:
            total += out_d
    return total


def count_actor_critic_params(
    policy_input_dim: int,
    value_input_dim: int,
    action_dim: int,
    policy_hidden_layer_sizes: Sequence[int],
    value_hidden_layer_sizes: Sequence[int],
    distribution_type: str = "tanh_normal",
    state_dependent_std: bool = False,
    bias: bool = True,
) -> Tuple[int, int, int]:
    """Computes exact trainable parameters matching Brax `make_ppo_networks`.

    In Brax:
    - `tanh_normal` policy outputs `2 * action_dim` (mean + log_std) from the final linear layer.
    - `normal` policy uses MLP trunk (`activate_final=True`) -> Dense(action_dim) mean +
      either Dense(action_dim) std (if state_dependent_std) or a learned 1D std vector (action_dim).
    - Value network outputs scalar `1`.
    """
    if distribution_type == "tanh_normal":
        policy_params = count_mlp_params(
            policy_input_dim, policy_hidden_layer_sizes, 2 * action_dim, bias=bias
        )
    elif distribution_type == "normal":
        hidden = [int(h) for h in policy_hidden_layer_sizes]
        if not hidden:
            trunk_out = int(policy_input_dim)
            trunk_params = 0
        else:
            trunk_out = hidden[-1]
            trunk_params = count_mlp_params(
                policy_input_dim, hidden[:-1], trunk_out, bias=bias
            )
        mean_head = trunk_out * action_dim + (action_dim if bias else 0)
        if state_dependent_std:
            std_head = trunk_out * action_dim + (action_dim if bias else 0)
        else:
            std_head = action_dim
        policy_params = trunk_params + mean_head + std_head
    else:
        raise ValueError(f"Unsupported distribution_type: {distribution_type}")

    value_params = count_mlp_params(
        value_input_dim, value_hidden_layer_sizes, 1, bias=bias
    )
    return policy_params, value_params, policy_params + value_params


def estimate_m3_actor_critic_params(
    obs_dim: int,
    value_obs_dim: int,
    action_dim: int,
    history_length: int = 5,
    include_actions: bool = True,
    policy_hidden_layer_sizes: Sequence[int] = (512, 256, 128),
    value_hidden_layer_sizes: Sequence[int] = (512, 256, 128),
    distribution_type: str = "tanh_normal",
    critic_uses_history: bool = False,
) -> ParameterBreakdown:
    """Computes reference parameter count for Model 3 (Causal History Stacking)."""
    pair_dim = obs_dim + (action_dim if include_actions else 0)
    m3_policy_in = obs_dim + history_length * pair_dim
    m3_value_in = (
        value_obs_dim + history_length * pair_dim if critic_uses_history else value_obs_dim
    )

    p_params, v_params, tot_params = count_actor_critic_params(
        policy_input_dim=m3_policy_in,
        value_input_dim=m3_value_in,
        action_dim=action_dim,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        distribution_type=distribution_type,
    )
    return ParameterBreakdown(
        model_id="M3",
        policy_input_dim=m3_policy_in,
        value_input_dim=m3_value_in,
        action_dim=action_dim,
        policy_hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(value_hidden_layer_sizes),
        policy_params=p_params,
        value_params=v_params,
        total_params=tot_params,
        notes=f"history_len={history_length}, include_actions={include_actions}",
    )


def estimate_m4_actor_critic_params(
    obs_dim: int,
    value_obs_dim: int,
    action_dim: int,
    history_length: int = 5,
    include_actions: bool = True,
    encoder_hidden_sizes: Sequence[int] = (256, 128),
    latent_dim: int = 32,
    policy_hidden_layer_sizes: Sequence[int] = (512, 256, 128),
    value_hidden_layer_sizes: Sequence[int] = (512, 256, 128),
    distribution_type: str = "tanh_normal",
) -> ParameterBreakdown:
    """Computes reference parameter count for Model 4 (Learned MLP History Encoder)."""
    pair_dim = obs_dim + (action_dim if include_actions else 0)
    hist_flat_dim = history_length * pair_dim
    enc_params = count_mlp_params(hist_flat_dim, encoder_hidden_sizes, latent_dim)
    m4_policy_in = obs_dim + latent_dim

    p_head, v_params, _ = count_actor_critic_params(
        policy_input_dim=m4_policy_in,
        value_input_dim=value_obs_dim,
        action_dim=action_dim,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        distribution_type=distribution_type,
    )
    p_total = p_head + enc_params
    return ParameterBreakdown(
        model_id="M4",
        policy_input_dim=obs_dim,
        value_input_dim=value_obs_dim,
        action_dim=action_dim,
        policy_hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(value_hidden_layer_sizes),
        policy_params=p_total,
        value_params=v_params,
        total_params=p_total + v_params,
        encoder_params=enc_params,
        notes=f"encoder={list(encoder_hidden_sizes)}->{latent_dim}, history_len={history_length}",
    )


def find_capacity_matched_hidden_sizes(
    input_dim: int,
    output_dim: int,
    target_params: int,
    base_ratios: Sequence[float] = (4.0, 2.0, 1.0),
    width_step: int = 16,
    min_width: int = 64,
    max_width: int = 4096,
    distribution_type: str = "tanh_normal",
) -> Tuple[int, ...]:
    """Finds tapered MLP hidden widths aligned to `width_step` matching `target_params`.

    Scales `base_ratios` uniformly by a multiplier `alpha`, then refines the first
    layer width within a small neighborhood to minimize |actual_params - target_params|.
    """
    ratios = [float(r) for r in base_ratios]
    norm = max(ratios[-1], 1e-6)
    ratios = [r / norm for r in ratios]

    def _round_step(val: float) -> int:
        rounded = int(round(val / width_step) * width_step)
        return max(min_width, min(max_width, rounded))

    def _eval_policy(hidden: Sequence[int]) -> int:
        if distribution_type == "tanh_normal":
            return count_mlp_params(input_dim, hidden, 2 * output_dim)
        p_params, _, _ = count_actor_critic_params(
            policy_input_dim=input_dim,
            value_input_dim=input_dim,
            action_dim=output_dim,
            policy_hidden_layer_sizes=hidden,
            value_hidden_layer_sizes=hidden,
            distribution_type=distribution_type,
        )
        return p_params

    # Binary search on scale factor alpha
    lo, hi = 1.0, float(max_width)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cand = tuple(_round_step(mid * r) for r in ratios)
        if _eval_policy(cand) < target_params:
            lo = mid
        else:
            hi = mid

    # Search nearby grid candidates for closest parameter match while maintaining monotonic taper
    alpha_center = 0.5 * (lo + hi)
    best_hidden = tuple(_round_step(alpha_center * r) for r in ratios)
    best_diff = abs(_eval_policy(best_hidden) - target_params)

    for delta in range(-4, 5):
        base_cand = [_round_step(alpha_center * r + delta * width_step) for r in ratios]
        # Fine-tune first hidden layer since input_dim * h0 shifts params smoothly
        for d0 in range(-6, 7):
            cand = list(base_cand)
            cand[0] = max(min_width, min(max_width, cand[0] + d0 * width_step))
            # Prefer non-increasing taper (h0 >= h1 >= ...)
            if any(cand[i] < cand[i + 1] for i in range(len(cand) - 1)):
                continue
            diff = abs(_eval_policy(cand) - target_params)
            if diff < best_diff:
                best_diff = diff
                best_hidden = tuple(cand)

    return best_hidden


def build_model_spec_from_config(
    config: Dict[str, Any],
    observation_size: Union[int, Dict[str, Any]],
    action_size: int,
) -> ModelArchitectureSpec:
    """Builds a resolved ModelArchitectureSpec for M1 or M2 from YAML config."""
    net_cfg = config.get("network", {})
    model_id = str(net_cfg.get("model_id", "M2")).upper()
    policy_obs_key = str(net_cfg.get("policy_obs_key", "state"))
    value_obs_key = str(net_cfg.get("value_obs_key", "privileged_state"))
    activation = str(net_cfg.get("activation", "swish"))
    distribution_type = str(net_cfg.get("distribution_type", "tanh_normal"))

    obs_dim, val_dim = _resolve_obs_dims(observation_size, policy_obs_key, value_obs_key)
    policy_hidden = tuple(int(x) for x in net_cfg.get("policy_hidden_layer_sizes", (512, 256, 128)))
    value_hidden = tuple(int(x) for x in net_cfg.get("value_hidden_layer_sizes", (512, 256, 128)))

    capacity_matched = False
    match_reference = ""
    target_policy_params: Optional[int] = None

    cm_cfg = net_cfg.get("capacity_matching", {})
    if model_id == "M2" and bool(cm_cfg.get("auto_match", False)):
        explicit_target = cm_cfg.get("target_policy_params")
        ref_model = str(cm_cfg.get("reference_model", "M3")).upper()
        ref_p_hidden = tuple(int(x) for x in cm_cfg.get("reference_policy_hidden", (512, 256, 128)))
        ref_v_hidden = tuple(int(x) for x in cm_cfg.get("reference_value_hidden", (512, 256, 128)))
        hist_len = int(cm_cfg.get("history_length", 5))
        inc_actions = bool(cm_cfg.get("include_actions", True))
        width_step = int(cm_cfg.get("width_step", 16))
        num_layers = int(cm_cfg.get("num_layers", len(ref_p_hidden)))

        if explicit_target is not None:
            target_policy_params = int(explicit_target)
            match_reference = f"explicit({target_policy_params})"
        elif ref_model == "M4":
            m4_ref = estimate_m4_actor_critic_params(
                obs_dim=obs_dim,
                value_obs_dim=val_dim,
                action_dim=action_size,
                history_length=hist_len,
                include_actions=inc_actions,
                policy_hidden_layer_sizes=ref_p_hidden,
                value_hidden_layer_sizes=ref_v_hidden,
                distribution_type=distribution_type,
            )
            target_policy_params = m4_ref.policy_params
            match_reference = f"M4(k={hist_len})"
        else:
            m3_ref = estimate_m3_actor_critic_params(
                obs_dim=obs_dim,
                value_obs_dim=val_dim,
                action_dim=action_size,
                history_length=hist_len,
                include_actions=inc_actions,
                policy_hidden_layer_sizes=ref_p_hidden,
                value_hidden_layer_sizes=ref_v_hidden,
                distribution_type=distribution_type,
            )
            target_policy_params = m3_ref.policy_params
            match_reference = f"M3(k={hist_len})"

        # Use tapered ratios matching `num_layers` (e.g., [4, 2, 1] for 3 layers)
        if len(ref_p_hidden) == num_layers:
            ratios = [float(x) for x in ref_p_hidden]
        else:
            ratios = [float(2 ** (num_layers - 1 - i)) for i in range(num_layers)]

        policy_hidden = find_capacity_matched_hidden_sizes(
            input_dim=obs_dim,
            output_dim=action_size,
            target_params=target_policy_params,
            base_ratios=ratios,
            width_step=width_step,
            distribution_type=distribution_type,
        )
        capacity_matched = True

        if bool(cm_cfg.get("match_critic", False)) and target_policy_params is not None:
            value_hidden = find_capacity_matched_hidden_sizes(
                input_dim=val_dim,
                output_dim=1,
                target_params=count_mlp_params(val_dim, ref_v_hidden, 1),
                base_ratios=ratios,
                width_step=width_step,
                distribution_type="tanh_normal",
            )

    p_params, v_params, tot_params = count_actor_critic_params(
        policy_input_dim=obs_dim,
        value_input_dim=val_dim,
        action_dim=action_size,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
        distribution_type=distribution_type,
    )

    return ModelArchitectureSpec(
        model_id=model_id,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=value_hidden,
        activation=activation,
        distribution_type=distribution_type,
        policy_obs_key=policy_obs_key,
        value_obs_key=value_obs_key,
        obs_dim=obs_dim,
        value_obs_dim=val_dim,
        action_dim=action_size,
        capacity_matched=capacity_matched,
        match_reference=match_reference,
        target_policy_params=target_policy_params,
        actual_policy_params=p_params,
        actual_value_params=v_params,
        actual_total_params=tot_params,
    )


def build_parameter_comparison_table(
    spec: ModelArchitectureSpec,
    baseline_policy_hidden: Sequence[int] = (512, 256, 128),
    baseline_value_hidden: Sequence[int] = (512, 256, 128),
    history_length: int = 5,
    include_actions: bool = True,
) -> List[ParameterBreakdown]:
    """Generates a side-by-side parameter comparison across M1, M2, M3, and M4."""
    m1_p, m1_v, m1_tot = count_actor_critic_params(
        policy_input_dim=spec.obs_dim,
        value_input_dim=spec.value_obs_dim,
        action_dim=spec.action_dim,
        policy_hidden_layer_sizes=baseline_policy_hidden,
        value_hidden_layer_sizes=baseline_value_hidden,
        distribution_type=spec.distribution_type,
    )
    m1_row = ParameterBreakdown(
        model_id="M1 (Baseline)",
        policy_input_dim=spec.obs_dim,
        value_input_dim=spec.value_obs_dim,
        action_dim=spec.action_dim,
        policy_hidden_layer_sizes=tuple(baseline_policy_hidden),
        value_hidden_layer_sizes=tuple(baseline_value_hidden),
        policy_params=m1_p,
        value_params=m1_v,
        total_params=m1_tot,
        notes="Current o_t only",
    )

    m2_row = ParameterBreakdown(
        model_id=f"{spec.model_id} (Active)" if spec.model_id == "M2" else "M2 (Config)",
        policy_input_dim=spec.obs_dim,
        value_input_dim=spec.value_obs_dim,
        action_dim=spec.action_dim,
        policy_hidden_layer_sizes=spec.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=spec.value_hidden_layer_sizes,
        policy_params=spec.actual_policy_params,
        value_params=spec.actual_value_params,
        total_params=spec.actual_total_params,
        notes=f"Matched to {spec.match_reference}" if spec.capacity_matched else "Explicit widths",
    )

    m3_row = estimate_m3_actor_critic_params(
        obs_dim=spec.obs_dim,
        value_obs_dim=spec.value_obs_dim,
        action_dim=spec.action_dim,
        history_length=history_length,
        include_actions=include_actions,
        policy_hidden_layer_sizes=baseline_policy_hidden,
        value_hidden_layer_sizes=baseline_value_hidden,
        distribution_type=spec.distribution_type,
    )

    m4_row = estimate_m4_actor_critic_params(
        obs_dim=spec.obs_dim,
        value_obs_dim=spec.value_obs_dim,
        action_dim=spec.action_dim,
        history_length=history_length,
        include_actions=include_actions,
        policy_hidden_layer_sizes=baseline_policy_hidden,
        value_hidden_layer_sizes=baseline_value_hidden,
        distribution_type=spec.distribution_type,
    )
    return [m1_row, m2_row, m3_row, m4_row]


def format_parameter_table(rows: Sequence[ParameterBreakdown]) -> str:
    """Formats a ParameterBreakdown list into a readable ASCII table."""
    header = (
        f"{'Model':<16} | {'Policy In':>9} | {'Policy Hidden':<18} | "
        f"{'Policy Params':>13} | {'Value Params':>12} | {'Total Params':>12}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        hidden_str = str(list(r.policy_hidden_layer_sizes))
        lines.append(
            f"{r.model_id:<16} | {r.policy_input_dim:>9} | {hidden_str:<18} | "
            f"{r.policy_params:>13,} | {r.value_params:>12,} | {r.total_params:>12,}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _get_flax_activation(name: str):
    """Maps string activation name to flax.linen activation function."""
    from flax import linen as nn

    mapping = {
        "swish": nn.swish,
        "silu": nn.silu,
        "elu": nn.elu,
        "relu": nn.relu,
        "tanh": nn.tanh,
        "gelu": nn.gelu,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Supported: {list(mapping.keys())}")
    return mapping[key]


def make_model_network_factory(
    spec: ModelArchitectureSpec,
    has_dict_obs: bool = True,
):
    """Returns a Brax `network_factory` callable configured for M1 or M2."""
    from brax.training.agents.ppo import networks as ppo_networks

    activation_fn = _get_flax_activation(spec.activation)
    kwargs: Dict[str, Any] = {
        "policy_hidden_layer_sizes": spec.policy_hidden_layer_sizes,
        "value_hidden_layer_sizes": spec.value_hidden_layer_sizes,
        "activation": activation_fn,
        "distribution_type": spec.distribution_type,
    }
    if has_dict_obs:
        kwargs["policy_obs_key"] = spec.policy_obs_key
        kwargs["value_obs_key"] = spec.value_obs_key

    return functools.partial(ppo_networks.make_ppo_networks, **kwargs)
