"""Policy network architectures and parameter-capacity matching utilities."""

from src.models.mlp_policy import (
    ModelArchitectureSpec,
    ParameterBreakdown,
    build_model_spec_from_config,
    build_parameter_comparison_table,
    count_actor_critic_params,
    count_mlp_params,
    estimate_m3_actor_critic_params,
    estimate_m4_actor_critic_params,
    find_capacity_matched_hidden_sizes,
    format_parameter_table,
    make_model_network_factory,
)

__all__ = [
    "ModelArchitectureSpec",
    "ParameterBreakdown",
    "build_model_spec_from_config",
    "build_parameter_comparison_table",
    "count_actor_critic_params",
    "count_mlp_params",
    "estimate_m3_actor_critic_params",
    "estimate_m4_actor_critic_params",
    "find_capacity_matched_hidden_sizes",
    "format_parameter_table",
    "make_model_network_factory",
]
