"""M3 policy input. The final axis holds time-ordered observation/action pairs."""


def initial_input(obs, action_width, length, xp):
    if length < 1:
        raise ValueError("history length must be positive")
    shape = (*obs.shape[:-1], length * (obs.shape[-1] + action_width))
    return xp.concatenate((obs, xp.zeros(shape, dtype=obs.dtype)), axis=-1)


def next_input(previous, action, observation, done, xp):
    """Advance one transition. On autoreset, observation is the new reset observation."""
    width = observation.shape[-1]
    pair_width = width + action.shape[-1]
    history = xp.concatenate(
        (previous[..., width + pair_width:], previous[..., :width], action), axis=-1
    )
    return xp.concatenate((observation, xp.where(done[..., None], 0, history)), axis=-1)
