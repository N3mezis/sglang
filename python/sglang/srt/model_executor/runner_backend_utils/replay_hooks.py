"""Hooks run after a full CUDA graph replay that can ask for the step to be replayed.

A captured step can depend on host work it cannot do itself, e.g. paged experts reading an
expert from disk that the step routed to. The hook does that work after the replay and asks
for another; the step must be safe to replay (its inputs unchanged, its writes idempotent).
The recurrent (mamba / linear-attention) state a step advances in place is not idempotent:
it is saved before the step and restored before each replay.
"""

from typing import Callable, List, Optional

import torch

_HOOKS: List[Callable[[], bool]] = []


def register_replay_hook(hook: Callable[[], bool]) -> None:
    if hook not in _HOOKS:
        _HOOKS.append(hook)


# A hook's work must settle the step: paged experts fixes at least one more layer per replay.
_MAX_REPLAYS = 1024


def needs_replay() -> bool:
    """Run every hook; whether any asks for the step to be replayed."""
    return any([hook() for hook in _HOOKS])


def save_recurrent_state(
    req_to_token_pool, req_pool_indices: torch.Tensor
) -> Optional[Callable[[], None]]:
    """Copy the recurrent state of the step's requests, if a hook may ask for a replay; the
    returned callable puts it back."""
    mamba_pool = getattr(req_to_token_pool, "mamba_pool", None)
    if not _HOOKS or mamba_pool is None:
        return None
    cache = mamba_pool.mamba_cache
    if mamba_pool._slot_siblings or cache.replayssm_d is not None:
        raise RuntimeError(
            "A replayed CUDA graph step restores the conv and temporal states only, not "
            "slot side states or ReplaySSM rings"
        )
    # Padded rows map to the reserved padding slot: its duplicates hold equal values.
    slots = req_to_token_pool.translate_mamba_indices(
        req_to_token_pool.get_mamba_indices(req_pool_indices)
    ).to(torch.int64)
    states = [s for s in (*cache.conv, cache.temporal) if s.numel()]
    saved = [s.index_select(1, slots) for s in states]

    def restore() -> None:
        for state, copy in zip(states, saved):
            state.index_copy_(1, slots, copy)

    return restore


def replay_until_settled(
    replay: Callable[[], None], restore: Optional[Callable[[], None]] = None
) -> None:
    """``replay`` again while a hook asks for it, after ``restore`` (see
    ``save_recurrent_state``)."""
    for _ in range(_MAX_REPLAYS):
        if not needs_replay():
            return
        if restore is not None:
            restore()
        replay()
    if needs_replay():
        raise RuntimeError(
            f"A CUDA graph replay hook still asks for a replay after {_MAX_REPLAYS} "
            "replays"
        )
