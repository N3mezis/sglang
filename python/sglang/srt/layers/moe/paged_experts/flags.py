"""Config resolution for paged-experts knobs that are exposed BOTH as server arguments and as
environment variables.

Precedence: explicit server argument (not None) > environment variable > built-in default. The env
fallback keeps existing deployments (compose ``.env`` files) working while the server argument is the
documented, discoverable interface. Resolution is lazy — module-level ``os.environ.get`` constants
freeze before the server args exist, so call sites resolve through here at first use instead.
"""

import os
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

_cache: dict = {}


def resolve(
    *,
    arg_name: str,
    env_name: str,
    cast: Callable[[str], T],
    default: T,
    cache_key: Optional[str] = None,
) -> T:
    """Resolve one knob with server-arg > env > default precedence. Cached after first resolution
    (these are boot-stable settings; per-step reads must not pay an attribute walk)."""
    key = cache_key or arg_name
    if key in _cache:
        return _cache[key]
    value: T = default
    env = os.environ.get(env_name)
    if env not in (None, ""):
        value = cast(env)
    try:
        from sglang.srt.server_args import get_global_server_args

        arg = getattr(get_global_server_args(), arg_name, None)
        if arg is not None:
            value = cast(arg)
    except Exception:
        pass  # engine-less contexts (offline tests) fall back to env/default
    _cache[key] = value
    return value
