"""
Action utility helpers for safety and normalization.

These helpers keep action generation robust across:
- model output parsing
- heuristic controllers
- API payload conversion
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from models.schemas import GridAction


def normalize_action_components(
    renewable_ratio: float,
    fossil_ratio: float,
    battery_action: float,
    ndigits: int = 3,
) -> tuple[float, float, float]:
    """Clamp + normalize action components and keep sum <= 1 after rounding."""
    ren = max(0.0, min(1.0, float(renewable_ratio)))
    fos = max(0.0, min(1.0, float(fossil_ratio)))
    bat = max(-1.0, min(1.0, float(battery_action)))

    total = ren + fos
    if total > 1.0 and total > 0:
        ren /= total
        fos /= total

    # Round for consistent logging/UI while preserving constraints.
    ren = round(ren, ndigits)
    fos = round(fos, ndigits)
    bat = round(bat, ndigits)

    total_rounded = ren + fos
    if total_rounded > 1.0:
        overflow = round(total_rounded - 1.0, ndigits + 2)
        # Remove overflow from fossil first, then renewable.
        reduce_fos = min(fos, overflow)
        fos = round(fos - reduce_fos, ndigits)
        overflow = round(overflow - reduce_fos, ndigits + 2)
        if overflow > 0:
            ren = round(max(0.0, ren - overflow), ndigits)

    return ren, fos, bat


def safe_grid_action(
    renewable_ratio: float,
    fossil_ratio: float,
    battery_action: float,
    ndigits: int = 3,
) -> GridAction:
    """Build a validated GridAction after normalization."""
    ren, fos, bat = normalize_action_components(
        renewable_ratio=renewable_ratio,
        fossil_ratio=fossil_ratio,
        battery_action=battery_action,
        ndigits=ndigits,
    )
    return GridAction(
        renewable_ratio=ren,
        fossil_ratio=fos,
        battery_action=bat,
    )


def coerce_grid_action(
    action_like: Any,
    default_action: Optional[GridAction] = None,
) -> Tuple[GridAction, Optional[str]]:
    """Convert action payloads to a valid GridAction with graceful fallback."""
    try:
        if isinstance(action_like, GridAction):
            return action_like, None

        if isinstance(action_like, dict):
            payload = action_like.get("action", action_like)
            return safe_grid_action(
                renewable_ratio=payload.get("renewable_ratio", 0.5),
                fossil_ratio=payload.get("fossil_ratio", 0.5),
                battery_action=payload.get("battery_action", 0.0),
            ), None
    except Exception as exc:
        if default_action is not None:
            return default_action, f"invalid_action_payload: {type(exc).__name__}"
        raise

    if default_action is not None:
        return default_action, "invalid_action_type"

    raise ValueError("Unable to coerce action payload into GridAction.")
