from env.action_utils import coerce_grid_action, normalize_action_components, safe_grid_action
from models.schemas import GridAction


def test_normalize_action_components_rounding_never_exceeds_one():
    # Regression case from baseline heuristic:
    # 0.396 + 0.605 -> 1.001 after rounding.
    renewable, fossil, battery = normalize_action_components(
        renewable_ratio=0.396,
        fossil_ratio=0.605,
        battery_action=0.0,
    )
    assert renewable + fossil <= 1.0
    assert -1.0 <= battery <= 1.0


def test_safe_grid_action_clamps_and_normalizes():
    action = safe_grid_action(
        renewable_ratio=1.7,
        fossil_ratio=0.8,
        battery_action=-1.7,
    )
    assert isinstance(action, GridAction)
    assert 0.0 <= action.renewable_ratio <= 1.0
    assert 0.0 <= action.fossil_ratio <= 1.0
    assert action.renewable_ratio + action.fossil_ratio <= 1.0
    assert -1.0 <= action.battery_action <= 1.0


def test_coerce_grid_action_invalid_payload_falls_back():
    default = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
    action, warning = coerce_grid_action({"renewable_ratio": "invalid"}, default_action=default)
    assert action == default
    assert warning is not None
