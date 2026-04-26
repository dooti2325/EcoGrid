from typing import Any, Dict
from uuid import uuid4
from pydantic import BaseModel, Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from env.action_utils import coerce_grid_action
from env.environment import EcoGridEnv
from models.schemas import GridAction, GridState

class ServerObservation(BaseModel):
    observation: GridState
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

class ServerEcoGridEnv(Environment):
    """
    Wrapper around EcoGridEnv to strictly satisfy openenv.core.Environment
    interfaces without breaking the local UI/CLI scripts.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._env = EcoGridEnv()
        self._oe_state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task = "easy"

    def reset(self) -> ServerObservation:
        self._oe_state = State(episode_id=str(uuid4()), step_count=0)
        # Default reset. The specific task is usually set prior, or defaults to easy.
        initial_state = self._env.reset(task=self._current_task, seed=42)
        return ServerObservation(
            observation=initial_state,
            reward=0.0,
            done=False,
            info={}
        )

    def step(self, action: GridAction | dict) -> ServerObservation:
        self._oe_state.step_count += 1
        safe_default = GridAction(
            renewable_ratio=0.5,
            fossil_ratio=0.5,
            battery_action=0.0,
        )
        parsed_action, action_warning = coerce_grid_action(
            action_like=action,
            default_action=safe_default,
        )
        try:
            result = self._env.step(parsed_action)
        except Exception as exc:
            # Never crash the API on malformed/edge payloads.
            fallback_state = self._env.state() if not self._env.is_done else self._env.reset()
            return ServerObservation(
                observation=fallback_state,
                reward=0.001,
                done=self._env.is_done,
                info={
                    "error": f"step_failed:{type(exc).__name__}",
                    "detail": str(exc),
                    "action_warning": action_warning or "step_exception_fallback",
                },
            )
        info = dict(result.info)
        if action_warning:
            info["action_warning"] = action_warning
        return ServerObservation(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=info,
        )

    @property
    def state(self) -> State:
        return self._oe_state
