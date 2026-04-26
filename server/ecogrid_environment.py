from typing import Any, Dict
from uuid import uuid4
from pydantic import BaseModel, Field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

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

    def step(self, action: GridAction) -> ServerObservation:
        self._oe_state.step_count += 1
        result = self._env.step(action)
        return ServerObservation(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info
        )

    @property
    def state(self) -> State:
        return self._oe_state
