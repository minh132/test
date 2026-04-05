from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentConfig:
    max_steps: int = 15
    max_verify_attempts: int = 2
    temperature: float = 0.7


@dataclass
class AgentRun:
    question: str
    answer: str
    trace: List[str] = field(default_factory=list)
    verified: bool = False
