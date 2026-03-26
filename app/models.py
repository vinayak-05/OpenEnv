from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    ticket_id: str
    customer_name: str
    customer_tier: Literal["free", "pro", "enterprise"]
    subject: str
    body: str
    sentiment: Literal["calm", "frustrated", "angry"]
    issue_type: Literal[
        "billing", "outage", "feature_request", "bug", "refund", "security"
    ]
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    assigned_team: Optional[Literal["billing", "support", "engineering", "security"]] = None
    labels: List[str] = Field(default_factory=list)
    reply_draft: Optional[str] = None
    resolved: bool = False
    escalated: bool = False


class Observation(BaseModel):
    task_id: str
    task_name: str
    instruction: str
    step_count: int
    max_steps: int
    queue: List[Ticket]
    recent_events: List[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: Literal[
        "set_priority",
        "assign_team",
        "add_label",
        "draft_reply",
        "resolve_ticket",
        "escalate_ticket",
        "noop",
    ]
    ticket_id: Optional[str] = None
    priority: Optional[Literal["low", "medium", "high", "critical"]] = None
    team: Optional[Literal["billing", "support", "engineering", "security"]] = None
    label: Optional[str] = None
    message: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=-1.0, le=1.0)
    components: Dict[str, float] = Field(default_factory=dict)
    rationale: str


class EnvStepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskSummary(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str


class ResetRequest(BaseModel):
    task_id: str = "easy"


class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"
    max_steps: int = 12
