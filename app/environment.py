from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from app.models import Action, EnvStepResult, Observation, Reward, TaskSummary, Ticket
from app.tasks import TaskDefinition, get_tasks


class SupportTriageEnv:
    def __init__(self, default_task_id: str = "easy") -> None:
        self.tasks: Dict[str, TaskDefinition] = get_tasks()
        if default_task_id not in self.tasks:
            raise ValueError(f"Unknown task id: {default_task_id}")

        self.current_task: TaskDefinition = self.tasks[default_task_id]
        self.queue: List[Ticket] = []
        self.step_count: int = 0
        self.done: bool = False
        self.last_score: float = 0.0
        self.recent_events: List[str] = []
        self.action_history: List[str] = []

        self.reset(default_task_id)

    def list_tasks(self) -> List[TaskSummary]:
        return [
            TaskSummary(
                task_id=task.task_id,
                name=task.name,
                difficulty=task.difficulty,
                objective=task.objective,
            )
            for task in self.tasks.values()
        ]

    def _build_observation(self) -> Observation:
        return Observation(
            task_id=self.current_task.task_id,
            task_name=self.current_task.name,
            instruction=self.current_task.instruction,
            step_count=self.step_count,
            max_steps=self.current_task.max_steps,
            queue=deepcopy(self.queue),
            recent_events=self.recent_events[-5:],
        )

    def _find_ticket(self, ticket_id: str | None) -> Ticket | None:
        if not ticket_id:
            return None
        for ticket in self.queue:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _apply_action(self, action: Action) -> Tuple[bool, str]:
        if action.action_type == "noop":
            return True, "No action taken"

        ticket = self._find_ticket(action.ticket_id)
        if ticket is None:
            return False, "Invalid or missing ticket_id"

        if action.action_type == "set_priority":
            if not action.priority:
                return False, "Missing priority"
            ticket.priority = action.priority
            return True, f"Set {ticket.ticket_id} priority to {action.priority}"

        if action.action_type == "assign_team":
            if not action.team:
                return False, "Missing team"
            ticket.assigned_team = action.team
            return True, f"Assigned {ticket.ticket_id} to {action.team}"

        if action.action_type == "add_label":
            if not action.label:
                return False, "Missing label"
            label = action.label.strip().lower()
            if label and label not in ticket.labels:
                ticket.labels.append(label)
            return True, f"Added label '{label}' to {ticket.ticket_id}"

        if action.action_type == "draft_reply":
            if not action.message:
                return False, "Missing message"
            ticket.reply_draft = action.message.strip()
            return True, f"Drafted response for {ticket.ticket_id}"

        if action.action_type == "resolve_ticket":
            ticket.resolved = True
            return True, f"Resolved {ticket.ticket_id}"

        if action.action_type == "escalate_ticket":
            ticket.escalated = True
            return True, f"Escalated {ticket.ticket_id}"

        return False, "Unknown action_type"

    def _reward_for_transition(
        self,
        previous_score: float,
        current_score: float,
        action: Action,
        valid_action: bool,
    ) -> Reward:
        delta = current_score - previous_score

        progress = max(-0.25, min(0.5, delta))
        step_cost = -0.02
        invalid_penalty = -0.15 if not valid_action else 0.0

        action_signature = action.model_dump_json()
        loop_penalty = -0.05 if action_signature in self.action_history[-3:] else 0.0

        safety_penalty = 0.0
        if action.action_type == "resolve_ticket":
            ticket = self._find_ticket(action.ticket_id)
            if ticket and not ticket.reply_draft:
                safety_penalty -= 0.2

        value = progress + step_cost + invalid_penalty + loop_penalty + safety_penalty
        value = max(-1.0, min(1.0, value))

        rationale = "Progress-based reward with penalties for invalid/repetitive/unsafe actions"
        components = {
            "progress_delta": round(progress, 4),
            "step_cost": step_cost,
            "invalid_penalty": invalid_penalty,
            "loop_penalty": loop_penalty,
            "safety_penalty": safety_penalty,
        }
        return Reward(value=round(value, 4), components=components, rationale=rationale)

    def reset(self, task_id: str | None = None) -> Observation:
        if task_id:
            if task_id not in self.tasks:
                raise ValueError(f"Unknown task id: {task_id}")
            self.current_task = self.tasks[task_id]

        self.queue = deepcopy(self.current_task.initial_tickets)
        self.step_count = 0
        self.done = False
        self.last_score = 0.0
        self.recent_events = [f"Environment reset for task '{self.current_task.task_id}'"]
        self.action_history = []
        return self._build_observation()

    def state(self) -> Dict[str, Any]:
        return {
            "task": {
                "task_id": self.current_task.task_id,
                "name": self.current_task.name,
                "difficulty": self.current_task.difficulty,
                "objective": self.current_task.objective,
                "instruction": self.current_task.instruction,
                "max_steps": self.current_task.max_steps,
            },
            "step_count": self.step_count,
            "done": self.done,
            "current_score": round(self.current_task.grader(self.queue), 4),
            "queue": [ticket.model_dump() for ticket in self.queue],
            "recent_events": self.recent_events[-10:],
        }

    def grade(self) -> float:
        return round(self.current_task.grader(self.queue), 4)

    def step(self, action: Action) -> EnvStepResult:
        if self.done:
            reward = Reward(
                value=-0.05,
                components={"post_done_penalty": -0.05},
                rationale="Episode already done",
            )
            return EnvStepResult(
                observation=self._build_observation(),
                reward=reward,
                done=True,
                info={"score": self.grade(), "message": "Episode already completed"},
            )

        previous_score = self.current_task.grader(self.queue)
        valid_action, event = self._apply_action(action)
        self.step_count += 1
        self.recent_events.append(event)
        self.action_history.append(action.model_dump_json())

        current_score = self.current_task.grader(self.queue)
        reward = self._reward_for_transition(previous_score, current_score, action, valid_action)
        self.last_score = current_score

        task_complete = current_score >= 0.98
        max_steps_reached = self.step_count >= self.current_task.max_steps
        self.done = task_complete or max_steps_reached

        info = {
            "score": round(current_score, 4),
            "valid_action": valid_action,
            "task_complete": task_complete,
            "max_steps_reached": max_steps_reached,
            "grader": self.current_task.grader.__name__,
        }

        return EnvStepResult(
            observation=self._build_observation(),
            reward=reward,
            done=self.done,
            info=info,
        )
