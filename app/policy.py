from __future__ import annotations

import json
import os
from typing import Dict

from openai import OpenAI

from app.models import Action, Observation


SYSTEM_PROMPT = (
    "You are a customer-support triage agent. Return only compact JSON for the next action."
)


def heuristic_action(observation: Observation) -> Action:
    queue = observation.queue

    for ticket in queue:
        if ticket.issue_type in {"outage", "security"} and ticket.priority != "critical":
            return Action(action_type="set_priority", ticket_id=ticket.ticket_id, priority="critical")

    for ticket in queue:
        if ticket.issue_type == "security" and ticket.assigned_team != "security":
            return Action(action_type="assign_team", ticket_id=ticket.ticket_id, team="security")

    for ticket in queue:
        if ticket.issue_type in {"outage", "bug"} and ticket.assigned_team != "engineering":
            return Action(action_type="assign_team", ticket_id=ticket.ticket_id, team="engineering")

    for ticket in queue:
        if ticket.issue_type in {"billing", "refund"} and ticket.assigned_team != "billing":
            return Action(action_type="assign_team", ticket_id=ticket.ticket_id, team="billing")

    for ticket in queue:
        if ticket.issue_type == "outage" and "outage" not in ticket.labels:
            return Action(action_type="add_label", ticket_id=ticket.ticket_id, label="outage")

    for ticket in queue:
        if ticket.issue_type == "security" and not ticket.escalated:
            return Action(action_type="escalate_ticket", ticket_id=ticket.ticket_id)

    for ticket in queue:
        if not ticket.reply_draft:
            message = "Sorry for the disruption. We are actively investigating and will provide updates shortly."
            if ticket.issue_type == "billing":
                message = "Sorry about the billing issue. We are validating the invoice and will share a correction."
            if ticket.issue_type == "security":
                message = "We are treating this as urgent and have escalated to security. Please rotate credentials now."
            if ticket.issue_type == "refund":
                message = "We received your refund request and are reviewing eligibility under policy."
            if ticket.issue_type == "bug":
                message = "Thanks for reporting this bug. We are collecting repro details and ETA from engineering."
            return Action(action_type="draft_reply", ticket_id=ticket.ticket_id, message=message)

    return Action(action_type="noop")


def llm_action(observation: Observation, model: str) -> Action:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return heuristic_action(observation)

    client = OpenAI(api_key=api_key)
    payload: Dict[str, object] = {
        "task_id": observation.task_id,
        "instruction": observation.instruction,
        "step_count": observation.step_count,
        "max_steps": observation.max_steps,
        "queue": [ticket.model_dump() for ticket in observation.queue],
        "recent_events": observation.recent_events,
    }

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Given the observation below, produce the next best action JSON with fields: "
                    "action_type, ticket_id (if relevant), priority/team/label/message when needed.\n"
                    f"Observation:\n{json.dumps(payload)}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)

    try:
        return Action(**parsed)
    except Exception:
        return heuristic_action(observation)
