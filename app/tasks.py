from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from app.models import Ticket


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    name: str
    difficulty: str
    objective: str
    instruction: str
    max_steps: int
    initial_tickets: List[Ticket]
    grader: Callable[[List[Ticket]], float]


def _find(queue: List[Ticket], ticket_id: str) -> Ticket | None:
    for ticket in queue:
        if ticket.ticket_id == ticket_id:
            return ticket
    return None


def grade_easy(queue: List[Ticket]) -> float:
    ticket = _find(queue, "T-EASY-1")
    if not ticket:
        return 0.0

    score = 0.0
    if ticket.priority == "critical":
        score += 0.25
    if ticket.assigned_team == "engineering":
        score += 0.25
    if "outage" in ticket.labels:
        score += 0.15
    if ticket.reply_draft and "sorry" in ticket.reply_draft.lower():
        score += 0.2
    if ticket.escalated:
        score += 0.1
    if ticket.resolved:
        score -= 0.2

    return max(0.0, min(1.0, score))


def grade_medium(queue: List[Ticket]) -> float:
    billing = _find(queue, "T-MED-1")
    fraud = _find(queue, "T-MED-2")
    feature = _find(queue, "T-MED-3")
    if not billing or not fraud or not feature:
        return 0.0

    score = 0.0

    if billing.assigned_team == "billing" and billing.priority in {"high", "critical"}:
        score += 0.25
    if billing.reply_draft and "invoice" in billing.reply_draft.lower():
        score += 0.15

    if fraud.assigned_team == "security" and fraud.escalated:
        score += 0.25
    if fraud.priority == "critical":
        score += 0.15

    if feature.priority == "low" and feature.assigned_team == "support":
        score += 0.1

    if feature.resolved:
        score -= 0.1

    return max(0.0, min(1.0, score))


def grade_hard(queue: List[Ticket]) -> float:
    outage = _find(queue, "T-HARD-1")
    security = _find(queue, "T-HARD-2")
    refund = _find(queue, "T-HARD-3")
    angry_bug = _find(queue, "T-HARD-4")
    if not outage or not security or not refund or not angry_bug:
        return 0.0

    score = 0.0

    if outage.priority == "critical" and outage.assigned_team == "engineering" and outage.escalated:
        score += 0.25
    if outage.reply_draft and "status page" in outage.reply_draft.lower():
        score += 0.1

    if security.assigned_team == "security" and security.escalated and security.priority == "critical":
        score += 0.2

    if refund.assigned_team == "billing" and refund.priority in {"medium", "high"}:
        score += 0.15
    if refund.reply_draft and "refund" in refund.reply_draft.lower():
        score += 0.1

    if angry_bug.assigned_team == "engineering" and angry_bug.priority in {"high", "critical"}:
        score += 0.1
    if angry_bug.reply_draft and "repro" in angry_bug.reply_draft.lower():
        score += 0.1

    unsafe_resolves = sum(1 for t in queue if t.resolved and not t.reply_draft)
    score -= 0.05 * unsafe_resolves

    return max(0.0, min(1.0, score))


def get_tasks() -> Dict[str, TaskDefinition]:
    easy_tickets = [
        Ticket(
            ticket_id="T-EASY-1",
            customer_name="Delta Dental Group",
            customer_tier="enterprise",
            subject="Production dashboard unavailable",
            body="All clinic dashboards show 503 since 09:12 UTC. Doctors cannot review patient queues.",
            sentiment="angry",
            issue_type="outage",
        ),
        Ticket(
            ticket_id="T-EASY-2",
            customer_name="Ariana P.",
            customer_tier="free",
            subject="How do I export CSV?",
            body="Can't find export on mobile app.",
            sentiment="calm",
            issue_type="feature_request",
        ),
    ]

    medium_tickets = [
        Ticket(
            ticket_id="T-MED-1",
            customer_name="Rutherford Logistics",
            customer_tier="pro",
            subject="Duplicate invoice this month",
            body="We were billed twice for March seats. Need corrected invoice today.",
            sentiment="frustrated",
            issue_type="billing",
        ),
        Ticket(
            ticket_id="T-MED-2",
            customer_name="M. Silva",
            customer_tier="pro",
            subject="Unknown login from new country",
            body="I got a sign-in alert from a country I've never visited.",
            sentiment="angry",
            issue_type="security",
        ),
        Ticket(
            ticket_id="T-MED-3",
            customer_name="Nora L.",
            customer_tier="free",
            subject="Please add dark mode",
            body="It would be great if app had dark mode scheduling.",
            sentiment="calm",
            issue_type="feature_request",
        ),
    ]

    hard_tickets = [
        Ticket(
            ticket_id="T-HARD-1",
            customer_name="Northwind Health",
            customer_tier="enterprise",
            subject="EHR sync outage in EU region",
            body="No records synced for 42 minutes. Clinical operations are degraded.",
            sentiment="angry",
            issue_type="outage",
        ),
        Ticket(
            ticket_id="T-HARD-2",
            customer_name="IT Admin - OneArc",
            customer_tier="enterprise",
            subject="Possible account takeover",
            body="Admin account password changed unexpectedly and API tokens rotated.",
            sentiment="angry",
            issue_type="security",
        ),
        Ticket(
            ticket_id="T-HARD-3",
            customer_name="Small Studio Co",
            customer_tier="pro",
            subject="Refund for accidental annual upgrade",
            body="Upgraded by mistake. Requesting refund under policy window.",
            sentiment="frustrated",
            issue_type="refund",
        ),
        Ticket(
            ticket_id="T-HARD-4",
            customer_name="Priya K.",
            customer_tier="pro",
            subject="Data export bug missing rows",
            body="CSV export omits some rows intermittently. Need workaround and ETA.",
            sentiment="frustrated",
            issue_type="bug",
        ),
    ]

    return {
        "easy": TaskDefinition(
            task_id="easy",
            name="Single Critical Outage Triage",
            difficulty="easy",
            objective="Identify the critical outage ticket, route correctly, and send a safe first response.",
            instruction=(
                "Prioritize and triage the urgent outage for enterprise customer. Avoid resolving before communicating."
            ),
            max_steps=8,
            initial_tickets=easy_tickets,
            grader=grade_easy,
        ),
        "medium": TaskDefinition(
            task_id="medium",
            name="Mixed Queue: Billing + Security + Feature",
            difficulty="medium",
            objective=(
                "Handle a mixed queue with proper routing, severity assignment, and customer-safe response drafts."
            ),
            instruction=(
                "Investigate all three tickets. Security event must be escalated quickly; feature request should remain low priority."
            ),
            max_steps=10,
            initial_tickets=medium_tickets,
            grader=grade_medium,
        ),
        "hard": TaskDefinition(
            task_id="hard",
            name="SLA-Critical Multi-Incident Queue",
            difficulty="hard",
            objective=(
                "Balance outage, security, refund, and bug tickets under SLA pressure while avoiding unsafe actions."
            ),
            instruction=(
                "Resolve nothing prematurely. Prioritize critical incidents, route tickets to correct teams, and provide precise draft replies."
            ),
            max_steps=12,
            initial_tickets=hard_tickets,
            grader=grade_hard,
        ),
    }
