---
title: OpenEnv Support Triage
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Customer Support Triage OpenEnv

A real-world OpenEnv environment that simulates **customer support inbox triage** for SaaS operations teams.

Agents must decide priority, routing, escalation, and customer-facing draft responses under realistic SLA pressure. This environment is designed to evaluate practical agent reliability, not game-playing.

## Why this environment

Support triage is a core operational workflow in real companies. A capable agent should:

- identify urgent incidents quickly,
- route tickets to the right teams,
- avoid unsafe resolution behavior,
- communicate clearly to customers.

This environment captures those constraints with deterministic task graders and trajectory-shaped rewards.

## OpenEnv interface

The environment implements:

- `reset(task_id)` → returns typed `Observation`
- `step(action)` → returns typed `observation`, `reward`, `done`, `info`
- `state()` → returns internal environment state snapshot

Typed models are in `app/models.py`:

- `Observation`
- `Action`
- `Reward`

Metadata is provided in `openenv.yaml`.

## Action space

The action schema is validated by Pydantic (`Action`) and available from `/tasks`.

`action_type` options:

- `set_priority` (`ticket_id`, `priority`)
- `assign_team` (`ticket_id`, `team`)
- `add_label` (`ticket_id`, `label`)
- `draft_reply` (`ticket_id`, `message`)
- `resolve_ticket` (`ticket_id`)
- `escalate_ticket` (`ticket_id`)
- `noop`

## Observation space

Each observation includes:

- task metadata (`task_id`, `task_name`, `instruction`)
- episode progress (`step_count`, `max_steps`)
- current queue of typed `Ticket` objects
- recent environment events

## Tasks (easy → medium → hard)

1. **easy** — Single critical outage triage
   - Objective: Prioritize outage, assign engineering, communicate safely.
2. **medium** — Mixed queue (billing + security + feature request)
   - Objective: Handle routing and severity tradeoffs across three ticket types.
3. **hard** — SLA-critical multi-incident queue
   - Objective: Balance outage/security urgency with refund + bug workflows while avoiding unsafe behavior.

All tasks have deterministic graders returning scores in `[0.0, 1.0]`.

## Reward design

Reward is trajectory-aware and meaningful:

- positive signal from grader score improvement (`progress_delta`)
- small per-step cost to discourage inefficient policies
- penalties for invalid actions
- penalties for repetitive loops
- safety penalty for resolving tickets without customer communication

This gives dense feedback instead of sparse terminal-only scoring.

## API endpoints

- `POST /reset` — reset task (`{"task_id":"easy|medium|hard"}`)
- `POST /step` — apply one action
- `GET /state` — full current environment state
- `GET /tasks` — available tasks + action schema
- `GET /grader` — current task grader score
- `POST /baseline` — run baseline policy over all 3 tasks

## Local setup

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t support-triage-openenv .
docker run -p 7860:7860 support-triage-openenv
```

### Docker API smoke test (curl)

After the container is running:

```bash
curl -s http://localhost:7860/tasks
curl -s -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{\"task_id\":\"easy\"}"
curl -s -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"action_type\":\"set_priority\",\"ticket_id\":\"T-EASY-1\",\"priority\":\"critical\"}"
curl -s http://localhost:7860/grader
curl -s -X POST http://localhost:7860/baseline -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o-mini\",\"max_steps\":12}"
```

## Baseline inference script

`baseline.py` runs all three tasks and prints reproducible scores.

- Uses OpenAI Python client when `OPENAI_API_KEY` is set (`temperature=0`).
- Falls back to deterministic heuristic policy when API key is absent.

Run:

```bash
python baseline.py --model gpt-4o-mini
```

## Hugging Face Spaces deployment

Create a new **Docker** Space and push this repo. Ensure Space metadata includes tag `openenv`.

Container starts on port `7860` and serves API from `app.server:app`.

## Validation checklist

Before submission:

1. `openenv validate` (if installed in your environment)
2. `docker build` and `docker run` succeed
3. `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline` respond correctly
4. baseline script completes and returns scores for easy/medium/hard

## Baseline reference scores

Run locally and record output from `python baseline.py`:

- easy: `0.85`
- medium: `0.55`
- hard: `0.55`
- average: `0.65`
