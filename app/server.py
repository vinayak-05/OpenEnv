from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.environment import SupportTriageEnv
from app.models import Action, BaselineRequest, ResetRequest
from baseline import run_baseline

app = FastAPI(title="OpenEnv Customer Support Triage", version="1.0.0")
env = SupportTriageEnv(default_task_id="easy")


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "customer-support-triage", "task": env.current_task.task_id}


@app.post("/reset")
def reset(payload: ResetRequest) -> dict:
    try:
        observation = env.reset(task_id=payload.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"observation": observation.model_dump()}


@app.post("/step")
def step(action: Action) -> dict:
    result = env.step(action)
    return result.model_dump()


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/tasks")
def tasks() -> dict:
    schema = Action.model_json_schema()
    return {
        "tasks": [task.model_dump() for task in env.list_tasks()],
        "action_schema": schema,
    }


@app.get("/grader")
def grader() -> dict:
    return {
        "task_id": env.current_task.task_id,
        "score": env.grade(),
        "done": env.done,
        "grader": env.current_task.grader.__name__,
    }


@app.post("/baseline")
def baseline(payload: BaselineRequest) -> dict:
    try:
        scores = run_baseline(model=payload.model, max_steps_override=payload.max_steps)
        return {"model": payload.model, "scores": scores, "fallback_used": False}
    except Exception as exc:
        scores = run_baseline(model="gpt-4o-mini", max_steps_override=payload.max_steps)
        return {
            "model": payload.model,
            "scores": scores,
            "fallback_used": True,
            "message": f"Baseline fallback used due to error: {type(exc).__name__}",
        }
