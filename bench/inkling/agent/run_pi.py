"""Runs the pi coding agent (https://github.com/badlogic/pi-mono, tested with 1.6.0) on the
eight checked tasks of tasks.py against a server at INKLING_URL (default serve.py's,
http://127.0.0.1:8011/v1), fully isolated: its own config directory holding that one
provider, its own HOME, no skills, MCP, extensions, context files, prompt templates or
themes, offline, one fresh sandbox a task. Each task's events go to OUT/<task>.jsonl and a
summary line to OUT/summary.jsonl; the server's log has each request's timings.

  python3 run_pi.py OUT [TASK...]
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks as agent_tasks  # noqa: E402

URL = os.environ.get("INKLING_URL", "http://127.0.0.1:8011/v1")
FLAGS = ["--no-skills", "--no-mcp", "--no-extensions", "--no-context-files", "--no-prompt-templates",
         "--no-themes", "--offline", "--no-approve", "--provider", "inkling", "--model", "inkling-2-mini",
         "--mode", "json", "-p"]
TIMEOUT = 20 * 60


def final_answer(lines):
    """The last assistant message's text in pi's JSON events."""
    answer = ""
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if isinstance(message, dict) and message.get("role") == "assistant":
            parts = message.get("content")
            if isinstance(parts, list):
                text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
                if text.strip():
                    answer = text
    return answer


def isolated_config(root):
    """pi's config directory with one provider, the server at URL, and nothing else."""
    agent = root / "agent"
    for sub in (agent, root / "home", root / "sessions"):
        sub.mkdir(parents=True, exist_ok=True)
    provider = {"baseUrl": URL, "api": "openai-completions", "apiKey": "none", "models": [{
        "id": "inkling-2-mini", "name": "Inkling 2 Mini (Splash)", "reasoning": True, "input": ["text"],
        "contextWindow": 65536, "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}
    (agent / "models.json").write_text(json.dumps({"providers": {"inkling": provider}}, indent=2))
    for name in ("settings.json", "auth.json"):
        (agent / name).write_text("{}")
    return root


def main():
    out = pathlib.Path(sys.argv[1]).resolve()
    names = set(sys.argv[2:])
    out.mkdir(parents=True, exist_ok=True)
    iso = isolated_config(out / "pi-config")
    pi = shutil.which("pi")
    for task in agent_tasks.TASKS:
        if names and task.name not in names:
            continue
        sandbox = out / "sandboxes" / task.name
        if sandbox.exists():
            shutil.rmtree(sandbox)
        sandbox.mkdir(parents=True)
        task.build(sandbox)
        env = {"PATH": os.environ["PATH"], "HOME": str(iso / "home"), "PI_CODING_AGENT_DIR": str(iso / "agent"),
               "PI_CODING_AGENT_SESSION_DIR": str(iso / "sessions"), "PI_OFFLINE": "1", "PI_TELEMETRY": "0",
               "TERM": "dumb", "LANG": "en_US.UTF-8"}
        started = time.time()
        try:
            result = subprocess.run([pi, *FLAGS, task.prompt], cwd=sandbox, env=env, capture_output=True, text=True,
                                    timeout=TIMEOUT)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr, code = "timeout", -1
        seconds = time.time() - started
        (out / f"{task.name}.jsonl").write_text(stdout)
        (out / f"{task.name}.stderr").write_text(stderr)
        answer = final_answer(stdout.splitlines())
        try:
            passed = bool(task.check(str(sandbox), answer))
        except Exception as error:  # a checker that crashes on the agent's files fails the task
            passed, answer = False, answer + f"\n[checker error: {error!r}]"
        record = {"task": task.name, "passed": passed, "seconds": round(seconds, 1), "exit": code,
                  "answer": answer[-400:]}
        with open(out / "summary.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{task.name:28s} {'PASS' if passed else 'FAIL'} {seconds:7.1f}s exit {code}", flush=True)


if __name__ == "__main__":
    main()
