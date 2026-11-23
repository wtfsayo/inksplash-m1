"""A minimal OpenAI chat-completions server for Inkling 2 Mini on the Splash engine.

The Splash server has no Inkling chat format (its parser reads <think>/<tool_call>
tags; Inkling writes role/content tokens), so this test server renders prompts with
the model's own chat template, drives `splash serve-native` through the server's
native-protocol client (server.runtime.MultiplexedRuntime) and parses Inkling's
messages:

  <|message_model|><|content_thinking|>...<|end_message|>        reasoning
  <|message_model|><|content_text|>...<|end_message|>            answer text
  <|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>
  <|content_model_end_sampling|>                                 end of turn

Every request is logged as one JSON line (prompt and generated ids, cached tokens,
prefill and decode times, draft acceptance) to --log. One request runs at a time.
A body may carry `prompt_token_ids` (a prompt rendered elsewhere, used as is).

  serve.py --model MODEL_DIR --log FILE [--port 8011] [--context 65536]

Run from the repository root after `make all`, with the server's requirements:
  uv run --with-requirements install/requirements.txt --with transformers \
      python bench/inkling/serve.py --model MODEL_DIR --log /tmp/inkling.jsonl
"""

import argparse
import json
import math
import pathlib
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MESSAGE_MODEL = 200001
CONTENT_TEXT = 200004
END_SAMPLING = 200006
CONTENT_THINKING = 200008
END_MESSAGE = 200010
INVOKE_JSON = 200049
INVOKE_TEXT = 200057
EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def parse_args():
    p = argparse.ArgumentParser()
    repo = pathlib.Path(__file__).resolve().parents[2]
    p.add_argument("--splash-root", default=str(repo), help="directory holding the server package")
    p.add_argument("--binary", default=str(repo / "build" / "splash"))
    p.add_argument("--model", required=True)
    p.add_argument("--served-name", default="inkling-2-mini")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8011)
    p.add_argument("--context", default="65536")
    p.add_argument("--memory", default="auto")
    p.add_argument("--default-effort", default="medium")
    p.add_argument("--log", required=True)
    p.add_argument("--engine-arg", action="append", default=[], help="extra serve-native argument (repeatable)")
    return p.parse_args()


ARGS = parse_args()
sys.path.insert(0, ARGS.splash_root)
from server import protocol as wire  # noqa: E402
from server import runtime as engine  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = AutoTokenizer.from_pretrained(f"{ARGS.model}/tokenizer")
RUNTIME = engine.MultiplexedRuntime(
    [ARGS.binary, "serve-native", ARGS.model, ARGS.context, ARGS.memory,
     "--persistent-cache", "off", *ARGS.engine_arg],
    pending_limit=4,
)
LOG_LOCK = threading.Lock()
GPU_LOCK = threading.Lock()  # one request at a time: the test measures single-request decode


def log(record):
    with LOG_LOCK, open(ARGS.log, "a") as f:
        f.write(json.dumps(record) + "\n")


def canonical_messages(messages):
    """OpenAI messages as Inkling's template takes them: text content, tool-call
    arguments as objects, reasoning as reasoning_content."""
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in (None, "text", "input_text")
            )
        if m.get("role") == "developer":
            m["role"] = "system"
        for key in ("reasoning", "reasoning_text"):
            if m.get(key) and not m.get("reasoning_content"):
                m["reasoning_content"] = m.pop(key)
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call = json.loads(json.dumps(call))
                args = call.get("function", {}).get("arguments")
                if isinstance(args, str):
                    try:
                        call["function"]["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        call["function"]["arguments"] = {"_raw": args}
                calls.append(call)
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


class Projector:
    """Inkling's generated tokens as reasoning, text and tool calls, incrementally."""

    def __init__(self):
        self.mode = None  # None: between messages; "name", "thinking", "text", "tool_json", "tool_text"
        self.name_ids, self.body_ids = [], []
        self.sent = 0  # characters of the current body already emitted
        self.calls = []
        self.reasoning_parts, self.text_parts = [], []
        self.raw = []

    def _flush(self, final):
        text = TOKENIZER.decode(self.body_ids, skip_special_tokens=False)
        if not final and text.endswith("\ufffd"):
            return None
        delta, self.sent = text[self.sent:], len(text)
        return delta

    def feed(self, token):
        """Events for one token: ('reasoning'|'text', delta) or ('call', {...})."""
        self.raw.append(token)
        events = []
        if token == MESSAGE_MODEL:
            events += self._close()
            self.mode, self.name_ids = "name", []
        elif token in (CONTENT_THINKING, CONTENT_TEXT, INVOKE_JSON, INVOKE_TEXT) and self.mode in ("name", None):
            self.mode = {CONTENT_THINKING: "thinking", CONTENT_TEXT: "text",
                         INVOKE_JSON: "tool_json", INVOKE_TEXT: "tool_text"}[token]
            self.body_ids, self.sent = [], 0
        elif token in (END_MESSAGE, END_SAMPLING):
            events += self._close()
        elif self.mode == "name":
            self.name_ids.append(token)
        elif self.mode in ("thinking", "text"):
            self.body_ids.append(token)
            delta = self._flush(False)
            if delta:
                kind = "reasoning" if self.mode == "thinking" else "text"
                (self.reasoning_parts if kind == "reasoning" else self.text_parts).append(delta)
                events.append((kind, delta))
        elif self.mode in ("tool_json", "tool_text"):
            self.body_ids.append(token)
        return events

    def _close(self):
        events = []
        if self.mode in ("thinking", "text"):
            delta = self._flush(True)
            if delta:
                kind = "reasoning" if self.mode == "thinking" else "text"
                (self.reasoning_parts if kind == "reasoning" else self.text_parts).append(delta)
                events.append((kind, delta))
        elif self.mode in ("tool_json", "tool_text"):
            name = TOKENIZER.decode(self.name_ids).strip()
            body = TOKENIZER.decode(self.body_ids, skip_special_tokens=False)
            if self.mode == "tool_json":
                try:
                    parsed = json.loads(body)
                    name = parsed.get("name", name)
                    args = parsed.get("args", {})
                except json.JSONDecodeError:
                    args = {"_unparsed": body}
            else:
                args = {"text": body}
            call = {"id": "call_" + uuid.uuid4().hex[:12], "name": name, "arguments": json.dumps(args)}
            self.calls.append(call)
            events.append(("call", call))
        self.mode, self.name_ids, self.body_ids, self.sent = None, [], [], 0
        return events


def sampling_of(body):
    temperature = float(body.get("temperature", 0.0) or 0.0)
    return wire.SamplingParameters(
        temperature=temperature,
        top_p=float(body.get("top_p", 1.0) or 1.0),
        top_k=int(body.get("top_k", 0) or 0),
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
    )


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/status":
            event = RUNTIME.status(timeout=10.0)
            data = bytes(event.json)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": ARGS.served_name, "object": "model", "owned_by": "splash"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        try:
            self._complete(body)
        except Exception as error:  # report, keep serving
            log({"error": repr(error), "at": time.time()})
            try:
                self._json(500, {"error": {"message": repr(error)}})
            except Exception:
                pass

    def _complete(self, body):
        received = time.time()
        effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort") or ARGS.default_effort
        if effort not in EFFORTS:
            effort = ARGS.default_effort
        messages = canonical_messages(body.get("messages", []))
        tools = body.get("tools") or None
        if body.get("prompt_token_ids"):
            # A prompt rendered elsewhere (the profiler's token files), as is.
            prompt = [int(t) for t in body["prompt_token_ids"]]
        else:
            rendered = TOKENIZER.apply_chat_template(messages, tools=tools, tokenize=True,
                                                     add_generation_prompt=False, reasoning_effort=effort)
            prompt = list(rendered["input_ids"] if hasattr(rendered, "keys") else rendered)
        max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16384)
        stream = bool(body.get("stream"))
        frame = wire.RequestFrame(
            request_id=0, priority=wire.RequestPriority.FOREGROUND, absolute_deadline_unix_micros=0,
            remaining_deadline_micros=0, logical_max_output_tokens=max_tokens, prompt_tokens=tuple(prompt),
            sampling=sampling_of(body), seed=int(body.get("seed", 0) or 0), constraint=wire.ConstraintMode.NONE,
            image_spans=(), image_pixels=b"", score_tokens=(), generation_prompt_tokens=0,
            flags=wire.RequestFlag(0), cache_salt=0)
        request = engine.GenerationRequest(frame, math.inf, None, wire.request_fingerprint(frame))
        projector = Projector()
        created, rid = int(received), "chatcmpl-" + uuid.uuid4().hex[:16]
        model = body.get("model", ARGS.served_name)
        events = []
        cond = threading.Condition()

        matched = []

        def on_event(_call, event):
            if isinstance(event, wire.StartEvent):
                matched.append(event.matched_prompt_tokens)
            if isinstance(event, wire.TokensEvent):
                with cond:
                    events.append(event.tokens)
                    cond.notify()

        def on_complete(_call):
            with cond:
                events.append(None)
                cond.notify()

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def send(delta, finish=None, usage=None):
            if not stream:
                return
            chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if usage is None else []}
            if usage is not None:
                chunk["usage"] = usage
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        with GPU_LOCK:
            started = time.time()
            send({"role": "assistant", "content": ""})
            call = RUNTIME.submit(request, on_event=on_event, on_complete=on_complete)
            first_token_at = None
            call_index = 0
            done = False
            while not done:
                with cond:
                    while not events:
                        cond.wait()
                    batch, events[:] = list(events), []
                for tokens in batch:
                    if tokens is None:
                        done = True
                        break
                    if first_token_at is None:
                        first_token_at = time.time()
                    for token in tokens:
                        for kind, value in projector.feed(token):
                            if kind == "reasoning":
                                send({"reasoning_content": value})
                            elif kind == "text":
                                send({"content": value})
                            else:
                                send({"tool_calls": [{"index": call_index, "id": value["id"], "type": "function",
                                                      "function": {"name": value["name"], "arguments": value["arguments"]}}]})
                                call_index += 1
            result = call.result()
        for kind, value in projector._close():
            if kind == "reasoning":
                send({"reasoning_content": value})
            elif kind == "text":
                send({"content": value})
            elif kind == "call":
                send({"tool_calls": [{"index": call_index, "id": value["id"], "type": "function",
                                      "function": {"name": value["name"], "arguments": value["arguments"]}}]})
                call_index += 1
        finish = "tool_calls" if projector.calls else ("length" if result.reason == wire.FinishReason.LENGTH else "stop")
        usage = {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                 "total_tokens": result.prompt_tokens + result.completion_tokens}
        decode_s = result.decode_micros / 1e6
        log({"at": started, "matched": matched[0] if matched else None, "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
             "prefill_ms": result.prefill_micros / 1e3, "decode_ms": result.decode_micros / 1e3,
             "wall_ms": result.wall_micros / 1e3, "decode_tps": result.completion_tokens / decode_s if decode_s else None,
             "drafted": result.drafted_tokens, "accepted": result.accepted_draft_tokens, "cycles": result.draft_cycles,
             "tokens_per_cycle": result.completion_tokens / result.draft_cycles if result.draft_cycles else None,
             "deduplicated": result.deduplicated_tokens, "finish": finish, "tool_calls": len(projector.calls),
             "reasoning_chars": sum(map(len, projector.reasoning_parts)), "text_chars": sum(map(len, projector.text_parts)),
             "effort": effort, "temperature": frame.sampling.temperature, "tools": len(tools or []),
             "ttft_ms": (first_token_at - started) * 1e3 if first_token_at else None,
             "raw_tail": TOKENIZER.decode(projector.raw[-40:], skip_special_tokens=False),
             "prompt_ids": prompt, "generated_ids": projector.raw})
        if stream:
            send({}, finish)
            if (body.get("stream_options") or {}).get("include_usage"):
                send(None, usage=usage)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        message = {"role": "assistant", "content": "".join(projector.text_parts) or None}
        if projector.reasoning_parts:
            message["reasoning_content"] = "".join(projector.reasoning_parts)
        if projector.calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for c in projector.calls]
        self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": model,
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage})


def main():
    print("starting engine...", flush=True)
    RUNTIME.wait_ready(timeout=1800)
    print(f"ready on {ARGS.host}:{ARGS.port}", flush=True)
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
