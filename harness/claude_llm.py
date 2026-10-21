"""Claude as LLaMEA's LLM (the llamea package has no Claude class).

    from harness.claude_llm import Claude_LLM
    llm = Claude_LLM(model="claude-sonnet-5-5")          # key from $ANTHROPIC_API_KEY

Every call appends one line to `usage_log` (JSONL: model, input/output/cache tokens, stop
reason), so the cost of a run can be computed afterwards. LLaMEA evaluates candidates in
worker processes, so the client is rebuilt after pickling (same pattern as llamea's OpenAI_LLM).
"""
from __future__ import annotations

import json
import os
import time

from llamea.llm import LLM


class Claude_LLM(LLM):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5-5",
                 temperature: float | None = None, max_tokens: int = 8192,
                 usage_log: str | None = None, **kwargs):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("no API key: pass api_key or set ANTHROPIC_API_KEY")
        super().__init__(api_key, model, None, **kwargs)
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        self.usage_log = os.path.abspath(usage_log) if usage_log else None
        self.client = self._make_client()

    def _make_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self.api_key, max_retries=8)

    @staticmethod
    def _split(session_messages):
        """Anthropic takes the system prompt separately and needs alternating user/assistant
        turns; consecutive messages of the same role are merged."""
        system, msgs = [], []
        for m in session_messages:
            role, content = m["role"], m["content"]
            if role == "system":
                system.append(content)
                continue
            role = "assistant" if role == "assistant" else "user"
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n\n" + content
            else:
                msgs.append({"role": role, "content": content})
        if not msgs or msgs[0]["role"] != "user":
            msgs.insert(0, {"role": "user", "content": "(start)"})
        return "\n\n".join(system), msgs

    def query(self, session_messages, max_retries: int = 5, default_delay: int = 10):
        system, msgs = self._split(session_messages)
        kw = dict(model=self.model, max_tokens=self.max_tokens, messages=msgs)
        if system:
            kw["system"] = system
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        attempt = 0
        while True:
            try:
                resp = self.client.messages.create(**kw)
                break
            except Exception:  # the SDK already retries rate limits / overload
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(default_delay * attempt)
        self._log_usage(resp)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _log_usage(self, resp):
        if not self.usage_log:
            return
        u = resp.usage
        rec = {"t": time.time(), "model": self.model, "stop_reason": resp.stop_reason,
               "input_tokens": getattr(u, "input_tokens", 0), "output_tokens": getattr(u, "output_tokens", 0),
               "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
               "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        os.makedirs(os.path.dirname(self.usage_log), exist_ok=True)
        with open(self.usage_log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ---- pickling: the client is not picklable, rebuild it ----
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.client = self._make_client()

    def __deepcopy__(self, memo):
        import copy
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k != "client":
                setattr(new, k, copy.deepcopy(v, memo))
        new.client = new._make_client()
        return new


# USD per million tokens (input, output), platform.claude.com/docs/en/about-claude/pricing,
# checked 2026-10-20. Cache reads 0.05x input (Opus/Sonnet 5.5), writes 1.25x (5 min).
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def cost_of(usage_log: str) -> dict:
    """Total tokens and USD of a usage log."""
    tot = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    if not os.path.exists(usage_log):
        return tot
    with open(usage_log) as fh:
        for line in fh:
            r = json.loads(line)
            pin, pout = PRICES.get(r["model"], (0.0, 0.0))
            tot["calls"] += 1
            tot["input_tokens"] += r["input_tokens"]
            tot["output_tokens"] += r["output_tokens"]
            tot["usd"] += (r["input_tokens"] * pin + r["output_tokens"] * pout
                           + r["cache_read_input_tokens"] * 0.05 * pin
                           + r["cache_creation_input_tokens"] * 1.25 * pin) / 1e6
    return tot
