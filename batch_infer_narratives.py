#!/usr/bin/env python3
"""
Batch inference of narratives via OpenRouter using the OpenAI Python SDK.

- Keeps the labeling prompt EXACTLY as-is from labeling_prompt.yaml
- Only replaces the literal token: PASTE THE NARRATIVE HERE
- Reads narratives from a JSON array with fields: narrative_id, text
- Writes results to JSONL: one line per narrative with
    {
      "narrative_id": ...,
      "model": ...,
      "output_json": {...},       # parsed JSON object from the model (best effort)
      "raw_response": "..."       # full text of the assistant response
    }

Example:
  python batch_infer_narratives.py \
    --input /mnt/data/narratives_small.json \
    --prompt /mnt/data/labeling_prompt.yaml \
    --output /mnt/data/inference_results.jsonl \
    --model anthropic/claude-3.5-sonnet \
    --env .env

Notes:
- Uses the OpenAI Python library configured to talk to OpenRouter by
  setting base_url="https://openrouter.ai/api/v1" and reading OPENROUTER_API_KEY
  from the environment (loaded via python-dotenv).
- Adds informative print() statements so progress is visible during execution.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    print("[FATAL] The 'openai' library is required. Install with: pip install openai", file=sys.stderr)
    raise

# dotenv (optional, but recommended)
_HAS_DOTENV = True
try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    _HAS_DOTENV = False

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Args:
    input: str
    prompt: str
    output: str
    model: str
    temperature: float = 0.0
    sleep: float = 0.0
    max_retries: int = 5
    timeout: float = 120.0
    verbose: bool = True
    env_path: Optional[str] = None


def parse_cli() -> Args:
    ap = argparse.ArgumentParser(description="Batch narrative inference via OpenRouter (OpenAI SDK).")
    ap.add_argument("--input", required=True, help="Path to JSON file (array of {narrative_id, text}).")
    ap.add_argument("--prompt", required=True, help="Path to YAML or text prompt with token 'PASTE THE NARRATIVE HERE'.")
    ap.add_argument("--output", required=True, help="Path to write JSONL results.")
    ap.add_argument("--model", required=True, help="Model id, e.g., 'anthropic/claude-3.5-sonnet'.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    ap.add_argument("--max-retries", type=int, default=5, help="Max retries per item on transient errors.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds).")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output.")
    ap.add_argument("--env", dest="env_path", default=None, help="Path to a .env file (optional).")
    ns = ap.parse_args()

    return Args(
        input=ns.input,
        prompt=ns.prompt,
        output=ns.output,
        model=ns.model,
        temperature=ns.temperature,
        sleep=ns.sleep,
        max_retries=ns.max_retries,
        timeout=ns.timeout,
        verbose=not ns.quiet,
        env_path=ns.env_path,
    )


def load_env(env_path: Optional[str]) -> None:
    """Load environment variables from .env if python-dotenv is available."""
    if not _HAS_DOTENV:
        print("[warn] 'python-dotenv' not installed; skipping .env load. Install with: pip install python-dotenv")
        return

    if env_path:
        print(f"[info] Loading environment from: {env_path}")
        ok = load_dotenv(dotenv_path=env_path, override=False)
        if not ok:
            print(f"[warn] Could not load .env at: {env_path} (continuing with existing environment)")
    else:
        # Auto-discover .env upward from CWD
        discovered = find_dotenv(usecwd=True)
        if discovered:
            print(f"[info] Loading environment from discovered .env: {discovered}")
            load_dotenv(dotenv_path=discovered, override=False)
        else:
            print("[info] No .env file found via auto-discovery; using existing environment")


def load_prompt(prompt_path: str) -> str:
    print(f"[info] Loading prompt from: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_inputs(input_path: str) -> List[Dict[str, Any]]:
    print(f"[info] Reading inputs from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects with fields: narrative_id, text")
    for i, rec in enumerate(data):
        if "narrative_id" not in rec or "text" not in rec:
            raise ValueError(f"Item at index {i} missing 'narrative_id' or 'text'")
    print(f"[info] Loaded {len(data)} narratives")
    return data


def mk_client(timeout: float) -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[FATAL] OPENROUTER_API_KEY is not set. Add it to your .env or export it in the environment.",
              file=sys.stderr)
        sys.exit(2)

    masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
    print(f"[info] Using OpenRouter key: {masked}")

    # Configure OpenAI SDK to talk to OpenRouter
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def build_messages(prompt_template: str, narrative_text: str) -> List[Dict[str, str]]:
    if "PASTE THE NARRATIVE HERE" not in prompt_template:
        print("[warn] Prompt missing 'PASTE THE NARRATIVE HERE'. Appending text instead.")
        combined = prompt_template + "\n\n" + narrative_text
    else:
        combined = prompt_template.replace("PASTE THE NARRATIVE HERE", narrative_text)
    return [{"role": "user", "content": combined}]


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except Exception:
            return None
    return None


def call_model(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float,
               max_retries: int = 5, verbose: bool = True) -> str:
    attempt, backoff = 0, 2.0
    while True:
        attempt += 1
        if verbose:
            print(f"[info] → Sending request (attempt {attempt}) to model: {model}")
        try:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            if verbose:
                print("[info] ✓ Received response")
            return resp.choices[0].message.content if resp and resp.choices else ""
        except Exception as e:
            retryable = any(s in str(e).lower() for s in [
                "rate limit", "timeout", "temporarily", "overloaded", "429", "502", "503", "504"
            ])
            if attempt < max_retries and retryable:
                print(f"[warn] Error: {e}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 30.0)
            else:
                print(f"[error] Request failed: {e}")
                raise


def process(args: Args) -> None:
    # 1) Load environment from .env (if present)
    load_env(args.env_path)

    # 2) Load prompt + inputs
    prompt_template = load_prompt(args.prompt)
    inputs = load_inputs(args.input)

    # 3) Make client
    client = mk_client(args.timeout)

    # 4) Iterate and write results
    print(f"[info] Writing results to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(inputs, start=1):
            narrative_id, text = rec.get("narrative_id"), rec.get("text", "")
            print(f"\n[progress] ({idx}/{len(inputs)}) Processing narrative_id={narrative_id}")
            messages = build_messages(prompt_template, text)
            try:
                raw = call_model(client, args.model, messages, args.temperature, args.max_retries, args.verbose)
                parsed = extract_json_from_text(raw)
                out_f.write(json.dumps({
                    "narrative_id": narrative_id,
                    "model": args.model,
                    "output_json": parsed,
                    "raw_response": raw,
                }, ensure_ascii=False) + "\n")
                print("[ok] JSON parsed" if parsed else "[warn] No JSON parsed; raw saved")
            except Exception as e:
                out_f.write(json.dumps({"narrative_id": narrative_id, "error": str(e)}, ensure_ascii=False) + "\n")
                print(f"[ERROR] {narrative_id}: {e}", file=sys.stderr)

            if args.sleep > 0:
                print(f"[info] Sleeping {args.sleep:.2f}s...")
                time.sleep(args.sleep)

    print(f"\n[done] Wrote results to: {args.output}")


def main():
    args = parse_cli()
    process(args)


if __name__ == "__main__":
    main()
