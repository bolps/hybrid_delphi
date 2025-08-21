#!/usr/bin/env python3
"""
Batch inference of narratives via OpenRouter.

- Keeps the labeling prompt EXACTLY as-is from labeling_prompt.yaml
- Only replaces the literal token: PASTE THE NARRATIVE HERE
- Reads narratives from a JSON array with fields: narrative_id, text
- Writes results to JSONL: one line per narrative with {narrative_id, model, output_json, raw_response}

Usage:
  python batch_infer_narratives.py \
    --input /mnt/data/narratives_small.json \
    --prompt /mnt/data/labeling_prompt.yaml \
    --output /mnt/data/inference_results.jsonl \
    --model anthropic/claude-3.5-sonnet

Requires:
  pip install requests python-dotenv
  .env file containing: OPENROUTER_API_KEY=sk-or-...
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # per docs

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_narratives(narratives_path: str) -> List[Dict[str, Any]]:
    with open(narratives_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of {narrative_id, text} objects.")
    return data

def prepare_prompt(prompt_template: str, narrative_text: str) -> str:
    # Preserve the prompt EXACTLY, only swap the placeholder text.
    safe_text = narrative_text.replace('"""', '\\"""')  # avoid breaking the triple-quoted block
    return prompt_template.replace("PASTE THE NARRATIVE HERE", safe_text)

def call_openrouter(api_key: str, model: str, user_prompt: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    Sends a single chat.completions request to OpenRouter.

    Returns the parsed JSON response (dict). Raises on fatal errors.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",  # per OpenRouter auth docs
        "Content-Type": "application/json",
        # Optional app attribution headers; safe to omit
        # "HTTP-Referer": "https://your-app.example",
        # "X-Title": "Batch Narrative Labeler",
    }

    payload = {
        "model": model,  # single model; you could also supply "models": [...] to enable routing/fallbacks
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        # You can tweak sampling if needed:
        # "temperature": 0.0
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        # Handle rate limits or transient 5xx with exponential backoff
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        # Other errors: raise with context
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

def extract_text_choice(api_json: Dict[str, Any]) -> str:
    """
    Pulls the assistant message text from the OpenRouter chat completion response.
    """
    choices = api_json.get("choices", [])
    if not choices:
        raise ValueError(f"Unexpected response format (no choices): {api_json}")
    msg = choices[0].get("message", {})
    content = msg.get("content")
    if not content:
        raise ValueError(f"Unexpected response format (no message.content): {api_json}")
    return content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to narratives JSON array")
    parser.add_argument("--prompt", required=True, help="Path to labeling_prompt.yaml")
    parser.add_argument("--output", required=True, help="Path to write JSONL results")
    parser.add_argument("--model", required=True, help="OpenRouter model id, e.g., anthropic/claude-3.5-sonnet")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay between requests (seconds)")
    args = parser.parse_args()

    # Load .env and get API key
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY in environment (e.g., set it in your .env).", file=sys.stderr)
        sys.exit(1)

    prompt_template = load_prompt(args.prompt)
    narratives = load_narratives(args.input)

    # Open output file for JSONL
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(narratives, start=1):
            narrative_id = item.get("narrative_id", f"row_{i}")
            text = item.get("text", "")
            if not text:
                print(f"[WARN] narrative_id={narrative_id} has empty text; skipping.", file=sys.stderr)
                continue

            user_prompt = prepare_prompt(prompt_template, text)
            try:
                api_json = call_openrouter(api_key=api_key, model=args.model, user_prompt=user_prompt)
                assistant_text = extract_text_choice(api_json)

                # Try to parse the model's JSON (the prompt instructs it to return JSON only).
                parsed: Any
                try:
                    parsed = json.loads(assistant_text)
                except json.JSONDecodeError:
                    # If the model returns extra whitespace or markers, attempt a light trim; else keep raw.
                    trimmed = assistant_text.strip()
                    try:
                        parsed = json.loads(trimmed)
                    except Exception:
                        parsed = None

                record = {
                    "narrative_id": narrative_id,
                    "model": api_json.get("model"),
                    "output_json": parsed,            # parsed dict if JSON was valid, else None
                    "raw_response": assistant_text,   # always keep the raw text for debugging
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            except Exception as e:
                err_rec = {
                    "narrative_id": narrative_id,
                    "error": str(e),
                }
                out_f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                print(f"[ERROR] {narrative_id}: {e}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done. Wrote results to: {args.output}")

if __name__ == "__main__":
    main()
