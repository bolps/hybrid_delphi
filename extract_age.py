#!/usr/bin/env python3
"""
extract_age.py
---------------
Extracts the writer's age for each narrative using an LLM via OpenRouter
and writes one JSONL row per narrative:

{
  "narrative_id": "...",
  "age": 23,                  # integer or null if unknown
  "confidence": 0.0-1.0,      # model's self-reported confidence (float)
  "rationale": "short why",   # brief reasoning
  "model": "openai/gpt-4o",
  "raw_response": "..."       # original model text
}

Usage examples:
  python extract_age.py --narratives-json narratives.json --out age_by_narrative.jsonl
  python extract_age.py --narratives-json narratives.json --narrative-id N123 --out one_age.jsonl
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from openai import OpenAI


AGE_SYSTEM_PROMPT = """You are an information extraction assistant.
Given a personal narrative, infer the writer's AGE at the time of writing.

Rules:
- Return ONLY a compact JSON object with fields: age (integer or null), confidence (float 0-1), rationale (short string).
- age must be an INTEGER if explicitly stated (e.g., "I'm 21", "21F", "21 years old"). If unsure or only a broad range (e.g., "early 20s"), set age to null.
- confidence reflects how certain you are in the integer you output (0.0–1.0). If age is null, confidence should be <= 0.3.
- Do NOT infer from publication platform alone. Use explicit self-reports or unequivocal clues only.
- Keep rationale short (<= 20 words).

Return JSON only. No extra text.
"""

def make_openrouter_client() -> OpenAI:
    """Create an OpenAI SDK client configured for OpenRouter (reads OPENROUTER_API_KEY from .env)."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing. Put it in a .env file next to this script.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

def call_llm(prompt: str, client: OpenAI, model: str, temperature: float, max_tokens: int) -> str:
    """Single-shot chat.completions call with system + user messages."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def load_narratives(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # minimal validation
    for i, n in enumerate(data):
        if "narrative_id" not in n or "text" not in n:
            raise ValueError(f"Narrative {i} missing required fields (narrative_id, text)")
    return data

def get_narrative_by_id(narratives: List[Dict[str, Any]], nid: str) -> Dict[str, Any]:
    for n in narratives:
        if n["narrative_id"] == nid:
            return n
    raise ValueError(f"Narrative with ID '{nid}' not found")

def build_prompt(narrative: Dict[str, Any]) -> str:
    nid = narrative["narrative_id"]
    text = narrative["text"]
    return (
        f"narrative_id: {nid}\n"
        f"narrative_text:\n{text}"
    )

def parse_json_strict(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        # minimal schema guard
        if not isinstance(obj, dict):
            return None
        if "age" not in obj or "confidence" not in obj or "rationale" not in obj:
            return None
        # normalize age
        age = obj["age"]
        if age is not None:
            if isinstance(age, (int, float, str)):
                # if float or numeric string, coerce to int
                try:
                    age_int = int(str(age).strip())
                except Exception:
                    return None
                obj["age"] = age_int
            else:
                return None
        # normalize confidence
        try:
            obj["confidence"] = float(obj["confidence"])
        except Exception:
            return None
        # rationale short string
        if not isinstance(obj["rationale"], str):
            return None
        return obj
    except Exception:
        return None

# (Optional) tiny regex assist if the LLM fails JSON parsing:
AGE_REGEXES = [
    r"\b(?:I[' ]?m|I am|Im)\s*(\d{1,2})\b",
    r"\b(\d{1,2})\s*(?:yo|y/o|years old)\b",
    r"\b(\d{1,2})[MFmf]\b",  # e.g., "21F", "18M"
]

def quick_age_guess(text: str) -> Optional[int]:
    for pat in AGE_REGEXES:
        m = re.search(pat, text)
        if m:
            try:
                val = int(m.group(1))
                if 10 <= val <= 99:
                    return val
            except Exception:
                pass
    return None

def process_one(narrative: Dict[str, Any], client: OpenAI, model: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    prompt = build_prompt(narrative)
    raw = call_llm(prompt, client, model, temperature, max_tokens)

    parsed = parse_json_strict(raw)
    if parsed is None:
        # fallback: try regex to salvage an explicit age if present
        guess = quick_age_guess(narrative["text"])
        parsed = {
            "age": guess,
            "confidence": 0.25 if guess is None else 0.6,
            "rationale": "Fallback regex match" if guess is not None else "No clear age; fallback",
        }

    return {
        "narrative_id": narrative["narrative_id"],
        "age": parsed["age"],
        "confidence": parsed["confidence"],
        "rationale": parsed["rationale"],
        "model": model,
        "raw_response": raw,
    }

def main():
    ap = argparse.ArgumentParser(description="Extract writer age per narrative via LLM (OpenRouter).")
    ap.add_argument("--narratives-json", required=True, help="Path to the narratives JSON file")
    ap.add_argument("--out", default="age_by_narrative.jsonl", help="Output JSONL path")
    ap.add_argument("--narrative-id", help="If provided, only process this narrative_id")
    ap.add_argument("--model", default="openai/gpt-5-nano", help="OpenRouter model name (e.g., openai/gpt-5-nano)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    narratives = load_narratives(args.narratives_json)
    if args.narrative_id:
        narratives = [get_narrative_by_id(narratives, args.narrative_id)]

    client = make_openrouter_client()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    with out_path.open("w", encoding="utf-8") as f:
        for n in narratives:
            try:
                row = process_one(n, client, args.model, args.temperature, args.max_tokens)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception as e:
                err = {
                    "narrative_id": n.get("narrative_id", "<unknown>"),
                    "error": str(e),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"[error] narrative {n.get('narrative_id')}: {e}")

    print(f"Wrote {wrote} rows to {out_path.resolve()}")

if __name__ == "__main__":
    main()
