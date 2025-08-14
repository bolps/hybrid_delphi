#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def load_questions(path: str) -> pd.DataFrame:
    """
    Reads the questions Excel and returns a normalized dataframe with:
    item_id, item_text, dimension_id, dimension_text
    """
    df = pd.read_excel(Path(path), sheet_name=0)
    # Try to normalize common column names:
    rename_map = {
        "Item ID": "item_id",
        "ItemId": "item_id",
        "ItemID": "item_id",
        "item_id": "item_id",
        "Item Text": "item_text",
        "ItemText": "item_text",
        "item_text": "item_text",
        "Dim ID": "dimension_id",
        "DimID": "dimension_id",
        "dimension_id": "dimension_id",
        "Dimension Text": "dimension_text",
        "DimensionText": "dimension_text",
        "dimension_text": "dimension_text",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in ["item_id", "item_text", "dimension_id", "dimension_text"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in questions file: {missing}. "
            "Please include columns for item_id, item_text, dimension_id, dimension_text."
        )
    return df[["item_id", "item_text", "dimension_id", "dimension_text"]]


def make_prompts(narrative: str, template_path: str, questions_df: pd.DataFrame):
    """
    Formats one prompt per row in questions_df using the template file.
    The template should contain string.format placeholders:
      {item_id}, {item_text}, {dimension_text}, {narrative}
    """
    template = Path(template_path).read_text(encoding="utf-8")
    prompts = []
    for _, row in questions_df.iterrows():
        prompt = template.format(
            item_id=row["item_id"],
            item_text=row["item_text"],
            dimension_text=row["dimension_text"],
            narrative=narrative,
        )
        prompts.append({"item_id": row["item_id"], "prompt": prompt})
    return prompts


def make_openrouter_client() -> OpenAI:
    """
    Creates an OpenAI client configured for OpenRouter, reading only OPENROUTER_API_KEY from .env.
    """
    load_dotenv()  # loads .env from current working dir
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing. Put it in a .env file next to this script.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def call_llm_openrouter(prompt: str, client: OpenAI, model: str, temperature: float, max_tokens: int) -> str:
    """
    Calls the chat.completions endpoint with a single user message.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Label items using an LLM via OpenRouter + OpenAI SDK")
    parser.add_argument("--questions-xlsx", required=True, help="Path to the Excel file with items/questions")
    parser.add_argument("--prompt-template", default="narrative_prompt.txt", help="Path to the prompt template file")
    parser.add_argument("--narrative-file", required=True, help="Path to the narrative text file")
    parser.add_argument("--out", default="labeled_items.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-4o", help="OpenRouter model name (e.g. openai/gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for completion")
    args = parser.parse_args()

    # Load inputs
    qdf = load_questions(args.questions_xlsx)

    npath = Path(args.narrative_file)
    if not npath.exists():
        raise FileNotFoundError(f"Narrative file not found: {args.narrative_file}")
    narrative = npath.read_text(encoding="utf-8")

    prompts = make_prompts(narrative, args.prompt_template, qdf)

    # Client
    client = make_openrouter_client()

    # Write outputs
    out_fp = Path(args.out)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0

    with out_fp.open("w", encoding="utf-8") as f:
        for p in prompts:
            try:
                content = call_llm_openrouter(
                    p["prompt"],
                    client=client,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                # Try to parse assistant output as JSON (your template should make the model return one line JSON)
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None

                row = {
                    "item_id": p["item_id"],
                    "model": args.model,
                    "raw_response": content,
                    "parsed": parsed,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                wrote += 1
            except Exception as e:
                err = {"item_id": p["item_id"], "error": str(e)}
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"[error] item {p['item_id']}: {e}")

    print(f"Wrote {wrote} rows to {out_fp.resolve()}")


if __name__ == "__main__":
    main()
