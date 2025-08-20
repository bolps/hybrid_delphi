#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Iterable

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


def load_narratives(path: str) -> List[Dict]:
    """
    Loads narratives from JSON file and returns a list of narrative objects.
    Each narrative should have: narrative_id, text, source, lang, subreddit (optional)
    """
    with open(path, 'r', encoding='utf-8') as f:
        narratives = json.load(f)
    
    # Validate that each narrative has required fields
    required_fields = ['narrative_id', 'text']
    for i, narrative in enumerate(narratives):
        missing_fields = [field for field in required_fields if field not in narrative]
        if missing_fields:
            raise ValueError(f"Narrative {i} missing required fields: {missing_fields}")
    
    return narratives


def get_narrative_by_id(narratives: List[Dict], narrative_id: str) -> str:
    """
    Retrieves the text of a specific narrative by its ID.
    """
    for narrative in narratives:
        if narrative['narrative_id'] == narrative_id:
            return narrative['text']
    
    raise ValueError(f"Narrative with ID '{narrative_id}' not found")


def _escape_non_placeholder_braces(template: str, placeholders: Iterable[str]) -> str:
    """
    Protects the known placeholders like {item_id} by swapping them to sentinel tokens,
    then escapes ALL other braces in the template (turns { -> {{ and } -> }}),
    and finally restores the protected placeholders.

    This lets you keep JSON in your template (with lots of braces) while still using
    str.format() only for the fields we explicitly support.
    """
    # 1) Protect placeholders
    protected = {}
    protected_template = template
    for name in placeholders:
        literal = "{" + name + "}"
        token = f"__PLACEHOLDER_{name.upper()}__"
        protected[name] = token
        protected_template = protected_template.replace(literal, token)

    # 2) Escape all remaining braces
    protected_template = protected_template.replace("{", "{{").replace("}", "}}")

    # 3) Restore placeholders
    for name, token in protected.items():
        protected_template = protected_template.replace(token, "{" + name + "}")

    return protected_template


def make_prompts(narrative: str, narrative_id: str, template_path: str, questions_df: pd.DataFrame):
    """
    Formats one prompt per row in questions_df using the template file.
    The template can safely contain JSON with braces. Use these placeholders:
      {item_id}, {item_text}, {dimension_text}, {narrative}, {narrative_id}
    """
    template = Path(template_path).read_text(encoding="utf-8")
    placeholders = ["item_id", "item_text", "dimension_text", "narrative", "narrative_id"]
    safe_template = _escape_non_placeholder_braces(template, placeholders)

    prompts = []
    for _, row in questions_df.iterrows():
        prompt = safe_template.format(
            item_id=row["item_id"],
            item_text=row["item_text"],
            dimension_text=row["dimension_text"],
            narrative=narrative,
            narrative_id=narrative_id,
        )
        prompts.append({
            "item_id": row["item_id"], 
            "prompt": prompt,
            "narrative_id": narrative_id
        })
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
    parser.add_argument("--narratives-json", required=True, help="Path to the narratives JSON file")

    # NOTE: --narrative-id is optional now. Required only when --process-all is NOT set.
    parser.add_argument("--narrative-id", required=False, help="ID of the specific narrative to use")

    parser.add_argument("--out", default="labeled_items.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-4o", help="OpenRouter model name (e.g. openai/gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens for completion")
    
    # Optional: process all narratives
    parser.add_argument("--process-all", action="store_true", help="Process all narratives in the JSON file")
    
    args = parser.parse_args()

    # Validate flags: --narrative-id is required unless --process-all is used.
    if not args.process_all and not args.narrative_id:
        parser.error("--narrative-id is required unless --process-all is provided")
    if args.process_all and args.narrative_id:
        print("[info] --process-all provided; --narrative-id will be ignored.")

    # Load inputs
    qdf = load_questions(args.questions_xlsx)
    narratives = load_narratives(args.narratives_json)

    # Determine which narratives to process
    if args.process_all:
        narrative_ids = [n['narrative_id'] for n in narratives]
        print(f"Processing all {len(narrative_ids)} narratives")
    else:
        narrative_ids = [args.narrative_id]
        print(f"Processing narrative: {args.narrative_id}")

    # Client
    client = make_openrouter_client()

    # Process each narrative
    for narrative_id in narrative_ids:
        try:
            narrative_text = get_narrative_by_id(narratives, narrative_id)
            prompts = make_prompts(narrative_text, narrative_id, args.prompt_template, qdf)

            # Determine output file name
            if args.process_all:
                out_file = Path(args.out).with_name(f"{narrative_id}_{Path(args.out).name}")
            else:
                out_file = Path(args.out)

            # Write outputs
            out_file.parent.mkdir(parents=True, exist_ok=True)
            wrote = 0

            with out_file.open("w", encoding="utf-8") as f:
                for p in prompts:
                    try:
                        content = call_llm_openrouter(
                            p["prompt"],
                            client=client,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        # Try to parse assistant output as JSON
                        try:
                            parsed = json.loads(content)
                        except Exception:
                            parsed = None

                        row = {
                            "item_id": p["item_id"],
                            "narrative_id": p["narrative_id"],
                            "model": args.model,
                            "raw_response": content,
                            "parsed": parsed,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
                        wrote += 1
                    except Exception as e:
                        err = {
                            "item_id": p["item_id"], 
                            "narrative_id": p["narrative_id"],
                            "error": str(e)
                        }
                        f.write(json.dumps(err, ensure_ascii=False) + "\\n")
                        print(f"[error] item {p['item_id']} (narrative {p['narrative_id']}): {e}")

            print(f"Wrote {wrote} rows for narrative {narrative_id} to {out_file.resolve()}")
            
        except Exception as e:
            print(f"[error] Failed to process narrative {narrative_id}: {e}")

    print("Processing complete!")


if __name__ == "__main__":
    main()
