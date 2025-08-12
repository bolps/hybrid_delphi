import json
import pandas as pd
from pathlib import Path

QUESTIONS_XLSX = "20250811_175831_Round3_Decisions.xlsx"
PROMPT_TEMPLATE = "reddit_label_prompt_template.txt"
POLICY_ID = "false_by_default_v1"

def load_questions(path=QUESTIONS_XLSX):
    df = pd.read_excel(Path("/mnt/data")/path, sheet_name="Sheet1")
    df = df.rename(columns={
        "Item ID": "item_id",
        "Item Text": "item_text",
        "Dim ID": "dimension_id",
        "Dimension Text": "dimension_text"
    })
    return df[["item_id","item_text","dimension_id","dimension_text"]]

def make_prompts(narrative: str):
    template_path = Path("/mnt/data")/PROMPT_TEMPLATE
    template = template_path.read_text(encoding="utf-8")
    Q = load_questions()
    prompts = []
    for _, row in Q.iterrows():
        prompt = template.format(
            item_id=row["item_id"],
            item_text=row["item_text"],
            dimension_text=row["dimension_text"],
            narrative=narrative
        )
        prompts.append({"item_id": row["item_id"], "policy": POLICY_ID, "prompt": prompt})
    return prompts

if __name__ == "__main__":
    sample_narrative = "Incolla qui il testo del post Reddit..."
    prompts = make_prompts(sample_narrative)
    out_path = Path("/mnt/data")/"per_item_prompts.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False)+"\n")
    print("Wrote", out_path, "with", len(prompts), "prompts under policy", POLICY_ID)
