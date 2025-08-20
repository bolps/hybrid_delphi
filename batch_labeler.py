#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch questionnaire labeler using Ollama via the OpenAI SDK (OpenAI-compatible API).

Input JSON (list of objects):
[
  {"narrative_id": "r_0001", "text": "...", "source": "...", "lang": "en"},
  ...
]

Outputs:
  - out_dir/<narrative_id>.json      (full per-item JSON with XAI)
  - out_dir/results_combined.jsonl   (one JSON object per narrative)
  - out_dir/results_summary.csv      (flattened rows: narrative_id,item_id,label,xai)

Usage:
  pip install openai>=1.40.0
  # Make sure Ollama is running locally and exposes the OpenAI-compatible API.
  # Example: default Ollama works at http://localhost:11434; base_url ends with /v1
  python3 batch_labeler_openai.py --in narratives.json --model llama3.1:8b --outdir results_run
"""

import argparse, csv, json, sys, textwrap, time, uuid, os
from datetime import datetime
from pathlib import Path

# --- OpenAI SDK, pointed to Ollama's OpenAI-compatible endpoint ---
from openai import OpenAI

# Default config (you can also use env vars OPENAI_BASE_URL and OPENAI_API_KEY)
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")  # dummy key is fine for Ollama

ITEMS = [
    ("IT10","Parla male delle mie amici/amiche e dice che sono dei/delle “poco di buono”"),
    ("IT12","Dice che sono impegnata in troppe cose e che lo trascuro, vorrebbe che passassimo la maggior parte del tempo insieme"),
    ("IT22","Mi accusa di flirtare o di tradirlo"),
    ("IT24","Non mi sento libera di vestirmi come voglio"),
    ("IT25","Cerca di controllare ciò che faccio e con chi mi vedo"),
    ("IT26","Non vuole che io veda la mia famiglia, i miei amici/amiche"),
    ("IT15","Sento che mi manda troppi messaggi o mi chiama in modo insistente"),
    ("IT11","Mi chiede di non vedere i miei amici/amiche"),
    ("IT20","Pretende prove d'amore e dimostrazioni di fedeltà nei suoi confronti"),
    ("IT21","Diventa estremamente geloso o possessivo"),
    ("IT23","Mi sento controllata, ho la sensazione che non si fidi di me e che voglia sapere sempre dove sono e con chi, me lo chiede continuamente"),
    ("IT45","Controlla come spendo i miei soldi"),
    ("IT46","Mi impedisce di avere accesso al conto corrente o alle mie risorse economiche"),
    ("IT48","Mi impedisce cercare un lavoro per essere indipendente"),
    ("IT47","Mi viene chiesto di giustificare ogni acquisto che faccio"),
    ("IT09","I miei amici/amiche mi hanno fatto notare che non piace loro come si comporta con me"),
    ("IT19","Le mie amiche o la mia famiglia sono preoccupate per me"),
    ("IT01","Sento il suo sostegno nelle cose che faccio"),
    ("IT03","Mi sento ascoltata e si mostra interessato quando ho qualcosa che voglio condividere"),
    ("IT04","Capisce che ho bisogno anche di miei spazi personali e comprende che ci sono momenti in cui mi piace stare da sola o con persone diverse da lui (amici/amiche, familiari ecc.)"),
    ("IT06","Mi vuole bene per quella che sono e rispetta le mie scelte"),
    ("IT07","Sento che possiamo confrontarci e anche litigare senza farci male"),
    ("IT08","In sua compagnia mi sento a mio agio e al sicuro"),
    ("IT02","Mi incoraggia a provare nuove cose"),
    ("IT05","Mi fa sentire libera di fare le mie scelte su come gestire il mio tempo"),
    ("IT44","Sento di potergli parlare di qualsiasi cosa"),
    ("IT17","Provo paura vicino a lui, non mi sento al sicuro"),
    ("IT38","Mi afferra per i capelli, mi strattona, mi mette le mani al collo, tenta di soffocarmi, mi dà dei pugni, mi schiaffeggia, mi getta addosso degli oggetti o mi fa male in qualche modo"),
    ("IT40","Rompe e lancia oggetti per intimidirmi"),
    ("IT43","Ho paura di vederlo perché so che potrebbe avere comportamenti che mi possono fare stare male o sentire in imbarazzo"),
    ("IT14","Quando faccio qualcosa che a lui non va bene poi non mi parla per giorni"),
    ("IT16","Non mi fa sentire bene quando parla di come mi prendo cura di me stessa e del mio aspetto fisico"),
    ("IT28","In sua compagnia mi capita di sentirmi a disagio e ho la sensazione di dover stare attenta a ciò che dico o a ciò che faccio, come se dovessi “camminare sulle uova” per evitare sue reazioni negative"),
    ("IT29","Mi butta giù, mi insulta o mi critica"),
    ("IT30","Mi fa sentire spesso sbagliata e mi dice che è colpa mia se sta male"),
    ("IT39","Dopo che mi ha aggredito (fisicamente o verbalmente) nega di averlo fatto e mi fa dubitare che sia realmente successo."),
    ("IT41","Urla o mi umilia di fronte ad altre persone"),
    ("IT27","Ha forti oscillazioni d'umore, il minuto prima si arrabbia e mi urla contro e quello successivo è dolce e si scusa"),
    ("IT13","Mi sento giudicata rispetto alle cose che faccio e a come organizzo la mia vita"),
    ("IT18","Sento disagio o non voglio fare qualcosa ma lo nascondo per paura che si arrabbi"),
    ("IT31","Mi dice che non valgo niente e che resterei sola se non stessi con lui perché nessuno mi vorrebbe"),
    ("IT32","Ha mostrato mie immagini intime ad altre persone senza il mio consenso"),
    ("IT42","Mi chiede insistentemente di fare sesso e non accetta quando dico di no o che non mi va"),
    ("IT33","Minaccia di mostrare ai miei amici o ai miei familiari mie immagini intime"),
    ("IT34","Mi dice che, se lo lasciassi, rivelerebbe ad altri dei miei segreti o delle cose molto personali che sa di me"),
    ("IT35","Minaccia di fare del male a me, ai miei amici/amiche o alla mia famiglia"),
    ("IT36","Minaccia di farsi del male a causa mia"),
    ("IT37","Minacce di distruggere le mie cose (telefono, vestiti, computer, auto, ecc.)"),
]

SYSTEM_PROMPT = """You are a strict labeler. You must read an English narrative and,
for each item in the provided list (items are in Italian), return:
- "label": true if the aspect is clearly present in the narrative, otherwise false.
- "xai": a short explanation in English (1–3 sentences) justifying the choice,
  citing words/clues from the narrative; if label=false, explain why there is no evidence.
Rules:
- Do not infer beyond what is written: avoid speculation.
- Consider synonyms and paraphrases.
- If the text is ambiguous or neutral regarding an item, mark it as false.
- Respond ONLY in valid JSON, no additional text.
Expected schema:
{
  "items": {
    "IT10": {"label": true|false, "xai": "..."},
    "IT12": {"label": true|false, "xai": "..."},
    ...
  },
  "meta": {
    "model": "<model-name>",
    "confidence_note": "short note on uncertainties or limitations"
  }
}
"""

def build_user_prompt(narrative: str) -> str:
    items_block = "\n".join([f'- {k}: {v}' for k,v in ITEMS])
    return textwrap.dedent(f"""\
    NARRATIVE (English):
    \"\"\"{narrative.strip()}\"\"\"

    LIST OF ITEMS TO EVALUATE (keep exactly these IDs and texts in Italian):
    {items_block}

    OUTPUT INSTRUCTIONS:
    - Return JSON strictly matching the schema described in the system message.
    - Use "label" (boolean) and "xai" (string) for each item.
    - Do not add any text outside the JSON.
    """)

def make_client(base_url: str = DEFAULT_BASE_URL, api_key: str = DEFAULT_API_KEY) -> OpenAI:
    # Point the OpenAI SDK to Ollama's OpenAI-compatible endpoint.
    return OpenAI(base_url=base_url, api_key=api_key)

def call_ollama_via_openai(client: OpenAI, model: str, narrative: str, temperature: float = 0.0, retries: int = 2) -> dict:
    last_err = None
    for _ in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(narrative)},
                ],
                temperature=temperature,
                stream=False,
                response_format={"type": "json_object"},  # enforce JSON
            )
            content = (resp.choices[0].message.content or "").strip()
            return json.loads(content)
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    raise RuntimeError(f"Ollama/OpenAI call failed: {last_err}")

def ensure_all_items(items_obj: dict) -> dict:
    out = {}
    for key, _text in ITEMS:
        entry = items_obj.get(key)
        if isinstance(entry, dict) and "label" in entry and "xai" in entry:
            out[key] = {"label": bool(entry["label"]), "xai": str(entry["xai"])}
        else:
            out[key] = {"label": False, "xai": "No clear evidence in the narrative for this item."}
    return out

def run_batch(in_path: Path, outdir: Path, model: str, temperature: float, base_url: str, api_key: str) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise AssertionError("Input JSON must be a list of narratives.")

    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "results_combined.jsonl"
    csv_path   = outdir / "results_summary.csv"

    client = make_client(base_url=base_url, api_key=api_key)

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["narrative_id", "item_id", "label", "xai"])

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i, entry in enumerate(data, 1):
            narrative_id = str(entry.get("narrative_id") or f"auto_{uuid.uuid4().hex[:8]}")
            narrative_txt = str(entry.get("text") or "")
            if not narrative_txt.strip():
                print(f"[{i}/{len(data)}] {narrative_id}: EMPTY TEXT — skipping.", file=sys.stderr)
                continue

            print(f"[{i}/{len(data)}] {narrative_id}: sending to model...", file=sys.stderr)
            try:
                res = call_ollama_via_openai(client=client, model=model, narrative=narrative_txt, temperature=temperature)
            except Exception as e:
                err_obj = {"narrative_id": narrative_id, "error": f"{type(e).__name__}: {e}"}
                (outdir / f"{narrative_id}.error.json").write_text(json.dumps(err_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  -> ERROR, written {narrative_id}.error.json", file=sys.stderr)
                continue

            items = ensure_all_items(res.get("items", {}))
            meta  = res.get("meta", {})
            meta.setdefault("model", model)

            out_obj = {"narrative_id": narrative_id, "items": items, "meta": meta}

            out_path = outdir / f"{narrative_id}.json"
            out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            jf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            for item_id, vals in items.items():
                csv_writer.writerow([narrative_id, item_id, bool(vals.get("label")), str(vals.get("xai", ""))])

            print(f"  -> saved {out_path.name}", file=sys.stderr)

    csv_file.close()
    print(f"\nAll done.\n- Per-narrative JSONs in: {outdir}\n- Combined JSONL: {jsonl_path}\n- Summary CSV: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch questionnaire labeler (Ollama via OpenAI SDK).")
    parser.add_argument("--in", dest="infile", required=True, help="Path to input JSON list of narratives.")
    parser.add_argument("--outdir", default=None, help="Output directory (default: results_<timestamp>)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name as known to Ollama (e.g., llama3.1:8b)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL (default: http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key (dummy ok for Ollama; default: 'ollama')")
    args = parser.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else Path(f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        run_batch(
            in_path=in_path,
            outdir=outdir,
            model=args.model,
            temperature=args.temperature,
            base_url=args.base_url,
            api_key=args.api_key,
        )
    except AssertionError as e:
        print(f"Bad input: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
