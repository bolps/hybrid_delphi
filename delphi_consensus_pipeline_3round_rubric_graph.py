import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import random
import re
from typing import List, Dict, Any, Tuple, Set
from tabulate import tabulate
import pandas as pd

random.seed(42)

# =====================
# Setup
# =====================
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# =====================
# Utilities
# =====================
def strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = re.sub(r'^\s*```(?:json|python)?\s*', '', s.strip())
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

def try_json_loads(s: str):
    try:
        return json.loads(strip_code_fences(s))
    except Exception:
        return None

def iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    def percentile(p: float) -> float:
        if n == 1:
            return vs[0]
        k = (p/100) * (n-1)
        f = int(k)
        c = min(f+1, n-1)
        if f == c:
            return vs[f]
        return vs[f] + (vs[c]-vs[f])*(k-f)
    return percentile(75) - percentile(25)

def mean(values: List[float]) -> float:
    return sum(values)/len(values) if values else 0.0

def safe_filename(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')

def log_response(expert, round_label="round1"):
    filename = f"logs/{timestamp}_{safe_filename(expert.name)}_{round_label}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"### {expert.name}\n")
        f.write(expert.response or "No response.\n")

def save_result_file(name, content):
    filename = f"results/{timestamp}_{safe_filename(name)}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename

def save_json_result(name, obj):
    filename = f"results/{timestamp}_{safe_filename(name)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return filename

# =====================
# API Client
# =====================
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# =====================
# Canonical items (with IDs)
# =====================
items = [
    "Sento il suo sostegno nelle cose che faccio",
    "Mi incoraggia a provare nuove cose",
    "Mi sento ascoltata e si mostra interessato quando ho qualcosa che voglio condividere",
    "Capisce che ho bisogno anche di miei spazi personali e comprende che ci sono momenti in cui mi piace stare da sola o con persone diverse da lui (amici/amiche, familiari ecc.)",
    "Mi fa sentire libera di fare le mie scelte su come gestire il mio tempo",
    "Mi vuole bene per quella che sono e rispetta le mie scelte",
    "Sento che possiamo confrontarci e anche litigare senza farci male",
    "In sua compagnia mi sento a mio agio e al sicuro",
    "I miei amici/amiche mi hanno fatto notare che non piace loro come si comporta con me",
    "Parla male delle mie amici/amiche e dice che sono dei/delle “poco di buono”",
    "Mi chiede di non vedere i miei amici/amiche",
    "Dice che sono impegnata in troppe cose e che lo trascuro, vorrebbe che passassimo la maggior parte del tempo insieme",
    "Mi sento giudicata rispetto alle cose che faccio e a come organizzo la mia vita",
    "Quando faccio qualcosa che a lui non va bene poi non mi parla per giorni",
    "Sento che mi manda troppi messaggi o mi chiama in modo insistente",
    "Non mi fa sentire bene quando parla di come mi prendo cura di me stessa e del mio aspetto fisico",
    "Provo paura vicino a lui, non mi sento al sicuro",
    "Sento disagio o non voglio fare qualcosa ma lo nascondo per paura che si arrabbi",
    "Le mie amiche o la mia famiglia sono preoccupate per me",
    "Pretende prove d'amore e dimostrazioni di fedeltà nei suoi confronti",
    "Diventa estremamente geloso o possessivo",
    "Mi accusa di flirtare o di tradirlo",
    "Mi sento controllata, ho la sensazione che non si fidi di me e che voglia sapere sempre dove sono e con chi, me lo chiede continuamente",
    "Non mi sento libera di vestirmi come voglio",
    "Cerca di controllare ciò che faccio e con chi mi vedo",
    "Non vuole che io veda la mia famiglia, i miei amici/amiche",
    "Ha forti oscillazioni d'umore, il minuto prima si arrabbia e mi urla contro e quello successivo è dolce e si scusa",
    "In sua compagnia mi capita di sentirmi a disagio e ho la sensazione di dover stare attenta a ciò che dico o a ciò che faccio, come se dovessi “camminare sulle uova” per evitare sue reazioni negative",
    "Mi butta giù, mi insulta o mi critica",
    "Mi fa sentire spesso sbagliata e mi dice che è colpa mia se sta male",
    "Mi dice che non valgo niente e che resterei sola se non stessi con lui perché nessuno mi vorrebbe",
    "Ha mostrato mie immagini intime ad altre persone senza il mio consenso",
    "Minaccia di mostrare ai miei amici o ai miei familiari mie immagini intime",
    "Mi dice che, se lo lasciassi, rivelerebbe ad altri dei miei segreti o delle cose molto personali che sa di me",
    "Minaccia di fare del male a me, ai miei amici/amiche o alla mia famiglia",
    "Minaccia di farsi del male a causa mia",
    "Minacce di distruggere le mie cose (telefono, vestiti, computer, auto, ecc.)",
    "Mi afferra per i capelli, mi strattona, mi mette le mani al collo, tenta di soffocarmi, mi dà dei pugni, mi schiaffeggia, mi getta addosso degli oggetti o mi fa male in qualche modo",
    "Dopo che mi ha aggredito (fisicamente o verbalmente) nega di averlo fatto e mi fa dubitare che sia realmente successo.",
    "Rompe e lancia oggetti per intimidirmi",
    "Urla o mi umilia di fronte ad altre persone",
    "Mi chiede insistentemente di fare sesso e non accetta quando dico di no o che non mi va",
    "Ho paura di vederlo perché so che potrebbe avere comportamenti che mi possono fare stare male o sentire in imbarazzo",
    "Sento di potergli parlare di qualsiasi cosa",
    "Controlla come spendo i miei soldi",
    "Mi impedisce di avere accesso al conto corrente o alle mie risorse economiche",
    "Mi viene chiesto di giustificare ogni acquisto che faccio",
    "Mi impedisce cercare un lavoro per essere indipendente"
    ]

item_id_map = {f"IT{idx+1:02d}": text for idx, text in enumerate(items)}
id_by_text = {v: k for k, v in item_id_map.items()}

# =====================
# Models & roles
# =====================
models = [
    "google/gemini-2.5-flash-lite",
    "x-ai/grok-3-mini",
    "openai/gpt-4.1-mini",
]
moderator_model = "google/gemini-2.5-flash-lite"

roles = [
    {
        "role": "Clinical Psychologist",
        "system_prompt": "You are a clinical psychologist specializing in relational and affective disorders.",
        "instructions": "Group the following items according to clinical psychological constructs."
    },
    {
        "role": "Psychometrician",
        "system_prompt": "You are a psychometrician skilled in factor analysis and scale design.",
        "instructions": "Identify and label latent dimensions using psychometric logic."
    },
    {
        "role": "Psycholinguist",
        "system_prompt": "You are a psycholinguist analyzing lexical semantics and meaning in statements.",
        "instructions": "Group items based on semantic similarity and latent linguistic meaning."
    }
]

common_task = """
You are given a list of psychological items (e.g., statements, traits, behaviors).
Your goal is to uncover latent dimensions underlying the list.

For each latent dimension:
- Provide a clear label or name.
- List the items that belong to that dimension.
- Give a brief rationale for the grouping based on your perspective.

Be thoughtful, coherent, and concise. Avoid overlaps between dimensions.
"""

# =====================
# Expert class
# =====================
class Expert:
    def __init__(self, name, model, temperature, system_prompt, instructions):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.instructions = instructions
        self.full_prompt = None
        self.response = None

    def build_prompt(self, items_for_prompt, common_task_text):
        items_str = "\n".join(f"- {item}" for item in items_for_prompt)
        self.full_prompt = (
            f"{self.instructions}\n\n"
            f"{common_task_text}\n"
            f"Here are the items:\n{items_str}"
        )

    def run(self, client: OpenAI):
        if not self.full_prompt:
            raise ValueError("Prompt not built yet. Call build_prompt first.")
        try:
            print(f"🔍 Running expert: {self.name} ({self.model})")
            result = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.full_prompt}
                ]
            )
            self.response = result.choices[0].message.content
            return self.response
        except Exception as e:
            print(f"❌ Error for {self.name} ({self.model}): {e}")
            return None

# =====================
# ROUND 1
# =====================
experts = []
for model in models:
    for role in roles:
        shuffled_items = items[:]
        random.shuffle(shuffled_items)
        expert = Expert(
            name=f"{role['role']} ({model})",
            model=model,
            temperature=random.uniform(0.5, 0.6),
            system_prompt=role["system_prompt"],
            instructions=role["instructions"]
        )
        expert.build_prompt(shuffled_items, common_task)
        experts.append(expert)

for expert in experts:
    result = expert.run(client)
    if result:
        log_response(expert, round_label="round1")
        print(f"\n🧠 {expert.name} says:\n{result}\n{'-'*60}")

# Moderator (Round 1)
delphi_moderator = Expert(
    name="Delphi Moderator",
    model=moderator_model,
    temperature=random.uniform(0.5, 0.6),
    system_prompt="You are a Delphi moderator facilitating a group discussion.",
    instructions="Aggregate the results from the experts, highlighting areas of consensus and disagreement."
)

expert_summaries = []
for expert in experts:
    if expert.response:
        summary = f"### {expert.name}\n{expert.response}"
        expert_summaries.append(summary)

random.shuffle(expert_summaries)
delphi_input = "\n\n".join(expert_summaries)
delphi_moderator_prompt = (
    f"{delphi_moderator.instructions}\n\n"
    f"Here are the responses from the experts:\n\n{delphi_input}\n\n"
    "Please:\n"
    "- Propose a synthesized set of latent dimensions based on the expert analyses.\n"
    "- For each dimension, list all the related items and a brief description of the dimension.\n"
    "- Keep the response structured and clear."
)
delphi_moderator.build_prompt([delphi_input], "")
delphi_moderator.full_prompt = delphi_moderator_prompt
delphi_result = delphi_moderator.run(client)
if delphi_result:
    save_result_file("Delphi_Moderator_Round1", delphi_result)

round1_responses = {expert.name: expert.response for expert in experts if expert.response}

# =====================
# ROUND 2
# =====================
round2_experts = []
for model in models:
    for role in roles:
        expert_name = f"{role['role']} ({model})"
        previous_response = round1_responses.get(expert_name, "")

        revised_prompt = (
            f"You previously completed an expert analysis of a set of psychological items. "
            f"Below is your original response:\n\n"
            f"{previous_response}\n\n"
            f"Additionally, a Delphi Moderator has provided an aggregated synthesis of all expert analyses. "
            f"Please read it carefully and then revise your response accordingly, aiming to:\n"
            f"- Align with the collective synthesis where appropriate\n"
            f"- Suggest improvements or corrections if you believe the synthesis missed something important\n"
            f"- Provide a final set of latent dimensions, each with grouped items and a rationale\n"
            f"- Summarize the changes you made in your revised response and the rationale behind your revisions\n\n"
            f"### Delphi Moderator's synthesis:\n{delphi_result}"
        )

        revised_expert = Expert(
            name=f"{expert_name} - Round 2",
            model=model,
            temperature=random.uniform(0.5, 0.6),
            system_prompt=role["system_prompt"],
            instructions="Review the synthesis and your original work, and provide a revised and final list of latent dimensions with grouped items."
        )
        revised_expert.full_prompt = revised_prompt
        round2_experts.append(revised_expert)

for expert in round2_experts:
    result = expert.run(client)
    if result:
        log_response(expert, round_label="round2")
        print(f"\n🧠 {expert.name} (Round 2) says:\n{result}\n{'-'*60}")

final_expert_summaries = [
    f"### {expert.name}\n{expert.response}" for expert in round2_experts if expert.response
]
random.shuffle(final_expert_summaries)
final_input = "\n\n".join(final_expert_summaries)
final_moderator_prompt = (
    "Based on the revised Round 2 expert inputs, produce a final synthesis:\n"
    "- List the final latent dimensions with grouped items and rationale\n"
    "- Indicate remaining disagreements or divergences if any\n\n"
    f"{final_input}"
)

final_moderator = Expert(
    name="Delphi Moderator - Final Aggregation",
    model=moderator_model,
    temperature=random.uniform(0.5, 0.6),
    system_prompt="You are a Delphi moderator finalizing the consensus.",
    instructions="Finalize the aggregation of the revised expert inputs."
)
final_moderator.full_prompt = final_moderator_prompt
final_result = final_moderator.run(client)
if final_result:
    save_result_file("Delphi_Moderator_Round2_Final", final_result)

# ---------- Structure Round 2 synthesis to JSON ----------
structure_prompt = f"""
You are a careful data curator.
Convert the following Round 2 synthesis into STRICT JSON that maps dimensions to items.
Use ONLY items from the provided canonical list with IDs.
If you see paraphrases or near-duplicates, match them to the closest canonical item text.

Return JSON with this exact schema:
{{
  "dimensions": [
    {{
      "name": "string",
      "definition": "string",
      "items": [ {{"id": "ITxx", "text": "exact canonical text"}}, ... ]
    }},
    ...
  ]
}}

Round 2 synthesis:
---
{final_result}
---

Canonical items with IDs:
{json.dumps([{"id": k, "text": v} for k,v in item_id_map.items()], ensure_ascii=False, indent=2)}
"""
structurer = Expert(
    name="Synthesis Structurer",
    model=moderator_model,
    temperature=0.2,
    system_prompt="You convert semi-structured text into clean, validated JSON. Never include commentary.",
    instructions="Reformat to strict JSON following the given schema."
)
structurer.full_prompt = structure_prompt
structured_result_raw = structurer.run(client)
structured_json = try_json_loads(structured_result_raw)
if not structured_json or "dimensions" not in structured_json:
    structured_json = {
        "dimensions": [{
            "name": "General",
            "definition": "Auto-generated fallback dimension (structuring failed).",
            "items": [{"id": k, "text": v} for k,v in item_id_map.items()]
        }]
    }
save_json_result("Round2_Structured_Dimensions", structured_json)

# =====================
# ROUND 3 — Updated with rubric and evidence fields
# =====================
round3_system = "You are a psychometrician skilled in factor analysis and scale design."
round3_user_intro = """
Evaluate items within their assigned dimensions for item reduction.

Use this rubric BEFORE scoring:
- Fit (1–5): 5=core/indispensable; 4=strong; 3=mixed/side-aspect; 2=tangential; 1=misfit.
- Clarity (1–5): 5=unambiguous; 4=minor tweak; 3=noticeable ambiguity; 2=confusing; 1=unclear.

For EACH item, FIRST write brief evidence:
- evidence_for (≤15 words): why it fits the dimension.
- evidence_against (≤15 words): why it might not fit.

Then assign numeric ratings and a recommendation.

Return a STRICT JSON array with objects of this exact schema:
[
  {
    "item_id": "ITxx",
    "dimension": "string",
    "evidence_for": "≤15 words",
    "evidence_against": "≤15 words",
    "fit": 1-5 integer,
    "clarity": 1-5 integer,
    "redundant": true/false,
    "overlaps_with": ["ITyy", ...],
    "recommendation": "Retain" | "Revise" | "Drop",
    "justification": "short reason (≤30 words)"
  },
  ...
]

Be concise and consistent. Redundancy refers to semantic overlap with other items in the SAME dimension.
"""

def build_dimensions_block(structured: Dict[str, Any]) -> str:
    lines = []
    for dim in structured["dimensions"]:
        lines.append(f'- Dimension: "{dim["name"]}" — {dim.get("definition", "")}')
        for it in dim["items"]:
            lines.append(f'  • {it["id"]}: {it["text"]}')
    return "\n".join(lines)

dimensions_block = build_dimensions_block(structured_json)

# Create Round 3 experts
round3_experts = []
for model in models:
    for role in roles:
        ex = Expert(
            name=f'{role["role"]} ({model}) - Round 3',
            model=model,
            temperature=random.uniform(0.5, 0.6),
            system_prompt=round3_system,
            instructions="Apply rubric; provide evidence_for/against; return STRICT JSON as specified."
        )
        ex.full_prompt = (
            round3_user_intro
            + "\n\nThese are the agreed dimensions and items:\n"
            + dimensions_block
        )
        round3_experts.append(ex)

# Run Round 3 experts
for ex in round3_experts:
    res = ex.run(client)
    if res:
        log_response(ex, round_label="round3")
        print(f"\n🧠 {ex.name} (Round 3) says:\n{res}\n{'-'*60}")

# Parse Round 3 outputs
def collect_evaluations(round3_experts: List[Expert]) -> Dict[str, Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = {}
    for ex in round3_experts:
        data = try_json_loads(ex.response or "")
        if not isinstance(data, list):
            m = re.search(r'\[\s*{.*}\s*\]', ex.response or "", re.S)
            if m:
                data = try_json_loads(m.group(0))
        if not isinstance(data, list):
            continue

        for row in data:
            try:
                item_id = row["item_id"]
                dim = row.get("dimension", "")
                fit = int(row.get("fit", 0))
                clarity = int(row.get("clarity", 0))
                redundant = bool(row.get("redundant", False))
                overlaps = row.get("overlaps_with", []) or []
                rec = row.get("recommendation", "")
                just = row.get("justification", "")
                e_for = row.get("evidence_for", "")
                e_against = row.get("evidence_against", "")

                if item_id not in agg:
                    agg[item_id] = {
                        "dimension": dim,
                        "fit_scores": [],
                        "clarity_scores": [],
                        "redundancy_flags": [],
                        "overlaps_with_all": [],   # list of lists per expert
                        "recommendations": [],
                        "justifications": [],
                        "evidence_for": [],
                        "evidence_against": []
                    }
                agg[item_id]["dimension"] = dim or agg[item_id]["dimension"]
                agg[item_id]["fit_scores"].append(fit)
                agg[item_id]["clarity_scores"].append(clarity)
                agg[item_id]["redundancy_flags"].append(redundant)
                agg[item_id]["overlaps_with_all"].append(overlaps)
                agg[item_id]["recommendations"].append(rec)
                agg[item_id]["justifications"].append(just)
                if e_for:
                    agg[item_id]["evidence_for"].append(e_for)
                if e_against:
                    agg[item_id]["evidence_against"].append(e_against)
            except Exception:
                continue
    return agg

aggregated = collect_evaluations(round3_experts)
save_json_result("Round3_Aggregated_Raw", aggregated)

# Compute basic decisions per item (pre-network)
def apply_decision_rules(agg: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    decisions = {}
    dims_to_items: Dict[str, List[str]] = {}

    for item_id, info in agg.items():
        dim = info.get("dimension", "Unassigned")
        fit_mean = mean(info.get("fit_scores", []))
        clarity_mean = mean(info.get("clarity_scores", []))
        fit_iqr = iqr(info.get("fit_scores", []))
        red_flags = info.get("redundancy_flags", [])
        # keep redundancy_ratio only for reporting (NOT for decisions)
        red_ratio = (sum(1 for x in red_flags if x) / len(red_flags)) if red_flags else 0.0

        # Rule-based (no redundancy here; redundancy is network-based)
        reason_bits = []
        if fit_mean < 3.5:
            decision = "Drop"
            reason_bits.append(f"fit_mean={fit_mean:.2f} < 3.5")
        elif fit_mean >= 4.0 and fit_iqr <= 1.0 and clarity_mean >= 4.0:
            decision = "Retain"
            reason_bits.append(f"fit_mean≥4 ({fit_mean:.2f}), IQR≤1 ({fit_iqr:.2f}), clarity≥4 ({clarity_mean:.2f})")
        elif fit_mean >= 3.5 and (clarity_mean < 4.0 or fit_iqr > 1.0):
            decision = "Revise"
            if clarity_mean < 4.0:
                reason_bits.append(f"clarity_mean={clarity_mean:.2f} < 4.0")
            if fit_iqr > 1.0:
                reason_bits.append(f"fit_IQR={fit_iqr:.2f} > 1.0")
        else:
            recs = info.get("recommendations", [])
            if recs:
                decision = max(set(recs), key=recs.count)
                reason_bits.append("fallback to majority recommendation")
            else:
                decision = "Revise"
                reason_bits.append("insufficient data; set to Revise")

        decisions[item_id] = {
            "item_id": item_id,
            "item_text": item_id_map.get(item_id, ""),
            "dimension": dim,
            "fit_mean": round(fit_mean, 3),
            "clarity_mean": round(clarity_mean, 3),
            "fit_iqr": round(fit_iqr, 3),
            "redundancy_ratio": round(red_ratio, 3),  # descriptive only
            "decision": decision,
            "reason": "; ".join(reason_bits),
            "n_raters": len(info.get("fit_scores", []))
        }
        dims_to_items.setdefault(dim, []).append(item_id)

    return decisions, dims_to_items

decisions, decisions_by_dim = apply_decision_rules(aggregated)

# =====================
# Network redundancy (within-dimension) — MIS pruning
# =====================

def build_redundancy_graph(
    aggregated: Dict[str, Dict[str, Any]],
    threshold: float = 0.50,
    denom_mode: str = "max",  # "max" (conservative under unequal raters) or "min" (stricter)
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build {dimension: {item_id: set(neighbors)}} with an undirected edge
    when the proportion of raters flagging the pair as overlapping is >= threshold.
    Only within the same dimension.
    """
    pair_counts: Dict[Tuple[str, str], int] = {}
    pair_denoms: Dict[Tuple[str, str], int] = {}

    n_raters_per_item = {it: len(info.get("fit_scores", [])) for it, info in aggregated.items()}
    dim_of_item = {it: info.get("dimension", "Unassigned") for it, info in aggregated.items()}

    # Count pairwise flags
    for item_id, info in aggregated.items():
        for overlaps in info.get("overlaps_with_all", []):
            for other in overlaps or []:
                if other not in aggregated:
                    continue
                if dim_of_item.get(item_id) != dim_of_item.get(other):
                    continue
                a, b = sorted([item_id, other])
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

                if denom_mode == "min":
                    denom = min(n_raters_per_item.get(a, 1), n_raters_per_item.get(b, 1))
                else:
                    denom = max(n_raters_per_item.get(a, 1), n_raters_per_item.get(b, 1))
                # keep the *largest* plausible denominator we've seen
                pair_denoms[(a, b)] = max(pair_denoms.get((a, b), 0), denom)

    graph_by_dim: Dict[str, Dict[str, Set[str]]] = {}
    for (a, b), c in pair_counts.items():
        denom = pair_denoms.get((a, b), 1) or 1
        ratio = c / denom
        dim = dim_of_item.get(a, "Unassigned")
        if ratio >= threshold:
            graph_by_dim.setdefault(dim, {})
            graph_by_dim[dim].setdefault(a, set()).add(b)
            graph_by_dim[dim].setdefault(b, set()).add(a)

    # Add isolated nodes so every item appears
    for it, dim in dim_of_item.items():
        graph_by_dim.setdefault(dim, {})
        graph_by_dim[dim].setdefault(it, set())

    return graph_by_dim

def apply_pairwise_MIS_pruning(
    decisions: Dict[str, Any],
    aggregated: Dict[str, Dict[str, Any]],
    threshold: float = 0.50,
    denom_mode: str = "max",
) -> None:
    """
    Within each dimension, retain a maximal independent set chosen greedily by quality:
    - Sort items by (fit_mean desc, clarity_mean desc, redundancy_ratio asc [tiebreaker])
    - Iterate: keep an item if not already dominated; drop all of its neighbors (edges >= threshold)
    This guarantees no two retained items are connected by a >= threshold overlap edge.
    """
    graph_by_dim = build_redundancy_graph(aggregated, threshold=threshold, denom_mode=denom_mode)

    for dim, adj in graph_by_dim.items():
        order = sorted(
            adj.keys(),
            key=lambda it: (
                -decisions[it]["fit_mean"],
                -decisions[it]["clarity_mean"],
                 decisions[it]["redundancy_ratio"]
            )
        )
        kept: Set[str] = set()
        dropped: Set[str] = set()

        for it in order:
            if it in dropped:
                continue
            # Keep this item
            kept.add(it)
            # Hard-drop neighbors to satisfy independence
            for nb in adj[it]:
                if nb in kept or nb in dropped:
                    continue
                if decisions[nb]["decision"] != "Drop":
                    decisions[nb]["decision"] = "Drop"
                    decisions[nb]["reason"] += f"; pairwise redundancy ≥{threshold:.2f} vs {it}"
                dropped.add(nb)

# Replace cluster safeguard with MIS pruning (threshold=0.50 by default)
apply_pairwise_MIS_pruning(decisions, aggregated, threshold=0.50, denom_mode="max")

# =====================
# Coverage safeguard (≥2 retained per dimension) — unchanged
# =====================
def enforce_coverage(decisions: Dict[str, Any], min_per_dim: int = 2) -> None:
    by_dim: Dict[str, List[Dict[str, Any]]] = {}
    for _, rec in decisions.items():
        by_dim.setdefault(rec["dimension"], []).append(rec)

    for dim, recs in by_dim.items():
        retained = [r for r in recs if r["decision"] == "Retain"]
        if len(retained) >= min_per_dim:
            continue

        # 1) Promote from REVISE
        revise_candidates = sorted(
            (r for r in recs if r["decision"] == "Revise"),
            key=lambda r: (-r["fit_mean"], -r["clarity_mean"])
        )
        while len(retained) < min_per_dim and revise_candidates:
            promoted = revise_candidates.pop(0)
            promoted["decision"] = "Retain"
            promoted["reason"] += "; promoted for coverage safeguard"
            retained.append(promoted)

        # 2) If still short, undrop the best DROPs
        if len(retained) < min_per_dim:
            drop_candidates_primary = sorted(
                (r for r in recs if r["decision"] == "Drop" and r["fit_mean"] >= 3.5),
                key=lambda r: (-r["fit_mean"], -r["clarity_mean"], r["redundancy_ratio"])
            )
            drop_candidates_fallback = sorted(
                (r for r in recs if r["decision"] == "Drop" and r not in drop_candidates_primary),
                key=lambda r: (-r["fit_mean"], -r["clarity_mean"], r["redundancy_ratio"])
            )
            for pool, tag in [
                (drop_candidates_primary, "; undropped (strong) for coverage safeguard"),
                (drop_candidates_fallback, "; undropped (last-resort) for coverage safeguard")
            ]:
                while len(retained) < min_per_dim and pool:
                    revived = pool.pop(0)
                    revived["decision"] = "Retain"
                    revived["reason"] += tag
                    retained.append(revived)

# --- Network Redundancy Summary Table ---------------------------------

def build_redundancy_graph_with_counts(
    aggregated, threshold=0.50, denom_mode="max"
):
    """Same as build_redundancy_graph, but also returns pair_counts/denoms."""
    pair_counts, pair_denoms = {}, {}
    n_raters = {it: len(info.get("fit_scores", [])) for it, info in aggregated.items()}
    dim_of = {it: info.get("dimension", "Unassigned") for it, info in aggregated.items()}

    for a, info in aggregated.items():
        for overlaps in info.get("overlaps_with_all", []):
            for b in overlaps or []:
                if b not in aggregated: 
                    continue
                if dim_of[a] != dim_of[b]:
                    continue
                A, B = tuple(sorted([a, b]))
                pair_counts[(A, B)] = pair_counts.get((A, B), 0) + 1
                denom = (min if denom_mode == "min" else max)(n_raters.get(A,1), n_raters.get(B,1))
                pair_denoms[(A, B)] = max(pair_denoms.get((A,B), 0), denom)

    graph_by_dim = {}
    for (A, B), c in pair_counts.items():
        denom = pair_denoms.get((A, B), 1) or 1
        ratio = c / denom
        if ratio >= threshold:
            dim = dim_of[A]
            graph_by_dim.setdefault(dim, {})
            graph_by_dim[dim].setdefault(A, set()).add(B)
            graph_by_dim[dim].setdefault(B, set()).add(A)

    # add isolated nodes
    for it, dim in dim_of.items():
        graph_by_dim.setdefault(dim, {})
        graph_by_dim[dim].setdefault(it, set())

    return graph_by_dim, pair_counts, pair_denoms

def pair_ratio(a, b, pair_counts, pair_denoms):
    A, B = tuple(sorted([a, b]))
    c = pair_counts.get((A, B), 0)
    d = pair_denoms.get((A, B), 1) or 1
    return c / d

# Rebuild graph WITH counts so we can print exact ratios for neighbors
threshold = 0.50
denom_mode = "max"
graph_by_dim, pair_counts, pair_denoms = build_redundancy_graph_with_counts(
    aggregated, threshold=threshold, denom_mode=denom_mode
)

# Build a dedicated network table
net_rows = []
for dim, adj in graph_by_dim.items():
    for it, nbrs in adj.items():
        ratios = [(nb, round(pair_ratio(it, nb, pair_counts, pair_denoms), 3)) for nb in sorted(nbrs)]
        net_rows.append({
            "item_id": it,
            "dimension": dim,
            "fit_mean": decisions[it]["fit_mean"],
            "clarity_mean": decisions[it]["clarity_mean"],
            "decision": decisions[it]["decision"],
            "reason": decisions[it]["reason"],
            "net_degree": len(nbrs),
            "net_neighbors": ",".join([nb for nb, _ in ratios]),
            "net_neighbor_ratios": "|".join([f"{nb}:{r:.2f}" for nb, r in ratios]),
            "net_max_overlap": max([r for _, r in ratios], default=0.0),
            "net_drop_partner": decisions[it].get("net_drop_partner", ""),
            "net_drop_ratio": decisions[it].get("net_drop_ratio", ""),
            "net_threshold": threshold,
            "net_denom_mode": denom_mode,
        })

# Save as its own TSV
hdr = ["item_id","dimension","fit_mean","clarity_mean","decision","reason",
       "net_degree","net_neighbors","net_neighbor_ratios","net_max_overlap",
       "net_drop_partner","net_drop_ratio","net_threshold","net_denom_mode"]

lines = ["\t".join(hdr)]
for row in sorted(net_rows, key=lambda r: (r["dimension"], r["item_id"])):
    lines.append("\t".join(str(row[k]) for k in hdr))

save_result_file("Network_Redundancy_Table_MIS", "\n".join(lines))
print("🧭 Wrote results/Network_Redundancy_Table_MIS.txt")

enforce_coverage(decisions, min_per_dim=2)

# =====================
# Save decisions and exports (unchanged structure)
# =====================
save_json_result("Round3_Decisions_with_Rubric_and_NetworkMIS", {"items": list(decisions.values())})

# Tabular text
lines = ["item_id\titem_text\tdimension\tfit_mean\tclarity_mean\tfit_iqr\tdecision\treason"]

for item_id in sorted(decisions.keys()):
    d = decisions[item_id]
    item_text = item_id_map.get(item_id, "").replace("\t", " ")
    lines.append(
    f'{item_id}\t{item_text}\t{d["dimension"]}\t{d["fit_mean"]}\t'
    f'{d["clarity_mean"]}\t{d["fit_iqr"]}\t'
    f'{d["decision"]}\t{d["reason"]}')
save_result_file("Round3_Decisions_Table_with_NetworkMIS", "\n".join(lines))

# Moderator-style Round 3 synthesis (minor wording tweak)
synth_input = {
    "dimensions": {}
}
for it, rec in decisions.items():
    dim = rec["dimension"]
    synth_input["dimensions"].setdefault(dim, []).append(
        {k: v for k, v in rec.items() if k not in {"item_text"}}
    )

synth_prompt = f"""
You are a Delphi moderator writing the Round 3 synthesis AFTER computation.
Given these decisions (JSON), briefly explain:
- Which items were retained, revised, or dropped by dimension
- Main reasons (e.g., rubric thresholds; network redundancy MIS with ≥0.50 edges)
- Any unresolved disagreements or borderline cases

Keep it structured and concise (max ~300 words).

Decisions JSON:
{json.dumps(synth_input, ensure_ascii=False, indent=2)}
"""
moderator3 = Expert(
    name="Delphi Moderator - Round 3 Synthesis",
    model=moderator_model,
    temperature=0.3,
    system_prompt="You write clear, neutral consensus summaries.",
    instructions="Summarize the computed decisions."
)
moderator3.full_prompt = synth_prompt
round3_synthesis = moderator3.run(client) or "Summary unavailable."
save_result_file("Delphi_Moderator_Round3_Synthesis_with_NetworkMIS", round3_synthesis)
print(f"\n📌 Delphi Moderator - Round 3 Synthesis:\n{round3_synthesis}\n{'-'*60}")

# Full log
full_log = {
    "timestamp": timestamp,
    "round2_structured": structured_json,
    "experts_round3": {e.name: e.response for e in round3_experts},
    "round3_aggregated": aggregated,
    "round3_decisions": decisions,
    "round3_synthesis": round3_synthesis
}
save_json_result("Full_Log_Round3_with_NetworkMIS", full_log)

print("\n✅ Round 3 updated: rubric-only per item + within-dimension MIS redundancy (≥0.50). See 'results/'.")

# =====================
# Table — User friendly export of the table
# =====================

# Assign dimension IDs
dimension_order = sorted(set(d["dimension"] for d in decisions.values()))
dimension_id_map = {dim: f"D{idx+1:02d}" for idx, dim in enumerate(dimension_order)}

# Decision priority order
decision_priority = {"Retain": 0, "Revise": 1, "Drop": 2}

# Prepare table data
table_data = []
for dim in dimension_order:
    dim_items = sorted(
        [d for d in decisions.values() if d["dimension"] == dim],
        key=lambda x: (
            decision_priority.get(x["decision"], 99),  # order by decision
            x["item_id"]  # then by item_id
        )
    )
    for rec in dim_items:
        table_data.append([
            rec["item_id"],
            rec["item_text"],
            dimension_id_map[rec["dimension"]],
            rec["dimension"],
            rec["fit_mean"],
            rec["clarity_mean"],
            rec["fit_iqr"],
            rec["redundancy_ratio"],
            rec["decision"],
            rec["reason"]
        ])

headers = [
    "Item ID", "Item Text", "Dim ID", "Dimension Text",
    "Fit Mean", "Clarity Mean", "Fit IQR", "Redundancy",
    "Decision", "Reason"
]

df = pd.DataFrame(table_data, columns=headers)
df.drop(columns=["Redundancy"], inplace=True)  # Remove Dim ID for clarity
excel_path = f"results/{timestamp}_Round3_Decisions.xlsx"
df.to_excel(excel_path, index=False)
print(f"\n💾 Final decisions table saved to: {excel_path}")

#print("\n📊 Final Decisions Table:")
#print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".2f"))