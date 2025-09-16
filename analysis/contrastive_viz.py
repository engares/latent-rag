"""Visualización de tripletas (query, positive, negative) para datasets
contrastivos / contrastive autoencoders.

Características:
 - Carga desde:
      * JSON de ejemplos originales (query + positive_passages + negative_passages)
      * JSON/JSONL de pares ya construidos (query, positive, negative)
      * Tripletas inline vía CLI (--triplet "q|||p|||n") múltiples veces
 - Muestreo secuencial o aleatorio (--sample-random)
 - Estadísticas básicas (longitudes, Jaccard con positivos/negativos)
 - Exportación HTML con coloreado de tokens según pertenencia:
        Verde  : tokens compartidos con query
        Azul   : sólo en positivo
        Rojo   : sólo en negativo
        Gris   : en query pero ausentes en el ejemplo correspondiente

Uso rápido:
  python -m analysis.contrastive_viz \
      --triplet "What is AI?|||AI is artificial intelligence.|||AI is a type of food." \
      --triplet "What is ML?|||ML stands for machine learning.|||ML is a movie title." \
      --html --save-html contrastive_triplets.html

Para construir desde ejemplos originales (como build_contrastive_pairs):
  python -m analysis.contrastive_viz --original-json raw_examples.json --max-negatives 1 --html
"""

from __future__ import annotations

import argparse
import json
import random
import os
from typing import List, Dict, Iterable, Optional, Tuple
from collections import Counter

# ------------------------- DATA LOADING ----------------------------------

def load_original_examples(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_pairs(path: str) -> List[Dict]:
    # supports JSON (list) or JSONL
    if path.endswith(".jsonl"):
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_contrastive_pairs_from_original(examples: List[Dict], max_negatives: int = 1) -> List[Dict]:
    pairs: List[Dict] = []
    for ex in examples:
        q = ex["query"]
        if not ex.get("positive_passages"):
            continue
        pos = ex["positive_passages"][0]["text"]
        negs = [n["text"] for n in ex.get("negative_passages", [])[:max_negatives]]
        for neg in negs:
            pairs.append({"query": q, "positive": pos, "negative": neg})
    return pairs


def parse_inline_triplets(raw_triplets: List[str]) -> List[Dict]:
    triplets = []
    for t in raw_triplets:
        parts = t.split("|||")
        if len(parts) != 3:
            print(f"[warn] Formato inválido (se ignora): {t}")
            continue
        q, p, n = [s.strip() for s in parts]
        triplets.append({"query": q, "positive": p, "negative": n})
    return triplets


# ------------------------- TEXT / TOKENS ---------------------------------

def tokenize(text: str) -> List[str]:
    return text.strip().split()


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ------------------------- HTML RENDERING --------------------------------

HTML_CSS = """
<style>
body {font-family: system-ui, Arial, sans-serif; margin: 16px; background:#fcfcfc;}
.triplet {border:1px solid #ddd; padding:10px 12px; margin:12px 0; border-radius:6px; background:#fff;}
.head {font-weight:600; margin-bottom:4px;}
.row {margin:4px 0; line-height:1.45;}
.tok {display:inline-block; margin:1px 2px 1px 0; padding:2px 5px; border-radius:4px; font-size:13px;}
.qonly {background:#eeeeee; color:#555;}
.overlap {background:#e8f5e9; color:#1b5e20;}
.posonly {background:#e3f2fd; color:#0d47a1;}
.negonly {background:#ffebee; color:#b71c1c;}
.meta {font-size:11px; color:#555; margin-top:6px; font-family:monospace;}
.legend span {margin-right:16px; font-size:12px;}
.legend .box {display:inline-block; width:14px; height:14px; margin-right:4px; border-radius:3px; vertical-align:middle;}
h2 {margin-top:0;}
</style>
"""


def color_tokens(reference: List[str], other: List[str], mode: str) -> str:
    """Color tokens of 'other' relative to 'reference'.

    mode: 'positive' or 'negative'
    overlap -> .overlap
    only in other -> .posonly / .negonly
    only in reference (not shown here) handled separately if needed.
    """
    ref_set = set(reference)
    html_parts = []
    cls_other = "posonly" if mode == "positive" else "negonly"
    for tok in other:
        base = tok
        if tok in ref_set:
            html_parts.append(f'<span class="tok overlap">{base}</span>')
        else:
            html_parts.append(f'<span class="tok {cls_other}">{base}</span>')
    return "".join(html_parts)


def color_query_tokens(query: List[str], other: List[str]) -> str:
    other_set = set(other)
    parts = []
    for tok in query:
        if tok in other_set:
            parts.append(f'<span class="tok overlap">{tok}</span>')
        else:
            parts.append(f'<span class="tok qonly">{tok}</span>')
    return "".join(parts)


def build_html(triplets: List[Dict], limit: int, title: str) -> str:
    blocks = []
    for idx, t in enumerate(triplets[:limit], 1):
        q = tokenize(t["query"])
        p = tokenize(t["positive"])
        n = tokenize(t["negative"])
        p_j = jaccard(q, p)
        n_j = jaccard(q, n)
        block = [
            '<div class="triplet">',
            f'<div class="head">Tripleta {idx}</div>',
            f'<div class="row"><strong>Query:</strong> {"".join(color_query_tokens(q, q))}</div>',
            f'<div class="row"><strong>Positive:</strong> {color_tokens(q, p, "positive")}</div>',
            f'<div class="row"><strong>Negative:</strong> {color_tokens(q, n, "negative")}</div>',
            f'<div class="meta">len(q)={len(q)} len(p)={len(p)} len(n)={len(n)} | Jaccard(q,pos)={p_j:.2f} Jaccard(q,neg)={n_j:.2f}</div>',
            '</div>'
        ]
        blocks.append("\n".join(block))
    legend = (
        '<div class="legend" style="margin:8px 0 16px 0;">'
        '<span><span class="box" style="background:#e8f5e9;border:1px solid #c8e6c9"></span>Overlap</span>'
        '<span><span class="box" style="background:#e3f2fd;border:1px solid #bbdefb"></span>Solo en positive</span>'
        '<span><span class="box" style="background:#ffebee;border:1px solid #ffcdd2"></span>Solo en negative</span>'
        '<span><span class="box" style="background:#eeeeee;border:1px solid #e0e0e0"></span>Solo en query</span>'
        '</div>'
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Contrastive Triplets</title>"
        + HTML_CSS
        + "</head><body>"
        + f"<h2>{title}</h2>"
        + legend
        + "".join(blocks)
        + "</body></html>"
    )
    return html


# ------------------------- STATS / SUMMARY --------------------------------

def summarize_triplets(triplets: List[Dict]) -> Dict:
    lens_q = [len(tokenize(t["query"])) for t in triplets]
    lens_p = [len(tokenize(t["positive"])) for t in triplets]
    lens_n = [len(tokenize(t["negative"])) for t in triplets]
    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0
    return {
        "count": len(triplets),
        "avg_len_query": _avg(lens_q),
        "avg_len_positive": _avg(lens_p),
        "avg_len_negative": _avg(lens_n),
    }


# ------------------------- MAIN PIPELINE ----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualizar tripletas contrastivas")
    p.add_argument("--original-json", help="JSON con ejemplos originales (query, positive_passages, negative_passages)")
    p.add_argument("--pairs-json", help="JSON o JSONL con tripletas ya construidas (query, positive, negative)")
    p.add_argument("--triplet", action="append", help="Añadir tripleta inline: 'query|||positive|||negative'")
    p.add_argument("--max-negatives", type=int, default=1, help="Máx. negativos a usar por ejemplo original")
    p.add_argument("--limit", type=int, default=20, help="Máx. tripletas a mostrar/exportar")
    p.add_argument("--sample-random", action="store_true", help="Muestreo aleatorio en lugar de primeros N")
    p.add_argument("--seed", type=int, help="Seed para muestreo aleatorio")
    p.add_argument("--html", action="store_true", help="Generar HTML coloreado")
    p.add_argument("--save-html", help="Ruta de salida HTML (por defecto: contrastive_triplets.html)")
    # Filtros de solapamiento
    p.add_argument("--filter-overlap", action="store_true", help="Activar filtrado por Jaccard (positivos suficientemente relacionados y negativos muy diferentes)")
    p.add_argument("--min-pos-jaccard", type=float, default=0.05, help="Mínimo Jaccard query-positive (si se filtra)")
    p.add_argument("--max-neg-jaccard", type=float, default=0.20, help="Máximo Jaccard query-negative (si se filtra)")
    return p.parse_args()


def main():
    args = parse_args()
    triplets: List[Dict] = []

    if args.pairs_json:
        triplets.extend(load_pairs(args.pairs_json))
    if args.original_json:
        orig = load_original_examples(args.original_json)
        triplets.extend(build_contrastive_pairs_from_original(orig, max_negatives=args.max_negatives))
    if args.triplet:
        triplets.extend(parse_inline_triplets(args.triplet))

    # Normalización básica: asegurar claves
    cleaned = []
    for t in triplets:
        if all(k in t for k in ("query", "positive", "negative")):
            cleaned.append({
                "query": str(t["query"]).strip(),
                "positive": str(t["positive"]).strip(),
                "negative": str(t["negative"]).strip(),
            })
    triplets = cleaned

    if not triplets:
        print("[info] No se cargaron tripletas. Proporcione --triplet o archivos.")
        return

    if args.sample_random and len(triplets) > args.limit:
        if args.seed is not None:
            random.seed(args.seed)
        triplets = random.sample(triplets, args.limit)
    else:
        triplets = triplets[: args.limit]

    # Filtrado por solapamiento
    removed_overlap = 0
    if args.filter_overlap:
        kept = []
        for t in triplets:
            q_tok = tokenize(t["query"])
            p_tok = tokenize(t["positive"])
            n_tok = tokenize(t["negative"])
            p_j = jaccard(q_tok, p_tok)
            n_j = jaccard(q_tok, n_tok)
            if p_j >= args.min_pos_jaccard and n_j <= args.max_neg_jaccard:
                kept.append(t)
            else:
                removed_overlap += 1
        triplets = kept

    summary = summarize_triplets(triplets)
    print("=== Resumen ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    if args.filter_overlap:
        print(f"filtradas_por_overlap: {removed_overlap}")

    print("\n=== Ejemplos (máx) ===")
    for i, t in enumerate(triplets, 1):
        print(f"[{i}]")
        print("  Q:", t["query"])  
        print("  +:", t["positive"])  
        print("  -:", t["negative"])  

    if args.html:
        html = build_html(triplets, limit=len(triplets), title="Tripletas Contrastivas")
        out_path = args.save_html or "contrastive_triplets.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[html guardado] {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
