"""Utility script to visualize denoising autoencoder (DAE) noise.

Features:
 1. Show side-by-side original vs noisy text for provided examples.
 2. Run multiple stochastic noise generations per example to observe variance.
 3. Summarize token length change and removal ratio statistics across runs.
 4. Optional diff-style highlighting (uses difflib) to make changes clear.

Usage (from repo root):
  python -m analysis.data_viz \
      --example "The quick brown fox jumps over the lazy dog" \
      --removal-prob 0.2 --swap-prob 0.15 --num-runs 5

Or provide multiple examples:
  python -m analysis.data_viz \
      --example "An autoencoder learns identity." \
      --example "Adding noise encourages robust representations." \
      --num-runs 3

You can also pass a text file (one example per line):
  python -m analysis.data_viz --file examples.txt --num-runs 4

All arguments are optional; if none provided, a small built-in sample set is used.
"""

from __future__ import annotations

import argparse
import difflib
import statistics
import random
import textwrap
from typing import Iterable, List, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from data.data_processing import add_noise


def _tokenize(text: str) -> List[str]:
    return text.split()


def noisy_variants(
    text: str,
    removal_prob: float,
    swap_prob: float,
    num_runs: int,
) -> List[Dict]:
    """Generate multiple noisy variants for a single text.

    Returns list of dicts with: noisy, length, removed_tokens, removal_ratio.
    """
    original_tokens = _tokenize(text)
    original_len = len(original_tokens)
    variants = []
    for _ in range(num_runs):
        noisy = add_noise(text, removal_prob=removal_prob, swap_prob=swap_prob)
        noisy_tokens = _tokenize(noisy)
        # Approximate removed tokens via multiset difference
        removed = max(0, original_len - len(noisy_tokens))
        variants.append(
            {
                "noisy": noisy,
                "length": len(noisy_tokens),
                "removed_tokens": removed,
                "removal_ratio": removed / original_len if original_len else 0.0,
            }
        )
    return variants


def highlight_diff(original: str, noisy: str) -> str:
    """Return a compact inline diff using difflib (tokens).

    Deletions: [-token-]
    Insertions: [+token+]
    (Insertions appear only as an artifact of swaps; algorithm never truly inserts.)
    """
    o_tokens = _tokenize(original)
    n_tokens = _tokenize(noisy)
    sm = difflib.SequenceMatcher(a=o_tokens, b=n_tokens)
    out_parts: List[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out_parts.extend(o_tokens[i1:i2])
        elif op == "delete":
            for t in o_tokens[i1:i2]:
                out_parts.append(f"[-{t}-]")
        elif op == "insert":
            for t in n_tokens[j1:j2]:
                out_parts.append(f"[+{t}+]")  # inserted (rare, from swaps side-effect)
        elif op == "replace":
            # Replacement approximated as deletion + insertion
            for t in o_tokens[i1:i2]:
                out_parts.append(f"[-{t}-]")
            for t in n_tokens[j1:j2]:
                out_parts.append(f"[+{t}+]")
    return " ".join(out_parts)


# ------------------------------ HTML COLOR DIFF ---------------------------------
HTML_CSS = """
<style>
.dae-wrapper {font-family: system-ui, sans-serif; line-height:1.4;}
.dae-block {margin:1em 0; padding:0.75em; border:1px solid #ccc; border-radius:6px; background:#fafafa;}
.dae-original {font-weight:600; margin-bottom:0.25em;}
.dae-row {margin:0.35em 0;}
.tok {padding:2px 4px; margin:1px 1px; display:inline-block; border-radius:3px; font-size:13px;}
.eq {background:#e8f5e9; color:#1b5e20;}
.del {background:#ffebee; color:#b71c1c; text-decoration:line-through;}
.ins {background:#e3f2fd; color:#0d47a1;}
.rep-old {background:#fff3e0; color:#e65100; text-decoration:line-through;}
.rep-new {background:#fff8e1; color:#ff6f00;}
.mov {background:#ede7f6; color:#4a148c;}
.meta {color:#555; font-size:12px; margin-bottom:4px; font-family:monospace;}
.legend span {margin-right:14px;}
.legend .box {display:inline-block; width:14px; height:14px; vertical-align:middle; margin-right:4px; border-radius:3px;}
</style>
"""

def build_colored_diff_html(original: str, noisy: str, mark_moves: bool = False) -> str:
    """Return HTML spans color-coding token-level changes.

    Because the DAE noise only removes and swaps adjacent tokens, true replacements almost never occur.
    Swaps appear to difflib as delete+insert pairs. If mark_moves=True, we attempt to detect tokens that
    were deleted then re-inserted (i.e., moved) and color them .mov instead of generic .ins.
    """
    from collections import Counter, deque
    o_tokens = _tokenize(original)
    n_tokens = _tokenize(noisy)
    sm = difflib.SequenceMatcher(a=o_tokens, b=n_tokens)
    parts = []

    # First pass gather deletes & inserts for potential move detection
    deleted_pool = Counter()
    inserted_order = []  # list of tokens in order they appear in insert segments
    opcodes = sm.get_opcodes()
    for op, i1, i2, j1, j2 in opcodes:
        if op == "delete":
            for t in o_tokens[i1:i2]:
                deleted_pool[t] += 1
        elif op == "replace":
            # treat original side as deletions, new side as insertions (rare here)
            for t in o_tokens[i1:i2]:
                deleted_pool[t] += 1
            for t in n_tokens[j1:j2]:
                inserted_order.append(t)
        elif op == "insert":
            for t in n_tokens[j1:j2]:
                inserted_order.append(t)

    # Multiset for moved detection: tokens both deleted and inserted -> moves
    move_pool = Counter()
    if mark_moves:
        for tok in inserted_order:
            if deleted_pool[tok] > 0:
                move_pool[tok] += 1
                deleted_pool[tok] -= 1

    # Second pass build HTML
    # We need to re-consume insertion tokens to decide if each is moved or plain insert.
    # So keep a queue of inserted tokens for mapping.
    inserted_queue = deque()
    for tok in inserted_order:
        inserted_queue.append(tok)

    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            for t in o_tokens[i1:i2]:
                parts.append(f'<span class="tok eq">{t}</span>')
        elif op == "delete":
            for t in o_tokens[i1:i2]:
                # If mark_moves and token scheduled as moved (remaining in move_pool), skip showing deletion to avoid duplication.
                if mark_moves and move_pool[t] > 0:
                    # Represent only once in its new position.
                    move_pool[t] -= 1
                else:
                    parts.append(f'<span class="tok del">{t}</span>')
        elif op == "insert":
            for t in n_tokens[j1:j2]:
                # Pop left to align with earlier queue population
                if inserted_queue:
                    inserted_queue.popleft()
                cls = "ins"
                if mark_moves and move_pool[t] >= 0:  # moved already accounted above? use .mov if token not newly inserted
                    # We can't differentiate which occurrence precisely; if token existed in original but showed up here after deletion
                    # it's considered moved (swap). We'll mark as moved if original had it.
                    if t in o_tokens:
                        cls = "mov"
                parts.append(f'<span class="tok {cls}">{t}</span>')
        elif op == "replace":
            # Rare here, but keep logic
            for t in o_tokens[i1:i2]:
                parts.append(f'<span class="tok rep-old">{t}</span>')
            for t in n_tokens[j1:j2]:
                parts.append(f'<span class="tok rep-new">{t}</span>')
    return "".join(parts)

def build_html_document(examples_payload, removal_prob: float, swap_prob: float, mark_moves: bool) -> str:
    legend = (
        '<div class="legend">'
        '<span><span class="box" style="background:#e8f5e9;border:1px solid #c8e6c9"></span>unchanged</span>'
        '<span><span class="box" style="background:#ffebee;border:1px solid #ffcdd2"></span>removed</span>'
        '<span><span class="box" style="background:#e3f2fd;border:1px solid #bbdefb"></span>inserted</span>'
        '<span><span class="box" style="background:#ede7f6;border:1px solid #d1c4e9"></span>moved (swap)</span>'
        '<span><span class="box" style="background:#fff3e0;border:1px solid #ffe0b2"></span>orig replace*</span>'
        '<span><span class="box" style="background:#fff8e1;border:1px solid #ffecb3"></span>new replace*</span>'
        '</div>'
    )
    note = '<p style="font-size:11px;color:#555;margin-top:4px">*Replacements rarely appear; the DAE noise does not substitute tokens, only removes or swaps them. Swaps are shown as moved when --mark-moves is enabled.</p>'
    blocks = []
    for idx, ex in enumerate(examples_payload, 1):
        original = ex["original"]
        variants = ex["variants"]
        block_lines = [f'<div class="dae-block">', f'<div class="dae-original">Example {idx} Original</div>', f'<div class="dae-row">{build_colored_diff_html(original, original, mark_moves=mark_moves)}</div>']
        for run_idx, variant in enumerate(variants, 1):
            diff_html = build_colored_diff_html(original, variant["noisy"], mark_moves=mark_moves)
            meta = (
                f"Run {run_idx}: len={variant['length']} rem={variant['removed_tokens']} "
                f"({variant['removal_ratio']:.1%})"
            )
            block_lines.append(f'<div class="meta">{meta}</div>')
            block_lines.append(f'<div class="dae-row">{diff_html}</div>')
        block_lines.append('</div>')
        blocks.append("\n".join(block_lines))
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>DAE Noise Visualization</title>"
        + HTML_CSS
        + "</head><body><div class='dae-wrapper'>"
        + f"<h2>DAE Noise Visualization (removal_prob={removal_prob}, swap_prob={swap_prob})</h2>"
        + legend
        + note
        + "".join(blocks)
        + "</div></body></html>"
    )
    return html


def summarize_variants(variants: List[Dict]) -> Dict:
    lengths = [v["length"] for v in variants]
    removal_ratios = [v["removal_ratio"] for v in variants]
    return {
        "runs": len(variants),
        "avg_length": statistics.mean(lengths) if lengths else 0.0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "avg_removal_ratio": statistics.mean(removal_ratios) if removal_ratios else 0.0,
    }


def _format_with_markers(original: str, noisy: str) -> str:
    """Return noisy text with tokens that disappeared from original wrapped in { }.

    This is a simpler visual than full diff when plotting.
    """
    o_tokens = _tokenize(original)
    n_tokens = _tokenize(noisy)
    # Map token counts
    from collections import Counter
    oc = Counter(o_tokens)
    nc = Counter(n_tokens)
    removed = []
    for tok, cnt in oc.items():
        if nc[tok] < cnt:
            removed.append(tok)
    marked = []
    removed_set = set(removed)
    for t in n_tokens:
        if t in removed_set:
            marked.append(f"{t}")  # could style later
        else:
            marked.append(t)
    return " ".join(marked)


def _plot_example(
    ax: Axes,
    original: str,
    variants: List[Dict],
    title: str,
    show_original: bool = True,
    wrap_width: int = 0,
):
    ax.axis("off")
    lines = []
    if show_original:
        lines.append("Original:")
        lines.append(original)
        lines.append("")
    for i, v in enumerate(variants, 1):
        lines.append(
            f"Run {i}: len={v['length']} rem={v['removed_tokens']} ({v['removal_ratio']:.0%})"
        )
        lines.append(_format_with_markers(original, v["noisy"]))
        lines.append("")
    if wrap_width and wrap_width > 10:
        wrapped = []
        for ln in lines:
            if ln.strip() == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.fill(ln, width=wrap_width).splitlines())
        lines = wrapped
    ax.set_title(title, fontsize=10)
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)


def demonstrate_noise(
    examples: Iterable[str],
    removal_prob: float,
    swap_prob: float,
    num_runs: int,
    show_diff: bool = True,
    make_figure: bool = False,
    max_examples_per_fig: int = 4,
    save_path: Optional[str] = None,
    wrap_width: int = 0,
    make_html: bool = False,
    html_path: Optional[str] = None,
    mark_moves: bool = False,
) -> None:
    all_variants: List[Dict] = []
    collected_for_fig = []  # list of (original, variants)
    collected_for_html = []
    for idx, text in enumerate(examples, start=1):
        print("=" * 80)
        print(f"Example {idx}")
        print("Original:")
        print(text)
        print("-" * 80)
        variants = noisy_variants(text, removal_prob, swap_prob, num_runs)
        all_variants.extend(variants)
        if make_figure:
            collected_for_fig.append((text, variants))
        if make_html:
            collected_for_html.append({"original": text, "variants": variants})
        summary = summarize_variants(variants)
        for run_i, v in enumerate(variants, start=1):
            print(f"Run {run_i}: length={v['length']} removed={v['removed_tokens']} ({v['removal_ratio']:.1%})")
            if show_diff:
                print("  noisy:")
                print("  " + v["noisy"])
        print("Summary:")
        print(
            f"  runs={summary['runs']} avg_len={summary['avg_length']:.2f} "
            f"min={summary['min_length']} max={summary['max_length']} avg_removal={summary['avg_removal_ratio']:.1%}"
        )
        # Single highlighted diff for first variant (if requested)
        if show_diff and variants:
            print("Diff (original -> first noisy):")
            print(highlight_diff(text, variants[0]["noisy"]))
        # When enough examples collected for a figure, render batch
        if make_figure and (
            len(collected_for_fig) == max_examples_per_fig
        ):
            _render_figure(collected_for_fig, removal_prob, swap_prob, save_path, wrap_width)
            collected_for_fig.clear()
    if all_variants:
        overall = summarize_variants(all_variants)
        print("=" * 80)
        print("Overall across all examples:")
        print(
            f"  total_runs={overall['runs']} avg_len={overall['avg_length']:.2f} "
            f"avg_removal={overall['avg_removal_ratio']:.1%}"
        )
    print("=" * 80)
    # Render remaining examples if any pending
    if make_figure and collected_for_fig:
        _render_figure(collected_for_fig, removal_prob, swap_prob, save_path, wrap_width)
    if make_html and collected_for_html:
        html = build_html_document(collected_for_html, removal_prob, swap_prob, mark_moves)
        out_path = html_path or "dae_noise_visualization.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[html saved] {out_path}")


def _render_figure(
    items: List, removal_prob: float, swap_prob: float, save_path: Optional[str], wrap_width: int
):
    rows = len(items)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows))
    if rows == 1:
        axes = [axes]
    for ax, (original, variants) in zip(axes, items):
        _plot_example(ax, original, variants, title="DAE Noise Variants", wrap_width=wrap_width)
    fig.suptitle(
        f"DAE Noise Visualization (removal_prob={removal_prob}, swap_prob={swap_prob})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save_path:
        root = save_path.rsplit(".", 1)[0]
        out_path = f"{root}.png" if not save_path.endswith(".png") else save_path
        fig.savefig(out_path, dpi=150)
        print(f"[figure saved] {out_path}")
    else:
        plt.show()


def _load_examples_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize DAE noise application")
    p.add_argument(
        "--example",
        action="append",
        help="Text example (can be used multiple times).",
    )
    p.add_argument("--file", help="Path to a text file with one example per line.")
    p.add_argument("--removal-prob", type=float, default=0.1, help="Token removal probability.")
    p.add_argument("--swap-prob", type=float, default=0.05, help="Adjacent token swap probability.")
    p.add_argument("--num-runs", type=int, default=3, help="Stochastic runs per example.")
    p.add_argument("--seed", type=int, help="Random seed for reproducibility (Python's random).")
    p.add_argument("--no-diff", action="store_true", help="Disable diff view.")
    p.add_argument("--figure", action="store_true", help="Generate matplotlib figure.")
    p.add_argument(
        "--save-fig", help="Path to save figure (PNG). If omitted, shows interactively."
    )
    p.add_argument(
        "--examples-per-fig", type=int, default=4, help="Max examples per figure."
    )
    p.add_argument(
        "--wrap-width", type=int, default=0, help="Wrap text in figure at this width (0 disables)."
    )
    p.add_argument("--html", action="store_true", help="Generate colored HTML diff visualization.")
    p.add_argument("--save-html", help="Path to save HTML (default: dae_noise_visualization.html)")
    p.add_argument("--mark-moves", action="store_true", help="Highlight moved (swapped) tokens.")
    return p.parse_args()


def main():
    args = parse_args()
    examples: List[str] = []
    if args.example:
        examples.extend(args.example)
    if args.file:
        try:
            examples.extend(_load_examples_from_file(args.file))
        except FileNotFoundError:
            print(f"[warn] file not found: {args.file}")
    if not examples:
        examples = [
            "Denoising autoencoders learn to reconstruct original inputs.",
            "Noise injection encourages robust latent representations.",
        ]
    if args.seed is not None:
        random.seed(args.seed)

    demonstrate_noise(
        examples=examples,
        removal_prob=args.removal_prob,
        swap_prob=args.swap_prob,
        num_runs=args.num_runs,
        show_diff=not args.no_diff,
        make_figure=args.figure,
        max_examples_per_fig=args.examples_per_fig,
        save_path=args.save_fig,
        wrap_width=args.wrap_width,
        make_html=args.html,
        html_path=args.save_html,
        mark_moves=args.mark_moves,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
