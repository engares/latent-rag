from __future__ import annotations
from typing import Dict, List, Optional
from datasets import load_dataset

__all__ = [
    "SquadExample",
    "load_squad",
]

SquadExample = Dict[str, str]  # keys: question, context, answer


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_squad(
    *,
    version: str = "v1",  # "v1" | "v2"
    split: str = "train",  # "train" | "validation"
    max_samples: Optional[int] = None,
    keep_unanswerable: bool = False,
) -> List[SquadExample]:
    """Return a list of canonical SQuAD examples.

    Args:
        version: Which dataset variant to use.  "v2" contains unanswerable
            questions.  "v1" provides only answerable ones.
        split: Dataset split to load.  Must be either "train" or "validation".
        max_samples: Optional hard cap on the number of examples returned.  Use
            this for quick prototyping.
        keep_unanswerable: If *False* (default) and *version == 'v2'*, examples
            without an answer are filtered out.

    Returns:
        A list of dicts with keys ``question``, ``context`` and ``answer`` (the
        first answer string provided by the annotation set).
    """

    if version not in {"v1", "v2"}:
        raise ValueError("version must be 'v1' or 'v2'")
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    hf_name = "squad_v2" if version == "v2" else "squad"
    ds = load_dataset(hf_name, split=split)

    examples: List[SquadExample] = []
    for ex in ds:
        question = ex["question"].strip()
        context = ex["context"].strip()

        # v2 contains lists of possible answers + a boolean flag
        if version == "v2":
            if not ex["answers"]["answer_start"] and not keep_unanswerable:
                # skip no annotated answer 
                continue
            answer = ex["answers"]["text"][0] if ex["answers"]["text"] else ""
        else:  # v1
            answer = ex["answers"][0]["text"]

        examples.append({"question": question, "context": context, "answer": answer})

    if max_samples is not None:
        examples = examples[: max_samples]

    return examples
