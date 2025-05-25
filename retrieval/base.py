from __future__ import annotations
from typing import Protocol, Sequence, Tuple, List

class BaseRetriever(Protocol):
    """Interface for all first-stage retrievers."""
    def build_index(self, corpus: Sequence[str]) -> None: ...
    def retrieve(self, query: str, k: int) -> List[Tuple[str, float]]: ...
