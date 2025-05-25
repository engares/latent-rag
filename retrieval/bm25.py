from pyserini.search import SimpleSearcher            # pip install pyserini>=0.21
from retrieval.base import BaseRetriever
import tempfile, os, json

class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = bm25_k1, b
        self._searcher = None
        self._tmpdir   = tempfile.mkdtemp()

    def build_index(self, corpus):
        # 1) escribir cada doc en un fichero JSONL (id + text)
        tmp_jsonl = os.path.join(self._tmpdir, "docs.jsonl")
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for i, doc in enumerate(corpus):
                f.write(json.dumps({"id": str(i), "contents": doc}) + "\n")

        # 2) invocar indexador Lucene
        from pyserini.index import build_index
        build_index(tmp_jsonl, self._tmpdir, overwrite=True)

        # 3) abrir buscador Lucene
        self._searcher = SimpleSearcher(self._tmpdir)
        self._searcher.set_bm25(self.k1, self.b)

    def retrieve(self, query, k):
        hits = self._searcher.search(query, k)
        return [(h.raw, float(h.score)) for h in hits]
