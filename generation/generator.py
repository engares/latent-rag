# generation/generator.py  – RAG (Refactor estilo train_vae)

from __future__ import annotations
import os, textwrap, logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI, AsyncOpenAI

def _load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.getLogger(__name__).warning("Prompt no encontrado: %s", path)
        return ""

@dataclass
class LLMSettings:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 512
    system_prompt_path: str = "./config/prompts/system_prompt.txt"
    system_prompt: str = field(init=False)

    def __post_init__(self):
        self.system_prompt = _load_prompt(self.system_prompt_path)

@dataclass
class GeneratorConfig:
    llm: LLMSettings = field(default_factory=LLMSettings)
    max_context_tokens: int = 4096
    provider: str = "openai"          
    extras: Dict[str, Any] = field(default_factory=dict)


###############################################################################
#  RAG GENERATOR                                                               #
###############################################################################

class RAGGenerator:
    """Generador basado en LLM + documentos recuperados."""

    def __init__(self, cfg: Dict[str, Any], **overrides):
        # fusinon YAML + overrides CLI
        gen_cfg_dict = {**cfg.get("generation", {}), **overrides}

        llm_cfg_dict = gen_cfg_dict.pop("llm", {})
        self.cfg = GeneratorConfig(
            llm=LLMSettings(**llm_cfg_dict),
            **{k: v for k, v in gen_cfg_dict.items() if k in {"max_context_tokens", "provider"}},
            extras={k: v for k, v in gen_cfg_dict.items() if k not in {"max_context_tokens", "provider"}}
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self._init_openai()

    # ---------------- PUBLIC API -------------------------------------------
    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        prompt = self._build_prompt(query, retrieved_docs)
        self.logger.debug("Prompt (%d chars) construido.", len(prompt))

        response = self.client.chat.completions.create(
            model=self.cfg.llm.model,
            temperature=self.cfg.llm.temperature,
            top_p=self.cfg.llm.top_p,
            max_tokens=self.cfg.llm.max_tokens,
            messages=[
                {"role": "system", "content": self.cfg.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()


    async def generate_async(self, query: str, retrieved_docs: List[str]) -> str:
        prompt = self._build_prompt(query, retrieved_docs)
        self.logger.debug("Prompt async (%d chars).", len(prompt))

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
            model=self.cfg.llm.model,
            temperature=self.cfg.llm.temperature,
            top_p=self.cfg.llm.top_p,
            max_tokens=self.cfg.llm.max_tokens,
            messages=[
                {"role": "system", "content": self.cfg.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # ---------------- INTERNALS -------------------------------------------
    def _init_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Variable OPENAI_API_KEY no definida. Añádela a tu .env o al entorno."
            )
        self.client = OpenAI(api_key=api_key)
        self.logger.info("API Key OpenAI cargada correctamente.")

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        context = self._truncate_docs(docs)
        joined = "\n\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(context))
        return textwrap.dedent(
            f"""\
            Utiliza exclusivamente la siguiente información para responder.\n\n{joined}\n\n
            Pregunta: {query}\n\nRespuesta:"""
        )

    def _truncate_docs(self, docs: List[str]) -> List[str]:
        max_chars = self.cfg.max_context_tokens * 4   # heuristic ≈ tokens*4
        out, acc = [], 0
        for d in docs:
            if acc + len(d) > max_chars:
                break
            out.append(d)
            acc += len(d)
        return out
