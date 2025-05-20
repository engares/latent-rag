from __future__ import annotations

import os
import textwrap
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import yaml
import openai
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

def load_prompt(prompt_path: str) -> str:
    """Carga el contenido de un archivo de prompt."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        _logger.warning("Prompt no encontrado en %s; se usará prompt vacío.", prompt_path)
        return ""

def _load_yaml_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Carga la configuración YAML. Si no existe, devuelve un diccionario vacío."""
    try:
        with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        _logger.warning("Config YAML no encontrado en %s; se usarán valores por defecto.", path or DEFAULT_CONFIG_PATH)
        return {}

@dataclass
class LLMSettings:
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: int = 512
    system_prompt_path: str = "./config/prompts/system_prompt.txt"
    system_prompt: str = field(init=False)

    def __post_init__(self):
        self.system_prompt = load_prompt(self.system_prompt_path)

@dataclass
class GeneratorConfig:
    llm: LLMSettings = field(default_factory=LLMSettings)
    max_context_tokens: int = 4096  



class RAGGenerator:
    """Generador de respuestas usando un LLM con contexto recuperado."""

    def __init__(self, config_path: Optional[str] = None, **overrides):
        # Cargar configuración desde YAML
        yaml_cfg = _load_yaml_config(config_path)
        cfg_dict = yaml_cfg.get("generation", {})
        cfg_dict.update(overrides)
        self.config = self._dict_to_config(cfg_dict)

        # Cargar API Key desde .env exclusivamente
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise EnvironmentError(
                "Debe definirse la variable de entorno OPENAI_API_KEY en un archivo .env o en el entorno del sistema."
            )
        _logger.info("API Key cargada correctamente desde .env")

    # API pública

    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        """Genera una respuesta dada la query y los documentos recuperados."""
        prompt = self._build_prompt(query, retrieved_docs)
        _logger.debug("Prompt construido (%d caracteres).", len(prompt))

        response = openai.ChatCompletion.create(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            max_tokens=self.config.llm.max_tokens,
            messages=[
                {"role": "system", "content": self.config.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    async def generate_async(self, query: str, retrieved_docs: List[str]) -> str:
        """Versión asíncrona de `generate` (requiere openai>=1.2 con soporte async)."""
        prompt = self._build_prompt(query, retrieved_docs)
        _logger.debug("Prompt construido (async) (%d caracteres).", len(prompt))

        response = await openai.ChatCompletion.acreate(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            max_tokens=self.config.llm.max_tokens,
            messages=[
                {"role": "system", "content": self.config.llm.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # Métodos internos

    @staticmethod
    def _dict_to_config(cfg: Dict[str, Any]) -> GeneratorConfig:
        """Convierte un diccionario en un objeto GeneratorConfig (profundidad 2)."""
        llm_cfg = cfg.get("llm", {})
        llm_settings = LLMSettings(**llm_cfg)
        return GeneratorConfig(llm=llm_settings, **{k: v for k, v in cfg.items() if k != "llm"})

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        """Construye el prompt combinando la query y los documentos recuperados, recortando si es necesario."""
        context = self._truncate_docs(docs)
        joined_context = "\n\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(context))

        prompt_template = textwrap.dedent(
            f"""\
            Utiliza exclusivamente la siguiente información para responder la pregunta.\n\n"""
            f"{joined_context}\n\n"
            f"Pregunta: {query}\n\nRespuesta:"""
        )
        return prompt_template

    def _truncate_docs(self, docs: List[str]) -> List[str]:
        """Recorta la lista de documentos para no exceder el límite de tokens aproximado."""
        max_chars = self.config.max_context_tokens * 4  
        acc_chars = 0
        selected = []
        for doc in docs:
            if acc_chars + len(doc) > max_chars:
                break
            selected.append(doc)
            acc_chars += len(doc)
        return selected
