"""RAG Generator Module.

Implements a retrieval-augmented generation (RAG) pipeline using LLMs and retrieved documents.
"""
from __future__ import annotations
import os
import textwrap
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI, AsyncOpenAI

def _load_prompt(path: str) -> str:
    """Load a text prompt from a file.

    Args:
        path: Path to the prompt file.

    Returns:
        The content of the file as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.getLogger(__name__).warning("Prompt not found: %s", path)
        return ""

@dataclass
class LLMSettings:
    """Configuration for the language model."""
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
    """Configuration for the RAG generator."""
    llm: LLMSettings = field(default_factory=LLMSettings)
    max_context_tokens: int = 4096
    provider: str = "openai"
    extras: Dict[str, Any] = field(default_factory=dict)

class RAGGenerator:
    """Generator based on LLMs and retrieved documents."""

    def __init__(self, config: Dict[str, Any], **overrides):
        """Initialize the RAG generator.

        Args:
            config: Configuration dictionary loaded from YAML or other sources.
            overrides: Additional configuration overrides.
        """
        generator_config = {**config.get("generation", {}), **overrides}

        llm_config = generator_config.pop("llm", {})
        self.config = GeneratorConfig(
            llm=LLMSettings(**llm_config),
            **{k: v for k, v in generator_config.items() if k in {"max_context_tokens", "provider"}},
            extras={k: v for k, v in generator_config.items() if k not in {"max_context_tokens", "provider"}}
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_openai()

    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        """Generate a response using the LLM.

        Args:
            query: User query string.
            retrieved_docs: List of retrieved documents to use as context.

        Returns:
            Generated response string.
        """
        prompt = self._build_prompt(query, retrieved_docs)
        self.logger.debug("Prompt (%d chars) constructed.", len(prompt))

        response = self.client.chat.completions.create(
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
        """Asynchronously generate a response using the LLM.

        Args:
            query: User query string.
            retrieved_docs: List of retrieved documents to use as context.

        Returns:
            Generated response string.
        """
        prompt = self._build_prompt(query, retrieved_docs)
        self.logger.debug("Prompt async (%d chars).", len(prompt))

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.chat.completions.create(
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

    def _initialize_openai(self) -> None:
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "Environment variable OPENAI_API_KEY is not defined. Add it to your .env or environment."
            )
        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI API key loaded successfully.")

    def _build_prompt(self, query: str, docs: List[str]) -> str:
        """Construct the prompt for the LLM.

        Args:
            query: User query string.
            docs: List of retrieved documents to use as context.

        Returns:
            Constructed prompt string.
        """
        context = self._truncate_docs(docs)
        joined = "\n\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(context))
        return textwrap.dedent(
            f"""\
            Use only the following information to respond.\n\n{joined}\n\n
            Question: {query}\n\nAnswer:"""
        )

    def _truncate_docs(self, docs: List[str]) -> List[str]:
        """Truncate documents to fit within the token limit.

        Args:
            docs: List of documents to truncate.

        Returns:
            List of truncated documents.
        """
        max_chars = self.config.max_context_tokens * 4  # heuristic ≈ tokens * 4
        output, accumulated = [], 0
        for doc in docs:
            if accumulated + len(doc) > max_chars:
                break
            output.append(doc)
            accumulated += len(doc)
        return output
