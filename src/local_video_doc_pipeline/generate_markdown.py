from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover
    ollama = None

from .config import PROJECT_ROOT
from .utils import configure_logging, ensure_dir, limpar_memoria

logger = configure_logging(__name__)


def generate_markdown_document(
    transcript_text: str,
    image_descriptions: Dict[str, str],
    output_path: str | Path | None = None,
    model_name: str = "llama3.1",
) -> str:
    """Gera o documento final em Markdown com base na transcrição e nas descrições visuais."""
    if ollama is None:
        raise ImportError("A biblioteca ollama não está instalada. Instale com: pip install ollama")

    markdown_file = Path(output_path) if output_path else PROJECT_ROOT / "documento_especificacao.md"
    ensure_dir(markdown_file.parent)

    image_context = "\n".join(
        f"- {Path(image_path).name}: {description.strip()}" for image_path, description in sorted(image_descriptions.items())
    )

    prompt = f"""
Atue como um Analista de Sistemas e Engenheiro de Software Sênior.

Com base na transcrição do vídeo a seguir:
{transcript_text}

E considerando as telas visuais a seguir:
{image_context}

Crie uma Especificação Técnica detalhada em formato Markdown.
Requisitos:
1. Estruture em seções como Visão Geral, Objetivo, Regras de Negócio, Fluxo de Usuário, Telas/Interface e Observações Técnicas.
2. Explique as regras de negócio extraídas do conteúdo falado.
3. Quando fizer sentido, insira imagens relevantes com a sintaxe Markdown: ![Descrição](caminho/para/imagem.jpg).
4. Use linguagem clara, técnica e em português.
5. Seja específico sobre comportamentos, validações, campos, regras operacionais e decisões de produto.
6. Se a transcrição não for totalmente clara, apresente hipóteses de implementação de forma prudente.

A resposta deve ser somente o conteúdo Markdown final, sem explicações extras.
""".strip()

    try:
        logger.info("Gerando documento final em Markdown com Llama 3.1...")
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        content = response.get("message", {}).get("content", str(response)).strip() if isinstance(response, dict) else str(response)
    except Exception as exc:
        logger.exception("Falha ao gerar o Markdown final com Ollama.")
        raise RuntimeError(f"Erro de geração do documento final: {exc}") from exc

    if not content:
        raise RuntimeError("O modelo retornou conteúdo vazio para o Markdown final.")

    if not content.lstrip().startswith("#"):
        content = "# Especificação Técnica\n\n" + content

    visuals_section = "\n\n## Evidências Visuais\n\n"
    visuals_items = []
    for image_path, description in sorted(image_descriptions.items()):
        filename = Path(image_path).name
        relative_path = os.path.relpath(image_path, start=str(markdown_file.parent))
        visuals_items.append(f"![{description.strip()[:80]}]({relative_path.replace('\\', '/')})")

    if visuals_items:
        content = content.rstrip() + visuals_section + "\n\n".join(visuals_items) + "\n"

    markdown_file.write_text(content, encoding="utf-8")
    logger.info(f"Documento salvo em: {markdown_file}")
    limpar_memoria()
    return content
