from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover
    ollama = None

from .utils import configure_logging, limpar_memoria

logger = configure_logging(__name__)


def describe_frames(frame_paths: List[str], model_name: str = "moondream") -> Dict[str, str]:
    """Envia cada foto para o Ollama e retorna a descrição técnica da interface."""
    if ollama is None:
        raise ImportError("A biblioteca ollama não está instalada. Instale com: pip install ollama")

    descriptions: Dict[str, str] = {}
    prompt = (
        "Descreva tecnicamente os elementos da interface, arquitetura, telas, fluxos e informações "
        "relevantes observados nesta imagem. Responda em português, em linguagem objetiva e técnica."
    )

    for frame_path in sorted(frame_paths):
        file_path = Path(frame_path)
        if not file_path.exists():
            logger.warning(f"Arquivo não encontrado para descrição: {file_path}")
            continue

        try:
            logger.info(f"Descrevendo imagem: {file_path.name}")
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt, "images": [str(file_path)]}
                ],
                stream=False,
            )
            content = response.get("message", {}).get("content", str(response)).strip() if isinstance(response, dict) else str(response)
            descriptions[str(file_path)] = content
        except Exception as exc:  # pragma: no cover
            logger.exception(f"Falha ao descrever a imagem {file_path.name}.")
            descriptions[str(file_path)] = f"Descrição indisponível: {exc}"

    limpar_memoria()
    return descriptions
