from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import OUTPUT_DIR, PROJECT_ROOT, TEMP_FRAMES_DIR
from .describe_frames import describe_frames
from .detect_scenes import detect_scene_frames
from .generate_markdown import generate_markdown_document
from .transcribe_audio import transcribe_audio
from .utils import configure_logging, ensure_dir, limpar_memoria

logger = configure_logging(__name__)


def run_pipeline(video_path: str | Path, output_markdown_path: str | Path | None = None) -> str:
    """Executa o pipeline completo: transcrição -> cenas -> visão -> Markdown final."""
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_file}")

    if output_markdown_path is None:
        output_markdown_path = PROJECT_ROOT / "documento_especificacao.md"

    ensure_dir(TEMP_FRAMES_DIR)
    ensure_dir(OUTPUT_DIR)

    logger.info("=== ETAPA 1: Transcrição de áudio ===")
    transcript_text, _ = transcribe_audio(video_file, PROJECT_ROOT / "audio")
    limpar_memoria()

    logger.info("=== ETAPA 2: Captura de frames por cena ===")
    frame_paths = detect_scene_frames(video_file, TEMP_FRAMES_DIR)
    limpar_memoria()

    logger.info("=== ETAPA 3: Descrição visual das imagens ===")
    image_descriptions: Dict[str, str] = describe_frames(frame_paths, model_name="moondream")
    limpar_memoria()

    logger.info("=== ETAPA 4: Geração do documento final ===")
    markdown_text = generate_markdown_document(
        transcript_text=transcript_text,
        image_descriptions=image_descriptions,
        output_path=output_markdown_path,
        model_name="llama3.1",
    )
    limpar_memoria()

    logger.info("Pipeline concluído com sucesso.")
    return markdown_text
