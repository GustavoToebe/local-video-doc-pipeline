from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import OUTPUT_DIR, PROJECT_ROOT, TEMP_FRAMES_DIR
from .describe_frames import describe_frames
from .detect_scenes import detect_scene_frames
from .generate_markdown import generate_markdown_document
from .transcribe_audio import transcribe_audio
from .utils import configure_logging, ensure_dir, limpar_memoria, prepare_video_source, validate_video_file

logger = configure_logging(__name__)


def run_pipeline(video_path: str | Path, output_markdown_path: str | Path | None = None) -> str:
    """Executa o pipeline completo: validação do input -> transcrição -> cenas -> visão -> Markdown final."""
    try:
        source_path = prepare_video_source(str(video_path), temp_dir=PROJECT_ROOT / "temp_downloads")
        video_file = validate_video_file(source_path)
        logger.info(f"Arquivo de vídeo validado: {video_file}")
    except Exception:
        logger.exception("Falha na preparação e validação do arquivo de vídeo.")
        raise

    if output_markdown_path is None:
        output_markdown_path = PROJECT_ROOT / "documento_especificacao.md"

    ensure_dir(TEMP_FRAMES_DIR)
    ensure_dir(OUTPUT_DIR)

    logger.info("=== ETAPA 1: Transcrição de áudio ===")
    try:
        transcript_text, _ = transcribe_audio(video_file, PROJECT_ROOT / "audio")
    except Exception:
        logger.exception("Erro na etapa de transcrição.")
        raise
    limpar_memoria()

    logger.info("=== ETAPA 2: Captura de frames por cena ===")
    try:
        frame_paths = detect_scene_frames(video_file, TEMP_FRAMES_DIR)
    except Exception:
        logger.exception("Erro na etapa de detecção de cenas.")
        raise
    limpar_memoria()

    logger.info("=== ETAPA 3: Descrição visual das imagens ===")
    try:
        image_descriptions: Dict[str, str] = describe_frames(frame_paths, model_name="moondream")
    except Exception:
        logger.exception("Erro na etapa de descrição visual.")
        raise
    limpar_memoria()

    logger.info("=== ETAPA 4: Geração do documento final ===")
    try:
        markdown_text = generate_markdown_document(
            transcript_text=transcript_text,
            image_descriptions=image_descriptions,
            output_path=output_markdown_path,
            model_name="llama3.1",
        )
    except Exception:
        logger.exception("Erro na geração do documento final em Markdown.")
        raise
    limpar_memoria()

    logger.info("Pipeline concluído com sucesso.")
    return markdown_text
