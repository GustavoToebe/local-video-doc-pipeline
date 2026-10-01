from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError:  # pragma: no cover
    WhisperModel = None

from .config import AUDIO_DIR
from .utils import configure_logging, ensure_dir, limpar_memoria

logger = configure_logging(__name__)


def extract_audio_from_video(video_path: str | Path, output_audio_path: str | Path) -> str:
    """Extrai o áudio do vídeo e salva em um arquivo WAV local."""
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {input_path}")

    output_path = Path(output_audio_path)
    ensure_dir(output_path.parent)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]

    logger.info("Extraindo áudio do vídeo com ffmpeg...")
    try:
        process = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("O executável ffmpeg não foi encontrado no PATH.") from exc

    if process.returncode != 0:
        error_trace = process.stderr.strip() or process.stdout.strip() or "Erro desconhecido"
        raise RuntimeError(f"Falha ao extrair áudio: {error_trace}")

    return str(output_path)


def transcribe_audio(video_path: str | Path, output_dir: str | Path | None = None) -> Tuple[str, str]:
    """Transcreve o áudio usando faster-whisper e grava a transcrição em texto."""
    if WhisperModel is None:
        raise ImportError(
            "A biblioteca faster-whisper não está instalada. Instale com: pip install faster-whisper"
        )

    output_root = ensure_dir(output_dir or AUDIO_DIR)
    audio_file = output_root / "audio_extraido.wav"
    transcript_file = output_root / "transcricao.txt"

    try:
        audio_path = extract_audio_from_video(video_path, audio_file)
    except Exception:
        logger.exception("Falha na extração do áudio.")
        raise

    device = "cuda" if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    logger.info("Carregando modelo Whisper small para transcrição...")
    model = WhisperModel("small", device=device, compute_type=compute_type)

    try:
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        transcript_lines = []
        for segment in segments:
            text = (segment.text or "").strip()
            if text:
                transcript_lines.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")

        transcript_text = "\n".join(transcript_lines).strip()
        transcript_file.write_text(transcript_text, encoding="utf-8")
        logger.info("Transcrição concluída com sucesso.")
    except Exception:
        logger.exception("Erro durante a transcrição com faster-whisper.")
        raise
    finally:
        del model
        limpar_memoria()

    return transcript_text, str(transcript_file)
