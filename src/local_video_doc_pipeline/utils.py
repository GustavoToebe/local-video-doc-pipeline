import gc
import logging
import re
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    import gdown
except ModuleNotFoundError:  # pragma: no cover
    gdown = None

try:
    import yt_dlp
except ModuleNotFoundError:  # pragma: no cover
    yt_dlp = None


PathLike = Union[str, Path]


def configure_logging(name: str = "video_doc_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def ensure_dir(path: PathLike) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_google_drive_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    return "drive.google.com" in host or "docs.google.com" in host or "googleusercontent.com" in host


def is_youtube_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    return host in {"youtu.be", "www.youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"}


def download_google_drive_video(url: str, destination_dir: PathLike) -> str:
    if gdown is None:
        raise ImportError("A biblioteca gdown não está instalada. Instale com: pip install gdown")

    destination = ensure_dir(destination_dir)
    file_path = destination / "video_from_drive.mp4"
    gdown.download(url, str(file_path), fuzzy=True, quiet=False)
    if not file_path.exists():
        raise FileNotFoundError(f"Não foi possível baixar o vídeo do Google Drive: {url}")
    return str(file_path)


def download_youtube_video(url: str, destination_dir: PathLike) -> str:
    if yt_dlp is None:
        raise ImportError("A biblioteca yt-dlp não está instalada. Instale com: pip install yt-dlp")

    destination = ensure_dir(destination_dir)
    output_template = str(destination / "youtube_video.%(ext)s")
    options = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    if info is None:
        raise RuntimeError(f"Não foi possível extrair informações do vídeo: {url}")

    downloaded_files = list(destination.glob("youtube_video.*"))
    if not downloaded_files:
        raise FileNotFoundError(f"Vídeo do YouTube não foi baixado: {url}")
    downloaded_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(downloaded_files[0])


def prepare_video_source(source: str | Path, temp_dir: PathLike | None = None) -> str:
    """Aceita caminho local ou URL e retorna um arquivo local válido para processamento."""
    source_value = str(source).strip()
    if not source_value:
        raise ValueError("O caminho do vídeo não pode estar vazio.")

    if is_url(source_value):
        working_dir = ensure_dir(temp_dir or Path("temp_downloads"))
        if is_google_drive_url(source_value):
            logger = configure_logging("video_doc_pipeline.urls")
            logger.info("URL do Google Drive detectada. Iniciando download...")
            return download_google_drive_video(source_value, working_dir)
        if is_youtube_url(source_value):
            logger = configure_logging("video_doc_pipeline.urls")
            logger.info("URL do YouTube detectada. Iniciando download...")
            return download_youtube_video(source_value, working_dir)
        raise ValueError("A URL informada não é suportada. Use YouTube ou Google Drive.")

    local_file = Path(source_value)
    if not local_file.exists():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {local_file}")
    if not local_file.is_file():
        raise ValueError(f"O caminho informado não é um arquivo válido: {local_file}")
    return str(local_file)


def validate_video_file(video_path: str | Path) -> Path:
    file_path = Path(video_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de vídeo inexistente: {file_path}")
    if file_path.stat().st_size <= 0:
        raise ValueError(f"Arquivo de vídeo vazio: {file_path}")

    allowed_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if file_path.suffix.lower() not in allowed_exts:
        raise ValueError(
            f"Extensão de vídeo não suportada: {file_path.suffix}. Use uma das extensões: {sorted(allowed_exts)}"
        )

    try:
        import cv2

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"O arquivo não pôde ser aberto como vídeo: {file_path}")
        ok, _ = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"O arquivo não contém frames válidos para processamento: {file_path}")
    except ModuleNotFoundError as exc:
        raise ImportError("OpenCV não está disponível para validar o arquivo de vídeo.") from exc

    return file_path


def limpar_memoria() -> None:
    """Libera memória do processo, com atenção especial à VRAM do CUDA."""
    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
