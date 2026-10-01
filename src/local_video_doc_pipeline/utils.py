import gc
import logging
from pathlib import Path
from typing import Union

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


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


def limpar_memoria() -> None:
    """Libera memória do processo, com atenção especial à VRAM do CUDA."""
    gc.collect()
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
