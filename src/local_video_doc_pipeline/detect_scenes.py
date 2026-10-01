from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:
    from scenedetect import ContentDetector, SceneManager, VideoManager
except ImportError:  # pragma: no cover
    try:
        from scenedetect.video_manager import VideoManager
        from scenedetect.scene_manager import SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:  # pragma: no cover
        ContentDetector = SceneManager = VideoManager = None

from .config import TEMP_FRAMES_DIR
from .utils import configure_logging, ensure_dir, limpar_memoria

logger = configure_logging(__name__)


def detect_scene_frames(video_path: str | Path, output_dir: str | Path | None = None) -> List[str]:
    """Detecta cenas no vídeo e salva frames em disco para análise visual."""
    if cv2 is None:
        raise ImportError("OpenCV não está instalado. Instale com: pip install opencv-python")

    directory = ensure_dir(output_dir or TEMP_FRAMES_DIR)
    saved_paths: List[str] = []

    if VideoManager is not None and SceneManager is not None and ContentDetector is not None:
        try:
            logger.info("Tentando detectar cenas com PySceneDetect...")
            video_manager = VideoManager([str(video_path)])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=30.0))
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scenes = scene_manager.get_scene_list()

            if scenes:
                for idx, scene in enumerate(scenes):
                    frame_number = scene[0].get_frames()  # type: ignore[attr-defined]
                    video_manager.set_current_frame(frame_number)
                    frame = video_manager.get_downscaled_frame()  # type: ignore[attr-defined]
                    if frame is None:
                        continue
                    file_path = directory / f"scene_{idx:04d}.jpg"
                    cv2.imwrite(str(file_path), frame)
                    saved_paths.append(str(file_path))
                logger.info(f"{len(saved_paths)} cenas detectadas com PySceneDetect.")
                return saved_paths
        except Exception as exc:  # pragma: no cover
            logger.warning(f"PySceneDetect falhou: {exc}. Usando fallback em OpenCV.")

    logger.info("Detectando cenas por mudança de histograma em OpenCV...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    previous_hist = None
    last_saved_index = -10
    frame_index = 0
    scene_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if previous_hist is not None:
            diff = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CHISQR)
            if diff > 30 and (frame_index - last_saved_index) > 10:
                file_path = directory / f"scene_{scene_index:04d}.jpg"
                cv2.imwrite(str(file_path), frame)
                saved_paths.append(str(file_path))
                last_saved_index = frame_index
                scene_index += 1

        previous_hist = hist
        frame_index += 1

    cap.release()
    limpar_memoria()

    if not saved_paths:
        logger.warning("Nenhuma cena detectada por comparação de histograma; salvando primeiro frame.")
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        if ok:
            file_path = directory / "scene_0000.jpg"
            cv2.imwrite(str(file_path), frame)
            saved_paths.append(str(file_path))
        cap.release()

    logger.info(f"Frames salvos: {len(saved_paths)}")
    return saved_paths
