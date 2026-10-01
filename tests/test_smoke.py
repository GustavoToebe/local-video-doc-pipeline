import pytest

from local_video_doc_pipeline.config import PROJECT_ROOT
from local_video_doc_pipeline.utils import is_google_drive_url, is_url, is_youtube_url, limpar_memoria, prepare_video_source


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_limpar_memoria_runs():
    limpar_memoria()
    assert True


def test_is_supported_url_helpers():
    assert is_url("https://www.youtube.com/watch?v=abc") is True
    assert is_google_drive_url("https://drive.google.com/file/d/abc/view") is True
    assert is_youtube_url("https://youtu.be/abc") is True


def test_prepare_video_source_rejects_missing_local_file():
    with pytest.raises(FileNotFoundError):
        prepare_video_source("arquivo_que_nao_existe.mp4")
