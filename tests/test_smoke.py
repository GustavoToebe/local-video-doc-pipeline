from pathlib import Path

from local_video_doc_pipeline.config import PROJECT_ROOT
from local_video_doc_pipeline.utils import limpar_memoria


def test_project_root_exists():
    assert PROJECT_ROOT.exists()


def test_limpar_memoria_runs():
    limpar_memoria()
    assert True
