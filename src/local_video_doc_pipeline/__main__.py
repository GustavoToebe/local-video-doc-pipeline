from .pipeline import run_pipeline


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Executa o pipeline de vídeo para documentação técnica.")
    parser.add_argument("video_path", help="Caminho do vídeo de entrada")
    parser.add_argument("--output", "-o", default=None, help="Caminho do Markdown de saída")
    args = parser.parse_args()

    run_pipeline(args.video_path, args.output)
