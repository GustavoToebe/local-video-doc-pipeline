from __future__ import annotations

import argparse

from local_video_doc_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cria uma especificação técnica em Markdown a partir de um vídeo local.")
    parser.add_argument("video_path", help="Caminho do vídeo de entrada.")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Caminho do arquivo Markdown de saída. Se não informado, usa documento_especificacao.md na raiz do projeto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_text = run_pipeline(args.video_path, args.output)
    print("\nPipeline concluído. Documento gerado com sucesso.")
    print(markdown_text[:400] + "..." if len(markdown_text) > 400 else markdown_text)


if __name__ == "__main__":
    main()
