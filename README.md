# Pipeline Local de Vídeo para Documentação Técnica

Este projeto implementa um pipeline 100% local em Python para transformar um vídeo em uma especificação técnica em Markdown. A arquitetura foi organizada em quatro etapas:

1. Extrair áudio e transcrever com `faster-whisper`.
2. Detectar cenas e salvar frames relevantes em disco.
3. Descrever as imagens com `moondream` via Ollama.
4. Gerar um documento Markdown final usando `llama3.1` ou `qwen2.5:7b`.

## Requisitos

- Python 3.10+
- ffmpeg instalado e disponível no `PATH`
- CUDA opcional para acelerar `faster-whisper` e `llama` no Ollama
- Ollama instalado e rodando localmente

## Instalação

```bash
cd C:\Users\ICONDESKTOP_02\Documents\local-video-doc-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

> Se você usa Linux/macOS, o comando do ambiente virtual será `source .venv/bin/activate`.

## Uso

```bash
python main.py "C:\caminho\para\video.mp4"
```

Ou com saída customizada:

```bash
python main.py "C:\caminho\para\video.mp4" --output "C:\caminho\para\documento.md"
```

## Estrutura do Projeto

```text
local-video-doc-pipeline/
├── main.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   └── local_video_doc_pipeline/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── utils.py
│       ├── transcribe_audio.py
│       ├── detect_scenes.py
│       ├── describe_frames.py
│       ├── generate_markdown.py
│       └── pipeline.py
├── audio/
├── temp_frames/
├── output/
└── documento_especificacao.md
```

## Observações Importantes

- A memória de vídeo da RTX 3050 tem 4 GB, então o pipeline é sequencial.
- Entre cada etapa, o projeto chama a função `limpar_memoria()`, que executa `gc.collect()` e `torch.cuda.empty_cache()` quando necessário.
- A detecção de cenas roda principalemte em CPU, para reduzir overhead de GPU.
- O Ollama é usado em etapas separadas para não carregar vários modelos de IA na VRAM ao mesmo tempo.

## Modelos esperados localmente

Antes de executar o pipeline, certifique-se de que o Ollama possui os modelos necessários:

```bash
ollama pull moondream
ollama pull llama3.1
```

Se preferir usar `qwen2.5:7b`, ajuste a variável `model_name` na etapa final.
