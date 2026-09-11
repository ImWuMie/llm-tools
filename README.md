# LLM Tools

Cross-platform toolchain for **downloading**, **serving**, **fine-tuning**, and **evaluating** LLMs.

- Package manager: [uv](https://docs.astral.sh/uv/)
- Download sources: Hugging Face and ModelScope, with automatic fallback
- Serving: [vLLM](https://docs.vllm.ai/) OpenAI-compatible API
- Training: Hugging Face `transformers` / `datasets` / `peft` / `trl` / `accelerate` (LoRA / QLoRA)

Minimum platforms: **Windows 10+** and **Linux**. Core logic is Python; `.sh` / `.ps1` files are thin wrappers.

> **Windows note:** native vLLM support is incomplete. On Windows, start vLLM through **WSL2** or **Docker**. LoRA training can run natively; QLoRA / bitsandbytes is automatically disabled on Windows.
>
> Optional unofficial path: install a community `vllm-windows` wheel, then `uv run python scripts/start_vllm.py --native` or set `VLLM_WINDOWS_BACKEND=native|auto`. Check with `uv run python scripts/install_vllm_windows.py --check`.

[中文文档](README_zh.md) · [Usage reference](USAGES.md)

## Features

- `--source hf|modelscope|auto` downloads into one local directory
- vLLM only needs `MODEL_DIR`; it does not care whether the snapshot came from HF or ModelScope
- Foreground / daemon serving, PID files, logs, `/v1/models` health checks, graceful stop
- Default txt dataset (two lines per turn) plus jsonl / ShareGPT / Alpaca / custom
- LoRA / QLoRA, checkpoint resume, adapter merge
- Docker GPU passthrough, unit tests, GitHub Actions

## Quick start

```bash
uv sync --extra download --extra dev
cp .env_example .env   # Windows: Copy-Item .env_example .env
# edit .env, then:
uv run python scripts/download_model.py --source auto --update-env
uv run python scripts/start_vllm.py --daemon          # Linux / WSL
uv run python examples/chat.py --prompt "Hello"
```

Windows serving:

```powershell
uv run python scripts\start_vllm.py --daemon --wsl
# or
docker compose up vllm
# optional unofficial native wheel:
uv run python scripts\install_vllm_windows.py --check
uv run python scripts\start_vllm.py --daemon --native
```

See [USAGES.md](USAGES.md) for every command, flag, and environment variable.

## Layout

```text
.
├── .env_example              # copy to .env (gitignored)
├── pyproject.toml
├── uv.lock
├── scripts/
│   ├── download_model.py
│   ├── start_vllm.py
│   ├── start_vllm_trained.py
│   ├── stop_vllm.py
│   ├── train.py
│   ├── merge_lora.py
│   ├── eval.py
│   ├── selfcheck.py
│   ├── install_vllm_windows.py
│   ├── model_sources/        # HF / ModelScope adapters
│   └── wrappers/             # bash + PowerShell
├── training/
│   ├── config.json
│   ├── system_prompt.txt     # empty = no system message
│   ├── data/sample.txt
│   └── output/
├── models/base/
├── examples/chat.py
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Configuration

| Layer | File | Purpose |
| --- | --- | --- |
| Runtime / download / serving | `.env` | model ids, tokens, vLLM host/port, paths |
| Training hyperparameters | `training/config.json` | LoRA, QLoRA, batch, lr, save |
| Training system prompt | `training/system_prompt.txt` | inserted only when non-empty |

Tokens (`HF_TOKEN`, `MODELSCOPE_API_TOKEN`, `VLLM_API_KEY`) are never printed in logs.

## Extras

| Extra | What it installs | When to use |
| --- | --- | --- |
| *(default)* | dotenv, openai, huggingface_hub, psutil | CLI, env, client examples |
| `download` | `hf_transfer`, `modelscope` | model download |
| `train` | torch, transformers, peft, trl, accelerate | LoRA / QLoRA |
| `infer` | vLLM (Linux only) | serving |
| `dev` | pytest, ruff | tests |

```bash
uv sync --extra download
uv sync --extra download --extra train
uv sync --extra download --extra train --extra infer   # Linux / WSL / Docker
uv sync --extra download --extra dev
```

If `modelscope` conflicts with torch / vLLM, keep download in a dedicated environment or use Docker for infer/train.

## License and privacy

- Toolchain code: [MIT](LICENSE)
- Model licenses follow the upstream model card. The default example `Qwen/Qwen2.5-7B-Instruct` is **not** this repo's license; review it before commercial use
- `training/data/sample.txt` is a synthetic demo, not a real dataset
- Do not commit `.env`, weights, customer chats, or personal data. Those paths are gitignored

## Tests

```bash
uv run pytest
uv run python scripts/selfcheck.py --offline
```

CI runs syntax checks, unit tests, and offline smoke tests on Linux and Windows.
