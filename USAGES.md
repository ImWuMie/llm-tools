# USAGES

Complete command reference for LLM Tools.

完整命令与参数说明。Overview: [README.md](README.md) · [README_zh.md](README_zh.md)

All Python scripts accept `--help`. Paths use `pathlib` and work on Windows and Linux. Logs never print tokens.

所有 Python 脚本都支持 `--help`。路径按 Windows / Linux 兼容方式处理。日志不会打印 token。

---

## 1. Install / 安装

Install uv:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Sync dependencies:

```bash
uv sync                              # core
uv sync --extra download             # Hugging Face + ModelScope
uv sync --extra download --extra train
uv sync --extra download --extra train --extra infer   # Linux / WSL / Docker only
uv sync --extra download --extra dev
```

| Extra | Packages | Use |
| --- | --- | --- |
| default | python-dotenv, openai, huggingface_hub, psutil, requests | CLI, env, OpenAI client |
| `download` | hf_transfer, modelscope | `download_model.py` |
| `train` | torch, transformers, datasets, peft, trl, accelerate, bitsandbytes (Linux) | `train.py`, `merge_lora.py` |
| `infer` | vllm (Linux marker) | `start_vllm.py` |
| `dev` | pytest, ruff | tests |

If `.env` is missing, scripts copy `.env_example` automatically. Edit it before downloading or serving.

若 `.env` 不存在，脚本会从 `.env_example` 复制。下载或启动前请先编辑。

---

## 2. Environment / `.env`

Layering:

| File | Role |
| --- | --- |
| `.env` | download, serving, paths, tokens |
| `training/config.json` | training hyperparameters |
| `training/system_prompt.txt` | system message; empty file = omitted |

CLI flags override `.env`. `.env` is gitignored.

### Source selection

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_SOURCE` | `auto` | `auto` / `hf` / `modelscope` |
| `MODEL_SOURCE_PRIORITY` | `hf,modelscope` | fallback order for `auto` |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | used by both sources if specific IDs are empty |
| `MODEL_OUTPUT_NAME` | `Qwen2.5-7B-Instruct` | local folder name under `DOWNLOAD_DIR` |
| `HF_MODEL_ID` | same as `MODEL_ID` | Hugging Face repo |
| `HF_REVISION` | `main` | HF revision |
| `HF_TOKEN` | empty | gated repos; redacted in logs |
| `HF_ENDPOINT` | `https://huggingface.co` | mirror if needed |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | `1` enables `hf_transfer` |
| `MODELSCOPE_MODEL_ID` | same as `MODEL_ID` | ModelScope id (may differ from HF) |
| `MODELSCOPE_REVISION` | `master` | ModelScope revision |
| `MODELSCOPE_API_TOKEN` | empty | redacted in logs |
| `MODELSCOPE_ENDPOINT` | empty | optional mirror |

### Local paths

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOWNLOAD_DIR` | `./models` | download root |
| `MODEL_DIR` | `./models/base` | vLLM base model path |
| `CHECKPOINT_PATH` | `./training/output` | trainer output |
| `BASE_MODEL` | `./models/base` | LoRA base |
| `ADAPTER_PATH` | `./training/output` | LoRA adapter |
| `TRAINED_MODEL_MODE` | `auto` | `auto` / `lora` / `merged` |
| `TRAINED_LORA_NAME` | `trained` | vLLM LoRA module name |
| `MERGED_MODEL_DIR` | `./training/output/merged` | merge output |
| `LOG_DIR` | `./logs` | server logs |
| `PID_DIR` | `./run` | PID files |

Final snapshot path:

```text
DOWNLOAD_DIR / MODEL_OUTPUT_NAME
# example: ./models/Qwen2.5-7B-Instruct
```

Point `MODEL_DIR` at that folder so serving does not care about the source.

### vLLM

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_HOST` | `0.0.0.0` | bind address |
| `VLLM_PORT` | `8000` | port |
| `VLLM_API_KEY` | `sk-local` | OpenAI-compatible key |
| `VLLM_SERVED_MODEL_NAME` | empty | `--served-model-name` |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU count |
| `GPU_MEMORY_UTILIZATION` | `0.9` | KV cache budget |
| `MAX_MODEL_LEN` | `4096` | context |
| `DTYPE` | `auto` | `auto` / `bf16` / `fp16` / ... |
| `QUANTIZATION` | empty | vLLM quant method, if any |
| `VLLM_TRUST_REMOTE_CODE` | `1` | needed by Qwen |
| `VLLM_MAX_NUM_SEQS` | `16` | concurrent sequences |
| `VLLM_HEALTH_TIMEOUT` | `180` | seconds to wait for `/v1/models` |
| `VLLM_WINDOWS_BACKEND` | `wsl` | `wsl` / `docker` / `native` / `auto` / `fail` |

---

## 3. Download / 下载模型

```bash
uv run python scripts/download_model.py --source hf
uv run python scripts/download_model.py --source modelscope
uv run python scripts/download_model.py --source auto --update-env
uv run python scripts/download_model.py --dry-run
```

```powershell
uv run python scripts\download_model.py --source auto --update-env
```

| Flag | Meaning |
| --- | --- |
| `--source auto\|hf\|modelscope` | override `MODEL_SOURCE` |
| `--model-id` | fallback id for both sources |
| `--hf-model-id` | Hugging Face repo |
| `--modelscope-model-id` | ModelScope id |
| `--revision` | fallback revision |
| `--hf-revision` | default `main` |
| `--modelscope-revision` | default `master` |
| `--output-dir` | download root (`DOWNLOAD_DIR`) |
| `--output-name` | folder name (`MODEL_OUTPUT_NAME`) |
| `--token` | token override for the selected source |
| `--force` | delete and re-download |
| `--dry-run` | print plan, do not download |
| `--update-env` | write `MODEL_DIR` back to `.env` |

Behavior:

1. CLI overrides `.env`.
2. `auto` tries `MODEL_SOURCE_PRIORITY` in order and falls back on failure.
3. If the target already has `config.json` + tokenizer + non-empty weights (and safetensors shards if an index exists), download is skipped unless `--force`.
4. On total failure the process exits non-zero and prints next steps (mirror, token, model id).

`--update-env` only changes `MODEL_DIR`. Tokens in `.env` are left untouched.

---

## 4. Serve base model / 启动基座模型

```bash
uv run python scripts/start_vllm.py --daemon
bash scripts/wrappers/start_vllm.sh --daemon
uv run python scripts/stop_vllm.py
```

```powershell
uv run python scripts\start_vllm.py --daemon --wsl
powershell -File scripts\wrappers\start_vllm.ps1 --daemon --wsl
uv run python scripts\stop_vllm.py
```

| Flag | Meaning |
| --- | --- |
| `--foreground` | run in the current terminal (default if `--daemon` is absent) |
| `--daemon` | background + PID + log |
| `--model-dir` | override `MODEL_DIR` |
| `--host` / `--port` | override bind address |
| `--wsl` | Windows: delegate to WSL2 |
| `--docker` | Windows: `docker compose up vllm` |
| `--native` | Windows: unofficial community wheel in this Python env |
| `--force-native` | alias of `--native`; skip WSL/Docker redirection |
| extra after `--` | forwarded to vLLM |

Startup checks:

1. Validate local model files.
2. Fail if the port is busy.
3. Write `run/vllm.pid` and `logs/vllm.log` in daemon mode.
4. Poll `GET /v1/models` until healthy or `VLLM_HEALTH_TIMEOUT`.

Stop:

```bash
uv run python scripts/stop_vllm.py
uv run python scripts/stop_vllm.py --service vllm
uv run python scripts/stop_vllm.py --service vllm_trained
uv run python scripts/stop_vllm.py --pid-file run/vllm.pid --timeout 20
```

On Windows, the script uses `VLLM_WINDOWS_BACKEND` unless `--wsl` / `--docker` / `--native` is passed.
`auto` means: use native if `import vllm` works, else WSL2, else Docker.

---

## 4.1 Optional native Windows vLLM / 可选原生 Windows 路径

Official vLLM does **not** support native Windows. The optional path is a community wheel
(`vllm-windows`), installed **outside** `uv sync --extra infer` (that extra is Linux-only).

官方 vLLM **不支持**原生 Windows。可选路径是社区 wheel（`vllm-windows`），不要装进
`uv sync --extra infer`（这个 extra 只给 Linux）。

```powershell
uv run python scripts\install_vllm_windows.py --check
# after a matching community wheel is installed in this env:
uv run python scripts\install_vllm_windows.py --check --write-env
uv run python scripts\start_vllm.py --daemon --native
uv run python scripts\start_vllm_trained.py --daemon --native
```

Typical community builds:

- https://github.com/SystemPanic/vllm-windows/releases
- https://github.com/devnen/vllm-windows/releases
- https://github.com/aivrar/vllm-windows-build/releases

Match Python (often 3.12), CUDA, and GPU arch to the wheel. Custom architectures such as
`Spark2_5ForCausalLM` may still fail even after a successful Windows install.

Python / CUDA / GPU 架构必须和 wheel 一致。像 `Spark2_5ForCausalLM` 这种自定义结构，
即便 Windows 包能装上，vLLM 也不一定能加载。

---

## 5. Train / 训练

Default txt format (`training/data/sample.txt`):

```text
# comments and empty lines are skipped by default
user line
assistant line
user line
assistant line
```

`training/system_prompt.txt` is empty by default. If you put text there, it is inserted as a `system` message at the start of every sample.

```bash
uv run python scripts/train.py --data training/data/sample.txt --data-format default --update-env
bash scripts/wrappers/train.sh --data training/data/sample.txt --data-format default
```

```powershell
uv run python scripts\train.py --data training\data\sample.txt --data-format default --update-env
powershell -File scripts\wrappers\train.ps1 --data training\data\sample.txt --data-format default
```

| Flag | Meaning |
| --- | --- |
| `--data` | **required** training file |
| `--data-format` | `default` / `txt` / `jsonl` / `sharegpt` / `alpaca` / `custom` |
| `--config` | default `training/config.json` |
| `--output-dir` | override `output_dir` |
| `--resume-from-checkpoint` | path, or `auto` to pick the latest `checkpoint-*` |
| `--update-env` | write `CHECKPOINT_PATH` / `ADAPTER_PATH` / `TRAINED_MODEL_MODE=lora` |

Converted samples are written to `training/data/processed/<stem>.jsonl` as:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Logs include conversion stats (line count, skipped empty/comments, UTF-8, sample count), loss, learning rate, and GPU memory when CUDA is visible.

On Windows, `quantization.load_in_4bit` falls back to LoRA without bitsandbytes.

Set `"save_merged_model": true` in `training/config.json` to merge after training.

### Data formats

**jsonl / messages**

```json
{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}
```

**ShareGPT**

```json
{"conversations":[{"from":"human","value":"q"},{"from":"gpt","value":"a"}]}
```

**Alpaca**

```json
{"instruction":"翻译","input":"hi","output":"你好"}
```

**custom** keys are configured in `training/config.json`:

```json
"custom_format": {
  "user_key": "user",
  "assistant_key": "assistant",
  "system_key": "system"
}
```

Empty-line / comment handling:

```json
"data": {
  "skip_empty_lines": true,
  "skip_comment_lines": true,
  "comment_prefix": "#",
  "encoding": "utf-8"
}
```

---

## 6. Merge LoRA / 合并 adapter

```bash
uv run python scripts/merge_lora.py --update-env
uv run python scripts/merge_lora.py \
  --base-model ./models/base \
  --adapter-path ./training/output \
  --output-dir ./training/output/merged
```

| Flag | Meaning |
| --- | --- |
| `--base-model` | default `BASE_MODEL` |
| `--adapter-path` | default `ADAPTER_PATH` |
| `--output-dir` | default `MERGED_MODEL_DIR` |
| `--dtype` | `auto` / `bf16` / `fp16` / `fp32` |
| `--update-env` | set `CHECKPOINT_PATH`, `MERGED_MODEL_DIR`, `TRAINED_MODEL_MODE=merged` |

If vLLM cannot enable LoRA, merge first, then serve in `merged` mode.

---

## 7. Serve trained model / 启动训练后模型

```bash
uv run python scripts/start_vllm_trained.py --daemon
uv run python scripts/start_vllm_trained.py --daemon --mode lora
uv run python scripts/start_vllm_trained.py --daemon --mode merged
bash scripts/wrappers/start_vllm_trained.sh --daemon
```

| Flag | Meaning |
| --- | --- |
| `--mode auto\|lora\|merged` | override `TRAINED_MODEL_MODE` |
| `--daemon` / `--foreground` | same as base server |
| `--wsl` / `--docker` / `--native` / `--force-native` | Windows backends |

`auto` detection:

- `adapter_config.json` / `adapter_model.safetensors` → `lora` (needs `BASE_MODEL`)
- full `config.json` + weights without adapter files → `merged`

LoRA serving uses vLLM `--enable-lora --lora-modules trained=<ADAPTER_PATH>`.
Chat requests should use `model=trained` (or `TRAINED_LORA_NAME`).

PID / log files: `run/vllm_trained.pid`, `logs/vllm_trained.log`.

---

## 8. Client examples / 调用示例

Python helper:

```bash
uv run python examples/chat.py --prompt "用一句话介绍 LoRA。"
uv run python examples/chat.py --stream --model trained
uv run python examples/chat.py --system "You are a concise assistant." --prompt "Hi"
```

OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
print(
    client.chat.completions.create(
        model="default",  # or "trained" for LoRA
        messages=[{"role": "user", "content": "你好"}],
    )
    .choices[0]
    .message.content
)
```

curl:

```bash
bash examples/curl_chat.sh default
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{"model":"default","messages":[{"role":"user","content":"你好"}]}'
```

```powershell
powershell -File examples\curl_chat.ps1 default
```

Health check:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-local"
```

---

## 9. Eval and self-check / 评估与自检

```bash
# convert + stats only
uv run python scripts/eval.py --data training/data/sample.txt --data-format default

# query a running server
uv run python scripts/eval.py --remote --max-samples 3 --model trained

uv run python scripts/selfcheck.py --offline
uv run python scripts/selfcheck.py
uv run pytest
uv run python -m compileall scripts tests examples
```

`eval.py --remote` calls `/v1/models` then chat completions, and reports exact-match / length-ratio against the gold assistant turns.

---

## 10. Docker

Requires NVIDIA Container Toolkit.

```bash
docker compose up --build vllm
docker compose --profile train up train
docker compose down
```

Volumes:

- `./models` → `/app/models`
- `./training` → `/app/training`
- `./logs` → `/app/logs`
- `./run` → `/app/run`
- `.env` (read-only)

GPU passthrough: `gpus: all` plus Compose `deploy.resources.reservations.devices`.

Default command serves `scripts/start_vllm.py --foreground`.

---

## 11. Wrappers

| Script | Action |
| --- | --- |
| `scripts/wrappers/start_vllm.sh` / `.ps1` | start base server |
| `scripts/wrappers/start_vllm_trained.sh` / `.ps1` | start trained server |
| `scripts/wrappers/stop_vllm.sh` / `.ps1` | stop servers |
| `scripts/wrappers/train.sh` / `.ps1` | train |

Wrappers `cd` to the repo root, set `PYTHONUTF8=1`, and exec `uv run python ...`.

---

## 12. Troubleshooting / 故障排查

| Symptom | Fix |
| --- | --- |
| missing `.env` | copied from `.env_example`; fill tokens and paths |
| port in use | `uv run python scripts/stop_vllm.py` or change `VLLM_PORT` |
| Windows vLLM | `--wsl`, `--docker`, or run the same uv commands inside WSL2 |
| Windows native vLLM | optional community wheel + `--native` / `VLLM_WINDOWS_BACKEND=native\|auto` |
| CUDA OOM | lower `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` / batch; enable 4-bit on Linux |
| download failed | set `HF_ENDPOINT` / `MODELSCOPE_ENDPOINT`, or add tokens for gated repos |
| validation failed | need `config.json`, tokenizer, non-empty weights; sharded models need index + shards |
| vLLM has no LoRA | `uv run python scripts/merge_lora.py --update-env` then `--mode merged` |
| `modelscope` import conflict | `uv sync --extra download` in a separate env; use Docker for infer/train |
| QLoRA on Windows | bitsandbytes is skipped; training continues as LoRA |

---

## 13. Acceptance commands / 验收命令

Linux:

```bash
uv sync
uv run python scripts/download_model.py --source hf
uv run python scripts/download_model.py --source modelscope
uv run python scripts/download_model.py --source auto
uv run python scripts/start_vllm.py --daemon
uv run python scripts/train.py --data training/data/sample.txt --data-format default
uv run python scripts/start_vllm_trained.py --daemon
```

Windows PowerShell:

```powershell
uv sync
uv run python scripts\download_model.py --source hf
uv run python scripts\download_model.py --source modelscope
uv run python scripts\download_model.py --source auto
uv run python scripts\start_vllm.py --daemon
uv run python scripts\train.py --data training\data\sample.txt --data-format default
uv run python scripts\start_vllm_trained.py --daemon
```

Expected:

- `--source hf` and `--source modelscope` land in the same `DOWNLOAD_DIR/MODEL_OUTPUT_NAME`
- `--source auto` falls back when the first source fails
- both snapshots can be loaded by `start_vllm.py` / `start_vllm_trained.py`
- default txt converts to messages JSONL
- tokens never appear in logs
- Windows vLLM defaults to WSL2 or Docker; `--native` is an unofficial opt-in
