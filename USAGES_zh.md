# 用法

LLM Tools 完整命令与参数说明（zh-CN）。

[English usage](USAGES.md) · [中文 README](README_zh.md) · [English README](README.md)

所有 Python 脚本都支持 `--help`。路径使用 `pathlib`，兼容 Windows / Linux。日志不会打印 token。

---

## 1. 安装

安装 uv：

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

同步依赖：

```bash
uv sync                              # 核心依赖
uv sync --extra download             # Hugging Face + ModelScope
uv sync --extra download --extra train
uv sync --extra download --extra train --extra infer   # 仅 Linux / WSL / Docker
uv sync --extra download --extra dev
uv sync --extra infer-hf --extra webui --extra eval
```

| Extra | 包含的包 | 用途 |
| --- | --- | --- |
| 默认 | python-dotenv, openai, huggingface_hub, psutil, requests | CLI、环境、OpenAI 客户端 |
| `download` | hf_transfer, modelscope | `download_model.py` |
| `train` | torch, transformers, datasets, peft, trl, accelerate, bitsandbytes（Linux） | `train.py`、`merge_lora.py` |
| `infer` | vllm（仅 Linux marker） | `start_vllm.py` |
| `infer-hf` | torch, transformers, peft | `start_hf.py` / `--engine hf` |
| `webui` | gradio | `examples/webui.py` |
| `eval` | sacrebleu, rouge-score | 更完整的评测指标 |
| `report` | tensorboard, wandb | 训练 `report_to` |
| `dev` | pytest, ruff | 测试 |

如果缺少 `.env`，脚本会从 `.env_example` 自动复制。下载或启动服务前请先编辑。

---

## 2. 环境变量 / `.env`

配置分层：

| 文件 | 作用 |
| --- | --- |
| `.env` | 下载、推理服务、路径、token |
| `training/config.json` | 训练超参 |
| `training/system_prompt.txt` | 系统提示词；空文件则不插入 |

命令行参数优先于 `.env`。`.env` 已被 gitignore。

### 模型来源

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODEL_SOURCE` | `auto` | `auto` / `hf` / `modelscope` |
| `MODEL_SOURCE_PRIORITY` | `hf,modelscope` | `auto` 模式下的尝试顺序 |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct` | 两个源的共用 ID（专用 ID 为空时使用） |
| `MODEL_OUTPUT_NAME` | `Qwen2.5-7B-Instruct` | `DOWNLOAD_DIR` 下的本地目录名 |
| `HF_MODEL_ID` | 与 `MODEL_ID` 相同 | Hugging Face 仓库 |
| `HF_REVISION` | `main` | HF revision |
| `HF_TOKEN` | 空 | 门禁仓库；日志中会脱敏 |
| `HF_ENDPOINT` | `https://huggingface.co` | 需要时改镜像 |
| `HF_HUB_ENABLE_HF_TRANSFER` | `0` | `1` 启用 `hf_transfer` |
| `MODELSCOPE_MODEL_ID` | 与 `MODEL_ID` 相同 | ModelScope ID（可能与 HF 不同） |
| `MODELSCOPE_REVISION` | `master` | ModelScope revision |
| `MODELSCOPE_API_TOKEN` | 空 | 日志中会脱敏 |
| `MODELSCOPE_ENDPOINT` | 空 | 可选镜像 |

### 本地路径

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DOWNLOAD_DIR` | `./models` | 下载根目录 |
| `MODEL_DIR` | `./models/base` | vLLM 基座模型路径 |
| `CHECKPOINT_PATH` | `./training/output` | 训练输出 |
| `BASE_MODEL` | `./models/base` | LoRA 基座 |
| `ADAPTER_PATH` | `./training/output` | LoRA adapter |
| `TRAINED_MODEL_MODE` | `auto` | `auto` / `lora` / `merged` |
| `TRAINED_LORA_NAME` | `trained` | vLLM LoRA 模块名 |
| `MERGED_MODEL_DIR` | `./training/output/merged` | 合并导出目录 |
| `LOG_DIR` | `./logs` | 服务日志 |
| `PID_DIR` | `./run` | PID 文件 |

最终快照路径：

```text
DOWNLOAD_DIR / MODEL_OUTPUT_NAME
# 例如: ./models/Qwen2.5-7B-Instruct
```

把 `MODEL_DIR` 指到该目录后，推理服务不再关心模型来自 HF 还是 ModelScope。

### vLLM

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `VLLM_HOST` | `0.0.0.0` | 监听地址 |
| `VLLM_PORT` | `8000` | 端口 |
| `VLLM_API_KEY` | `sk-local` | OpenAI 兼容 API Key |
| `VLLM_SERVED_MODEL_NAME` | 空 | `--served-model-name` |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU 数量 |
| `GPU_MEMORY_UTILIZATION` | `0.9` | KV cache 显存占比 |
| `MAX_MODEL_LEN` | `4096` | 上下文长度 |
| `DTYPE` | `auto` | `auto` / `bf16` / `fp16` / ... |
| `QUANTIZATION` | 空 | vLLM 量化方法（如有） |
| `VLLM_TRUST_REMOTE_CODE` | `1` | Qwen 等模型需要 |
| `VLLM_MAX_NUM_SEQS` | `16` | 并发序列数 |
| `VLLM_HEALTH_TIMEOUT` | `600` | 等待 `/v1/models` 的秒数 |
| `VLLM_WINDOWS_BACKEND` | `wsl` | `wsl` / `docker` / `native` / `auto` / `fail` |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | 原生 Windows 默认关闭；仅在 ninja+MSVC 能 JIT 时设为 `1` |

---

## 3. 下载模型

```bash
uv run python scripts/download_model.py --source hf
uv run python scripts/download_model.py --source modelscope
uv run python scripts/download_model.py --source auto --update-env
uv run python scripts/download_model.py --dry-run
```

```powershell
uv run python scripts\download_model.py --source auto --update-env
```

| 参数 | 含义 |
| --- | --- |
| `--source auto\|hf\|modelscope` | 覆盖 `MODEL_SOURCE` |
| `--model-id` | 覆盖两个源的 `MODEL_ID`，优先于 `.env` 里的 `HF_MODEL_ID` / `MODELSCOPE_MODEL_ID` |
| `--hf-model-id` | Hugging Face 仓库 |
| `--modelscope-model-id` | ModelScope ID |
| `--revision` | 回退 revision |
| `--hf-revision` | 默认 `main` |
| `--modelscope-revision` | 默认 `master` |
| `--output-dir` | 下载根目录（`DOWNLOAD_DIR`） |
| `--output-name` | 目录名（`MODEL_OUTPUT_NAME`） |
| `--token` | 覆盖当前源的 token |
| `--force` | 删除后重新下载 |
| `--dry-run` | 只打印计划，不下载 |
| `--update-env` | 把 `MODEL_DIR` 写回 `.env` |
| `--data-source` | `auto` / `local` / `hf` / `modelscope` |
| `--data-split` | hub split，默认 `train` |

行为：

1. 命令行覆盖 `.env`。
2. `auto` 按 `MODEL_SOURCE_PRIORITY` 依次尝试，失败则 fallback。`--source auto` **会覆盖** `.env MODEL_SOURCE`；要用 .env 请不加 `--source`。卡住算失败，AutoDL / 国内网络请用 `--source modelscope` 或 `MODEL_SOURCE_PRIORITY=modelscope,hf`。
3. 目标目录已有 `config.json`、tokenizer、非空权重（若存在 index 则还需分片齐全）时默认跳过，除非加 `--force`。
4. 所有源都失败则非 0 退出，并打印下一步建议（镜像、token、模型 ID）。

`--update-env` 只改 `MODEL_DIR`，不会动 `.env` 里的 token。

---

## 4. 启动基座模型

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

| 参数 | 含义 |
| --- | --- |
| `--foreground` | 前台运行（未指定 `--daemon` 时的默认行为） |
| `--daemon` | 后台运行，并写 PID / 日志 |
| `--model-dir` | 覆盖 `MODEL_DIR` |
| `--host` / `--port` | 覆盖监听地址 |
| `--wsl` | Windows：转到 WSL2 |
| `--docker` | Windows：启动 `docker compose` 服务 `vllm` |
| `--native` | Windows：使用当前 Python 环境里已安装的社区 wheel |
| `--force-native` | `--native` 的别名；跳过 WSL/Docker 重定向 |
| `--` 之后的额外参数 | 转发给 vLLM |

启动检查：

1. 校验本地模型文件。
2. 端口被占用则失败。
3. daemon 模式写入 `run/vllm.pid` 和 `logs/vllm.log`。
4. 轮询 `GET /v1/models`，直到健康或超过 `VLLM_HEALTH_TIMEOUT`。进程若提前退出，启动器会立即失败并打印日志尾部。

停止：

```bash
uv run python scripts/stop_vllm.py
uv run python scripts/stop_vllm.py --service vllm
uv run python scripts/stop_vllm.py --service vllm_trained
uv run python scripts/stop_vllm.py --pid-file run/vllm.pid --timeout 20
```

在 Windows 上，除非传入 `--wsl` / `--docker` / `--native`，否则使用 `VLLM_WINDOWS_BACKEND`。
`auto` 表示：若 `import vllm` 成功则走原生，否则 WSL2，再否则 Docker。

---

## 4.1 可选：原生 Windows vLLM

官方 vLLM **不支持**原生 Windows。可选路径是社区 wheel（vllm-windows），
安装在 uv sync --extra infer **之外**（该 extra 只给 Linux）。

建议用独立的 Python 3.12 环境（例如 .venv-vllm-win），避免非官方 wheel 和 uv sync --extra train 混装。

`powershell
uv run python scripts\install_vllm_windows.py --check
uv run python scripts\install_vllm_windows.py --install --wheel-url <whl-url> --yes --write-env
uv run python scripts\start_vllm.py --daemon --native
uv run python scripts\start_vllm_trained.py --daemon --native
`

--install 会自动加上 PyTorch cu130 extra index（uv 使用 --index-strategy unsafe-best-match）。
原生启动还会：

- 把 venv 的 Scripts（ninja）和 	vm_ffi/lib 加到 PATH / DLL 搜索路径
- 默认 VLLM_USE_FLASHINFER_SAMPLER=0（FlashInfer sampler JIT 需要 ninja+MSVC）
- 社区 xgrammar DLL 失败时打桩，聊天服务仍可启动
- 进程在 /v1/models 就绪前退出则立即失败

常见社区构建：

- https://github.com/SystemPanic/vllm-windows/releases
- https://github.com/devnen/vllm-windows/releases
- https://github.com/aivrar/vllm-windows-build/releases

Python（常见为 3.12）、CUDA、GPU 架构必须与 wheel 一致。像 Spark2_5ForCausalLM
这种自定义结构，即便 Windows 包能装上，仍需 --engine hf。

---

## 4.2 Transformers 回退 / 量化导出 / WebUI

自定义结构（例如 `Spark2_5ForCausalLM`）请走 transformers 服务：

```bash
uv sync --extra infer-hf
uv run python scripts/start_vllm.py --daemon --engine hf
# 或
uv run python scripts/start_hf.py --daemon
uv run python scripts/stop_vllm.py --service hf
```

`INFER_ENGINE=auto` 在本环境能识别模型类型或 Spark plugin 时走 vLLM，否则走 `hf`。plugin 必须装进 `uv run` 的 `.venv`，装在 AutoDL 系统 Python 里不生效。


量化导出（需自行安装转换器）：

```bash
uv run python scripts/export_quant.py --method gguf --dry-run
uv run python scripts/export_quant.py --method awq --update-env
```

WebUI：

```bash
uv sync --extra webui
uv run python examples/webui.py
```

Windows 社区 wheel（不会自动挑选 CUDA 版本）：

```powershell
uv run python scripts\install_vllm_windows.py --check
uv run python scripts\install_vllm_windows.py --install --wheel-url https://example.invalid/vllm.whl --yes --write-env
```

## 5. 训练

默认 txt 格式（`training/data/sample.txt`）：

```text
# 默认跳过注释和空行
user 行
assistant 行
user 行
assistant 行
```

`training/system_prompt.txt` 默认是空文件。如果写入文本，会作为 `system` message 插到每条样本开头。

```bash
uv run python scripts/train.py --data training/data/sample.txt --data-format default --update-env
bash scripts/wrappers/train.sh --data training/data/sample.txt --data-format default
```

```powershell
uv run python scripts\train.py --data training\data\sample.txt --data-format default --update-env
powershell -File scripts\wrappers\train.ps1 --data training\data\sample.txt --data-format default
```

| 参数 | 含义 |
| --- | --- |
| `--data` | **必填**训练文件 |
| `--data-format` | `default` / `txt` / `jsonl` / `sharegpt` / `alpaca` / `custom` |
| `--config` | 默认 `training/config.json` |
| `--output-dir` | 覆盖 `output_dir` |
| `--resume-from-checkpoint` | 路径，或 `auto` 选择最新的 `checkpoint-*` |
| `--update-env` | 写入 `CHECKPOINT_PATH` / `ADAPTER_PATH` / `TRAINED_MODEL_MODE=lora` |

转换后的样本写到 `training/data/processed/<stem>.jsonl`，格式为：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

日志包含转换统计（行数、跳过的空行/注释、UTF-8、样本数）、loss、学习率，以及 CUDA 可见时的显存占用。

在 Windows 上，`quantization.load_in_4bit` 会降级为不使用 bitsandbytes 的普通 LoRA。

在 `training/config.json` 里设置 `"save_merged_model": true` 可在训练后自动合并。

### 数据格式

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

**custom** 的字段名在 `training/config.json` 中配置：

```json
"custom_format": {
  "user_key": "user",
  "assistant_key": "assistant",
  "system_key": "system"
}
```

空行 / 注释处理：

```json
"data": {
  "skip_empty_lines": true,
  "skip_comment_lines": true,
  "comment_prefix": "#",
  "encoding": "utf-8"
}
```

---

## 6. 合并 LoRA

```bash
uv run python scripts/merge_lora.py --update-env
uv run python scripts/merge_lora.py \
  --base-model ./models/base \
  --adapter-path ./training/output \
  --output-dir ./training/output/merged
```

| 参数 | 含义 |
| --- | --- |
| `--base-model` | 默认 `BASE_MODEL` |
| `--adapter-path` | 默认 `ADAPTER_PATH` |
| `--output-dir` | 默认 `MERGED_MODEL_DIR` |
| `--dtype` | `auto` / `bf16` / `fp16` / `fp32` |
| `--update-env` | 设置 `CHECKPOINT_PATH`、`MERGED_MODEL_DIR`、`TRAINED_MODEL_MODE=merged` |

如果当前 vLLM 无法启用 LoRA，请先合并，再以 `merged` 模式启动。

---

## 7. 启动训练后模型

```bash
uv run python scripts/start_vllm_trained.py --daemon
uv run python scripts/start_vllm_trained.py --daemon --mode lora
uv run python scripts/start_vllm_trained.py --daemon --mode merged
bash scripts/wrappers/start_vllm_trained.sh --daemon
```

| 参数 | 含义 |
| --- | --- |
| `--mode auto\|lora\|merged` | 覆盖 `TRAINED_MODEL_MODE` |
| `--daemon` / `--foreground` | 与基座服务相同 |
| `--wsl` / `--docker` / `--native` / `--force-native` | Windows 后端 |

`auto` 检测规则：

- `adapter_config.json` / `adapter_model.safetensors` → `lora`（需要 `BASE_MODEL`）
- 完整 `config.json` + 权重且没有 adapter 文件 → `merged`

LoRA 服务使用 vLLM `--enable-lora --lora-modules trained=<ADAPTER_PATH>`。
聊天请求应使用 `model=trained`（或 `TRAINED_LORA_NAME`）。

PID / 日志：`run/vllm_trained.pid`、`logs/vllm_trained.log`。

---

## 8. 调用示例

Python 辅助脚本：

```bash
uv run python examples/chat.py --prompt "用一句话介绍 LoRA。"
uv run python examples/chat.py --stream --model trained
uv run python examples/chat.py --system "You are a concise assistant." --prompt "你好"
```

OpenAI SDK：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-local")
print(
    client.chat.completions.create(
        model="default",  # LoRA 时用 "trained"
        messages=[{"role": "user", "content": "你好"}],
    )
    .choices[0]
    .message.content
)
```

curl：

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

健康检查：

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-local"
```

---

## 9. 评估与自检

```bash
# 只做转换和统计
uv run python scripts/eval.py --data training/data/sample.txt --data-format default

# 请求正在运行的服务
uv run python scripts/eval.py --remote --max-samples 3 --model trained

uv run python scripts/selfcheck.py --offline
uv run python scripts/selfcheck.py
uv run pytest
uv run python -m compileall scripts tests examples
```

`eval.py --remote` 会先请求 `/v1/models`，再发 chat completions，并对照 gold assistant 轮次报告 exact-match / length-ratio。

---

## 10. Docker

需要 NVIDIA Container Toolkit。

```bash
docker compose up --build vllm
docker compose --profile train up train
docker compose down
```

挂载：

- `./models` → `/app/models`
- `./training` → `/app/training`
- `./logs` → `/app/logs`
- `./run` → `/app/run`
- `.env`（只读）

GPU 透传：`gpus: all`，以及 Compose `deploy.resources.reservations.devices`。

默认命令为 `scripts/start_vllm.py --foreground`。

---

## 11. 包装脚本

| 脚本 | 作用 |
| --- | --- |
| `scripts/wrappers/start_vllm.sh` / `.ps1` | 启动基座服务 |
| `scripts/wrappers/start_vllm_trained.sh` / `.ps1` | 启动训练后服务 |
| `scripts/wrappers/stop_vllm.sh` / `.ps1` | 停止服务 |
| `scripts/wrappers/train.sh` / `.ps1` | 训练 |

包装脚本会先 `cd` 到仓库根目录，设置 `PYTHONUTF8=1`，再执行 `uv run python ...`。

---

## 12. 故障排查

| 现象 | 处理 |
| --- | --- |
| 缺少 `.env` | 已从 `.env_example` 复制；请填写 token 和路径 |
| 端口占用 | `uv run python scripts/stop_vllm.py` 或修改 `VLLM_PORT` |
| Windows vLLM | `--wsl`、`--docker`，或在 WSL2 内运行同样的 uv 命令 |
| Windows 原生 vLLM | 可选社区 wheel + `--native` / `VLLM_WINDOWS_BACKEND=native\|auto` |
| CUDA OOM | 降低 `GPU_MEMORY_UTILIZATION` / `MAX_MODEL_LEN` / batch；Linux 上可开 4-bit |
| 下载失败 | 设置 `HF_ENDPOINT` / `MODELSCOPE_ENDPOINT`，门禁仓库需补 token |
| 校验失败 | 需要 `config.json`、tokenizer、非空权重；分片模型还需要 index 和各分片 |
| vLLM 不支持 LoRA | `uv run python scripts/merge_lora.py --update-env`，然后 `--mode merged` |
| `modelscope` 导入冲突 | 在单独环境执行 `uv sync --extra download`；推理/训练用 Docker |
| Windows 上的 QLoRA | 会跳过 bitsandbytes，继续按普通 LoRA 训练 |

---

## 13. 验收命令

Linux：

```bash
uv sync
uv run python scripts/download_model.py --source hf
uv run python scripts/download_model.py --source modelscope
uv run python scripts/download_model.py --source auto
uv run python scripts/start_vllm.py --daemon
uv run python scripts/train.py --data training/data/sample.txt --data-format default
uv run python scripts/start_vllm_trained.py --daemon
```

Windows PowerShell：

```powershell
uv sync
uv run python scripts\download_model.py --source hf
uv run python scripts\download_model.py --source modelscope
uv run python scripts\download_model.py --source auto
uv run python scripts\start_vllm.py --daemon
uv run python scripts\train.py --data training\data\sample.txt --data-format default
uv run python scripts\start_vllm_trained.py --daemon
```

预期结果：

- `--source hf` 和 `--source modelscope` 都落到同一个 `DOWNLOAD_DIR/MODEL_OUTPUT_NAME`
- `--source auto` 会在前一个源失败时自动 fallback
- 两套快照都能被 `start_vllm.py` / `start_vllm_trained.py` 加载
- 默认 txt 会转换成 messages JSONL
- 日志中不会出现 token
- Windows 上 vLLM 默认走 WSL2 或 Docker；`--native` 是非官方可选路径
