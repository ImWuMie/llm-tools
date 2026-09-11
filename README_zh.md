# LLM Tools

跨平台 LLM 工具链，覆盖 **下载、推理、微调、评估**。

- 依赖管理：[uv](https://docs.astral.sh/uv/)
- 模型来源：Hugging Face 与 ModelScope，支持自动 fallback
- 推理服务：[vLLM](https://docs.vllm.ai/) OpenAI 兼容 API
- 训练：Hugging Face `transformers` / `datasets` / `peft` / `trl` / `accelerate`（LoRA / QLoRA）

最低支持 **Windows 10+** 和 **Linux**。核心逻辑全部在 Python 里；`.sh` / `.ps1` 只是薄包装。

> **Windows 注意：** vLLM 没有完善的原生 Windows 支持。Windows 请通过 **WSL2** 或 **Docker** 启动推理。LoRA 训练可以原生尝试；QLoRA / bitsandbytes 在 Windows 上会自动降级为普通 LoRA。
>
> 可选非官方路径：先安装社区 `vllm-windows` wheel，再执行 `uv run python scripts/start_vllm.py --native`，或设置 `VLLM_WINDOWS_BACKEND=native|auto`。检测命令：`uv run python scripts/install_vllm_windows.py --check`。

[English README](README.md) · [完整用法](USAGES.md)

## 功能

- `--source hf|modelscope|auto` 下载到统一本地目录
- vLLM 只认 `MODEL_DIR`，不关心模型来自 HF 还是 ModelScope
- 前台 / 后台启动、PID、日志、`/v1/models` 健康检查、优雅停止
- 默认 txt 两行一轮，同时支持 jsonl / ShareGPT / Alpaca / custom
- LoRA / QLoRA、断点续训、adapter 合并
- Docker GPU 透传、单元测试、GitHub Actions

## 快速开始

```bash
uv sync --extra download --extra dev
cp .env_example .env   # Windows: Copy-Item .env_example .env
# 编辑 .env 后再执行：
uv run python scripts/download_model.py --source auto --update-env
uv run python scripts/start_vllm.py --daemon          # Linux / WSL
uv run python examples/chat.py --prompt "你好"
```

Windows 推理：

```powershell
uv run python scripts\start_vllm.py --daemon --wsl
# 或
docker compose up vllm
# 可选：非官方原生 wheel
uv run python scripts\install_vllm_windows.py --check
uv run python scripts\start_vllm.py --daemon --native
```

全部命令、参数、环境变量见 [USAGES.md](USAGES.md)。

## 目录

```text
.
├── .env_example              # 复制为 .env（不提交）
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
│   ├── model_sources/        # HF / ModelScope 适配层
│   └── wrappers/             # bash + PowerShell
├── training/
│   ├── config.json
│   ├── system_prompt.txt     # 空文件 = 不插入 system message
│   ├── data/sample.txt
│   └── output/
├── models/base/
├── examples/chat.py
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## 配置分层

| 层级 | 文件 | 用途 |
| --- | --- | --- |
| 运行 / 下载 / 推理 | `.env` | 模型 ID、token、vLLM 地址、路径 |
| 训练超参 | `training/config.json` | LoRA、QLoRA、batch、学习率、保存 |
| 训练系统提示 | `training/system_prompt.txt` | 非空才插入每条样本 |

`HF_TOKEN`、`MODELSCOPE_API_TOKEN`、`VLLM_API_KEY` 只用于认证，日志会脱敏。

## 依赖 extra

| Extra | 安装内容 | 使用场景 |
| --- | --- | --- |
| *(默认)* | dotenv、openai、huggingface_hub、psutil | CLI、配置、调用示例 |
| `download` | `hf_transfer`、`modelscope` | 模型下载 |
| `train` | torch、transformers、peft、trl、accelerate | LoRA / QLoRA |
| `infer` | vLLM（仅 Linux） | 推理服务 |
| `dev` | pytest、ruff | 测试 |

```bash
uv sync --extra download
uv sync --extra download --extra train
uv sync --extra download --extra train --extra infer   # Linux / WSL / Docker
uv sync --extra download --extra dev
```

如果 `modelscope` 与 torch / vLLM 冲突，请把下载放到独立环境，或用 Docker 跑推理和训练。

## 许可证与隐私

- 本仓库代码：[MIT](LICENSE)，只覆盖工具链本身
- 模型许可证以上游模型卡片为准。默认示例 `Qwen/Qwen2.5-7B-Instruct` **不是**本仓库许可证，商用前请自行阅读
- `training/data/sample.txt` 是虚构示例，不是真实业务数据
- 不要把 `.env`、权重、客户对话、个人数据提交进 git；相关路径已忽略

## 测试

```bash
uv run pytest
uv run python scripts/selfcheck.py --offline
```

GitHub Actions 会在 Linux 和 Windows 上跑语法检查、单元测试和离线冒烟。
