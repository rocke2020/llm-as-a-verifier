# Terminal-Bench hello-world with DeepSeek V4 Flash

**Date:** 2026-04-28  
**Task:** hello-world  
**Dataset:** terminal-bench-core==0.1.1  
**Agent:** terminus  
**Model:** deepseek/deepseek-v4-flash  
**Result:** PASSED (100% accuracy)

## Run Details

- **Run ID:** 2026-04-28__10-06-40
- **Total input tokens:** 866
- **Total output tokens:** 140
- **Agent runtime:** ~13s
- **Total trial time:** ~37s
- **Tests passed:** test_hello_file_exists, test_hello_file_content

## Issues Encountered and Fixes

### 1. Dataset `head` version has broken `dataset_path` (FileNotFoundError)

**Error:** `FileNotFoundError: No such file or directory: '.../tmplp1f8rxy/tasks'`

**Root cause:** The registry entry for `terminal-bench-core==head` specifies `dataset_path: "./tasks"` on the `main` branch, but the `tasks/` directory was renamed to `original-tasks/` in the repo. The registry was not updated.

**Fix:** Use the stable `terminal-bench-core==0.1.1` dataset instead of `head`. The v0.1.x branch still has the correct `tasks/` directory.

### 2. SOCKS proxy causes APIConnectionError

**Error:** `RetryError[... raised APIConnectionError]`

**Root cause:** The shell has `all_proxy=socks5://127.0.0.1:7890` set. LiteLLM uses httpx which requires the `socksio` package for SOCKS proxy support, but it was not installed.

**Fix:** Install `socksio` into the terminal-bench environment:
```bash
uv pip install --python /path/to/terminal-bench/bin/python "httpx[socks]"
```

### 3. DeepSeek V4 Flash rejects `response_format` with JSON schema (BadRequestError)

**Error:** `BadRequestError: DeepseekException - {"error":{"message":"This response_format type is unavailable now",...}}`

**Root cause:** The terminus agent sends `response_format=CommandBatchResponse` (a Pydantic model) to LiteLLM, which converts it to `json_schema` mode. DeepSeek V4 Flash does not support `json_schema` response format — only `json_object`. However, LiteLLM's generic deepseek provider param list incorrectly includes `response_format` as supported, so the fallback (embedding the schema in the prompt text) never triggers.

**Fix:** Created `scripts/run_tb.py` — a wrapper that monkey-patches `get_supported_openai_params` to remove `response_format` from the supported params list for deepseek-v4 models. This forces terminal-bench to use its text-based JSON schema fallback instead.

## How to Reproduce the Passing Run

```bash
# One-time setup: install socksio for SOCKS proxy support
uv pip install --python ~/.local/share/uv/tools/terminal-bench/bin/python "httpx[socks]"

# Run with the patched wrapper (shebang points to terminal-bench's Python)
./scripts/run_tb.py run \
    --dataset terminal-bench-core==0.1.1 \
    --agent terminus \
    --model deepseek/deepseek-v4-flash \
    --task-id hello-world
```