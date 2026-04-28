#!/Users/rocke_dong/.local/share/uv/tools/terminal-bench/bin/python
"""Wrapper to run terminal-bench with DeepSeek V4 Flash.

Patches LiteLLM's supported-params check so that `response_format` is
treated as unsupported for deepseek-v4-flash (the API rejects json_schema
mode even though LiteLLM's generic deepseek provider list includes it).
"""

from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params as _orig,
)
import litellm.litellm_core_utils.get_supported_openai_params as _mod


def _patched(model, *args, **kwargs):
    params = _orig(model, *args, **kwargs)
    if params and "deepseek-v4" in (model or ""):
        params = [p for p in params if p != "response_format"]
    return params


_mod.get_supported_openai_params = _patched

import terminal_bench.llms.lite_llm as _llm_mod
_llm_mod.get_supported_openai_params = _patched

from terminal_bench.cli.tb.main import app
app()
