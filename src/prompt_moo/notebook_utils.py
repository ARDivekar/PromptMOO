"""Utilities for working with PromptMOO in Jupyter notebooks.

This module provides helper functions for dealing with module reloading
and Registry cleanup in interactive environments like Jupyter notebooks.
"""

import importlib
import sys


def reload_prompt_moo_modules():
    """Reload all prompt_moo modules to clear Registry state.

    This is useful in Jupyter notebooks when you've modified code
    and want to reload it without restarting the kernel.

    Usage:
        from prompt_moo.notebook_utils import reload_prompt_moo_modules
        reload_prompt_moo_modules()
    """
    # List of prompt_moo modules in dependency order
    modules_to_reload = [
        "prompt_moo.data_structures",
        "prompt_moo.tasks_utils",
        "prompt_moo.llm_workers",
        "prompt_moo.prompt_template_utils",
        "prompt_moo.data_input",
        "prompt_moo.task_predictor",
        "prompt_moo.loss_computer",
        "prompt_moo.gradient_computer",
        "prompt_moo.prompt_optimizer",
        "prompt_moo.observability",
        "prompt_moo.prompt_trajectory",
        "prompt_moo.algorithm",
    ]

    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            # print(f"✓ Reloaded {module_name}")


def clear_worker_pools(*pools):
    """Stop and cleanup LLM instances.

    Args:
        *pools: Variable number of LLM instances to stop

    Usage:
        task_llm = LLM.options(mode="ray").init(...)
        optimizer_llm = LLM.options(mode="ray").init(...)
        # ... use LLMs ...
        clear_worker_pools(task_llm, optimizer_llm)
    """
    for pool in pools:
        try:
            pool.stop()
            print(f"✓ Stopped LLM: {pool.name}")
        except Exception as e:
            # pool.name should always exist for LLM instances
            pool_name = pool.name if pool is not None else 'unknown'
            print(f"⚠ Error stopping LLM {pool_name}: {e}")
