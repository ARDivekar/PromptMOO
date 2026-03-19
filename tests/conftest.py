"""
Shared pytest configuration and fixtures for PromptMOO tests.

Provides:
- API key loading from .env file
- skip_no_api_key marker for e2e tests
- LLM worker factory fixtures (task_llm, shared_limits) using SlowBurnLLM
- Coroutine leak detection via sys.unraisablehook + gc.collect()
- Pytest configuration (timeout, markers, pythonpath)
"""
import gc
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env at import time (before any fixture runs)
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Register custom markers and set default timeout."""
    config.addinivalue_line("markers", "integration: marks tests requiring real API calls (may cost money)")
    config.addinivalue_line("markers", "unit: marks pure unit tests (no network)")
    config.addinivalue_line("markers", "e2e: end-to-end tests with real LLM calls")

    timeout = config.getoption("--timeout", default=None)
    if timeout is None:
        config._inicache.setdefault("timeout", "120")
        config._inicache.setdefault("timeout_method", "thread")


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------
def _have_api_key() -> bool:
    return len(os.getenv("OPENROUTER_API_KEY", "")) > 0


skip_no_api_key = pytest.mark.skipif(
    not _have_api_key(),
    reason="OPENROUTER_API_KEY not set (env var or .env file)",
)


@pytest.fixture(scope="session")
def api_key():
    """Load OpenRouter API key from .env file. Skips if unavailable."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key is None or len(key) == 0:
        pytest.skip("OPENROUTER_API_KEY not set (needed for integration tests)")
    return key


# ---------------------------------------------------------------------------
# LLM worker factories (SlowBurnLLM-based)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def shared_limits():
    """Create shared LimitSet for LLM workers (fresh per test)."""
    from runner import create_shared_limits
    return create_shared_limits()


@pytest.fixture(scope="function")
def task_llm(api_key, shared_limits):
    """Create a task LLM worker using runner.py's exact config.

    Yields the worker and stops it after the test.
    """
    from runner import create_task_llm
    llm = create_task_llm(llm="llama3.1", api_key=api_key, limits=shared_limits)
    yield llm
    llm.stop()


# ---------------------------------------------------------------------------
# File descriptor counting (retained for potential future use)
# ---------------------------------------------------------------------------
def count_open_fds() -> int:
    """Count open file descriptors for the current process."""
    pid = os.getpid()
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except FileNotFoundError:
        try:
            return len(os.listdir("/dev/fd"))
        except Exception:
            return -1


# ---------------------------------------------------------------------------
# Coroutine leak detection
# ---------------------------------------------------------------------------
_coroutine_leak_events: list = []
_coroutine_leak_lock = threading.Lock()


def _coroutine_leak_hook(unraisable):
    """Intercept unraisable exceptions to detect leaked coroutines."""
    msg = str(unraisable.err_msg or "") + str(unraisable.object or "")
    if "coroutine" in msg and "was never awaited" in msg:
        with _coroutine_leak_lock:
            _coroutine_leak_events.append({
                "exc_type": unraisable.exc_type,
                "message": msg,
                "object": repr(unraisable.object),
            })
    sys.__unraisablehook__(unraisable)


@pytest.fixture(autouse=True)
def detect_coroutine_leaks():
    """Autouse fixture: detects unawaited coroutine leaks after each test."""
    original_hook = sys.unraisablehook
    sys.unraisablehook = _coroutine_leak_hook
    with _coroutine_leak_lock:
        _coroutine_leak_events.clear()

    yield

    gc.collect()
    gc.collect()
    time.sleep(0.1)
    gc.collect()

    with _coroutine_leak_lock:
        leaks = list(_coroutine_leak_events)
        _coroutine_leak_events.clear()

    sys.unraisablehook = original_hook

    if len(leaks) > 0:
        details = "\n".join(f"  - {e['object']}: {e['message']}" for e in leaks)
        pytest.fail(
            f"{len(leaks)} coroutine(s) were never awaited:\n{details}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Cleanup between tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup between tests: gc.collect to release resources."""
    yield
    gc.collect()
