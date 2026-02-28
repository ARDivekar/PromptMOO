"""
LLM Worker: AsyncIO-based worker for high-throughput LLM calls with structured parsing.

Architecture:
- LLM: AsyncIO worker with async methods for concurrent LLM calls
- Uses litellm.acompletion for async API calls
- Handles rate limiting and token tracking via shared LimitSet
- Supports structured output parsing into Pydantic models with automatic retry on parse failures

Key Features:
1. Async Execution: 10-50x speedup for I/O-bound LLM calls
2. Rate Limiting: Shared limits across multiple LLM instances via LimitSet
3. Structured Parsing: Automatic JSON extraction and validation into Pydantic models
4. Automatic Retries: Parsing failures trigger ValueError for retry_on handling

Usage Example with Structured Parsing:
    from pydantic import BaseModel
    from concurry import RateLimit, LimitSet, CallLimit
    import asyncio

    # Define structured output schema
    class Analysis(BaseModel):
        sentiment: str
        score: float
        summary: str


    # Initialize LLM with retry on parsing failures and transient API errors
    llm = LLM.options(
        retry_on={
            # Retry on parsing failures, timeouts, and transient API errors:
            "call_llm": [ValueError, asyncio.TimeoutError, litellm.APIError, litellm.RateLimitError],
            "*": []
        },
        num_retries={"call_llm": 3, "*": 0},  # Up to 3 retries for parsing
        # Create shared rate limits
        limits=LimitSet(
            limits=[
                # 500 calls/min (Provider Limit)
                CallLimit(window_seconds=60, capacity=500),
                # 10M input tokens/min (Cost Control)
                RateLimit(key="input_tokens", window_seconds=60, capacity=10_000_000),
                # 1M output tokens/min (Cost Control)
                RateLimit(key="output_tokens", window_seconds=60, capacity=1_000_000),
            ],
            mode="asyncio",
            shared=True,
        )
    ).init(
        name="analyzer",
        model_name="gpt-4o",
        api_key="sk-...",
        temperature=0.1,
        max_tokens=512,
        timeout=120.0,
        verbosity=1,
    )

    # Call LLM with structured output (returns parsed Pydantic model)
    result = llm.call_llm(
        prompt="Analyze this review: The movie was great!",
        response_model=Analysis
    ).result()

    print(f"Sentiment: {result.sentiment}, Score: {result.score}")

    llm.stop()
"""

import asyncio
import logging
from typing import Any, List, Optional, Type, TypeVar, Union

import litellm, time, hashlib
import httpx
from concurry.core.synch import async_gather
from concurry import worker
# from instructor.utils import extract_json_from_codeblock
from morphic import Typed
from pydantic import BaseModel, Field

# Suppress litellm's verbose info messages and logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

# Generic type for Pydantic models
T = TypeVar("T", bound=BaseModel)

BATCH_INVOCATION_TIMEOUT : float = 900
SINGLE_INVOCATION_TIMEOUT : float = 60

def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def estimate_tokens(text: str, *, chars_per_token: float = 3.0) -> int:
    """Rough estimate of token count."""
    return int(len(text) // chars_per_token)


def extract_json_from_codeblock(content: str) -> str:
    """
    Extract JSON from a string that may contain extra text.

    The function looks for the first '{' and the last '}' in the string and
    returns the content between them, inclusive. If no braces are found,
    the original string is returned.

    Args:
        content: The string that may contain JSON

    Returns:
        The extracted JSON string
    """

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1:
        json_content = content[first_brace : last_brace + 1]
    else:
        json_content = content  # Return as is if no JSON-like content found

    return json_content


#@worker(mode="asyncio")
@worker
class LLM(Typed):
    """AsyncIO worker for concurrent LLM calls with rate limiting and structured parsing.

    This worker:
    - Uses async methods for 10-50x speedup on I/O-bound LLM calls
    - Supports structured output parsing into Pydantic models with automatic retry on parse failures
    - Uses shared limits (passed via .options(limits=...)) across all LLM instances
    - Acquires rate limits once per batch (not per call)
    - Estimates token usage upfront for batch limit acquisition
    - Updates limits with actual token counts from litellm after completion

    Attributes:
        name: Descriptive name for this LLM (e.g., "task_llm", "optimizer_llm")
        model_name: LLM model identifier (e.g., "meta-llama/llama-3.1-8b-instruct")
        api_key: API key for LLM service
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens per generation (default: 1000)
        timeout: Timeout in seconds for API calls (default: 120.0)

    Structured Parsing:
        Pass a Pydantic model class via response_model parameter to automatically parse LLM output.
        On parsing failure, raises ValueError which triggers automatic retry via retry_on.

        Example with structured parsing:
            from pydantic import BaseModel

            class Analysis(BaseModel):
                sentiment: str
                score: float
                summary: str

            llm = LLM.options(
                mode="asyncio",
                retry_on={
                    # Retry on parsing failures and transient API errors:
                    "call_llm": [ValueError, litellm.APIError, litellm.RateLimitError],
                    "*": []
                },
                num_retries={"call_llm": 3, "*": 0},  # Up to 3 retries for parsing
                limits=limits
            ).init(
                name="analyzer",
                model_name="gpt-4o",
                api_key="sk-...",
                temperature=0.1,
                max_tokens=512
            )

            # Returns parsed Pydantic model (retries automatically on parse failure)
            result = llm.call_llm(
                prompt="Analyze this review: ...",
                response_model=Analysis
            ).result()

    Note:
        Rate limits should be created externally and passed via .options(limits=...)
        to enable sharing limits across multiple LLM instances.

        For initialization with .options(), see concurry Worker Pools documentation.

    Example:
        # Initialize with .options() (recommended for pools and rate limiting)
        from concurry import RateLimit, LimitSet, CallLimit

        limits = LimitSet(
            limits=[
                CallLimit(window_seconds=60, capacity=500),
                RateLimit(key="input_tokens", window_seconds=60, capacity=10_000_000),
                RateLimit(key="output_tokens", window_seconds=60, capacity=1_000_000),
            ],
            shared=True,
            mode="asyncio"
        )

        llm = LLM.options(
            mode="asyncio",
            num_retries=5,
            limits=limits
        ).init(
            name="task_llm",
            model_name="gpt-4o",
            api_key="sk-...",
            temperature=0.1,
            max_tokens=256
        )
    """

    name: str = Field(..., description="LLM name")
    model_name: str = Field(..., description="LLM model identifier")
    api_key: str = Field(..., description="API key for LLM service")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1)
    timeout: float = Field(default=SINGLE_INVOCATION_TIMEOUT, gt=0.0)
    provider_order: Optional[List[str]] = Field(default=None, description="Order of providers to try")

    async def call_llm(
        self,
        *,
        prompt: str,
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        verbosity: int = 1,
    ) -> Union[str, T]:
        """Execute async LLM call with optional structured parsing and rate limiting.

        This unified method handles:
        1. Text queries with optional system prompt
        2. Token usage estimation and rate limit acquisition
        3. API call to LiteLLM with proper message formatting
        4. Structured output parsing into Pydantic models (if requested)
        5. Token usage tracking and limit updates

        Args:
            prompt: Text prompt to send to LLM
            response_model: Optional Pydantic model class for structured output parsing
            system_prompt: Optional system message
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug (with LLM I/O)

        Returns:
            Text response (str) OR parsed Pydantic model (if response_model provided)

        Raises:
            ValueError: If response_model provided and parsing fails (triggers retry)
            Exception: If LLM API call fails
        """
        if not self.api_key:
            raise ValueError("API key not set")

        # === Build Messages ===
        messages = []
        request_id = _prompt_id(prompt)
        start_time = time.time()

        if verbosity >= 2:
            print(f"[{self.name}] | [{request_id}] Starting LLM Call")

        # Add system message if provided
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        # Text-only message format
        messages.append({"role": "user", "content": prompt})

        # === Estimate Token Usage ===
        estimated_text = estimate_tokens(prompt)
        if system_prompt is not None:
            estimated_text += estimate_tokens(system_prompt)

        # Use 5x multiplier + 50 tokens base overhead per message for safety
        # This accounts for: system messages, JSON formatting, role tags, etc.
        estimated_input_tokens = int(estimated_text * 5.0) + 50
        estimated_output_tokens = self.max_tokens

        # === Acquire Rate Limits ===
        requested_usage = {
            "input_tokens": estimated_input_tokens,
            "output_tokens": estimated_output_tokens,
            "call_count": 1,
        }
        if verbosity >= 3:
            print(f"[{self.name}] Requesting resources: {requested_usage}")

        with self.limits.acquire(
            requested=requested_usage,
        ) as acq:
            if verbosity >= 3:
                print(f"[{self.name}] [{request_id}] Acquired limits at {time.time()}")

            try:
                # === Prepare Model Name ===
                model = "openrouter/" + self.model_name

                # === HTTPX ===
                custom_client = httpx.Client(http2=False)

                # === Execute LLM Call ===
                if verbosity >= 3:
                    print(f"[{self.name}] | [{request_id}] Sending request to LLM.")
                litellm.drop_params = True
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=model,
                        messages=messages,
                        api_key=self.api_key,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        extra_body=self._build_extra_body(),
                        num_retries=0,
                        timeout=self.timeout - 1,
                        client=custom_client,
                    ),
                    timeout=self.timeout,
                )

                elapsed_time = time.time() - start_time 
                if verbosity >= 3:
                    print(f"[{self.name}] | [{request_id}] LLM responded in {elapsed_time:.2f}")

                # === Extract Token Usage ===
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                if verbosity >= 3:
                    print(f"\n[{self.name}] Received response from LLM:")
                    print(
                        f"  Input tokens: {input_tokens}\n  Output tokens: {output_tokens}\n"
                    )

                # === Extract Response Text ===
                response_text = response.choices[0].message.content

                # === Parse into Pydantic Model (if requested) ===
                parsed_response = response_text
                if response_model is not None:
                    try:
                        # 1. Extract JSON (handles markdown code fences)
                        json_str = extract_json_from_codeblock(response_text)
                        # 2. Validate against schema
                        parsed_response = response_model.model_validate_json(json_str)

                        if verbosity >= 3:
                            print(
                                f"[{self.name}] Successfully parsed response into {response_model.__name__}"
                            )

                    except asyncio.TimeoutError:
                        elapsed = time.time() - start_time 
                        print(f"[{self.name}] [{request_id}] TIMEOUT after {elapsed:.2f}s")
                        raise

                    except Exception as e:
                        # Parsing failed - raise ValueError to trigger automatic retry
                        if verbosity >= 2:
                            print(
                                f"[{self.name}] Parsing failed for {response_model.__name__}: {e}"
                            )
                            print(
                                f"[{self.name}] Raw response: {response_text[:200]}..."
                            )
                        raise ValueError(
                            f"Failed to parse LLM response into {response_model.__name__}: {e}"
                        ) from e

                # === Update Limits with Actual Usage ===
                # API call succeeded - update with actual token counts
                acq.update(
                    usage={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "call_count": 1,
                    }
                )

                return parsed_response

            except ValueError as e:
                # Parsing failed, but API call succeeded - update with actual usage before re-raising
                if "Failed to parse LLM response" in str(e):
                    acq.update(
                        usage={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "call_count": 1,
                        }
                    )
                raise e
            except Exception as e:
                # API call failed - update with estimated usage only
                acq.update(
                    usage={
                        "input_tokens": estimated_input_tokens,
                        "output_tokens": 0,
                        "call_count": 1,
                    }
                )
                raise e

    async def call_llm_batch(
        self,
        *,
        prompts: List[str],
        response_model: Optional[Type[T]] = None,
        system_prompt: Optional[str] = None,
        verbosity: int = 1,
    ) -> List[Union[str, T]]:
        """Execute multiple LLM calls concurrently.

        Args:
            prompts: List of text prompts
            response_model: Optional Pydantic model class for structured output parsing
            system_prompt: Optional system message (applied to all queries)
            verbosity: 0=silent, 1=default, 2=detailed, 3=debug

        Returns:
            List of text responses (str) or parsed Pydantic models if response_model provided
        """
        if len(prompts) == 0:
            return []

        batch_id = hashlib.sha256(("".join(prompts)).encode()).hexdigest()[:8]
        start_time = time.time()
        
        if verbosity >= 3:
            print(f"\n[{self.name}] [BATCH {batch_id}] Starting batch of {len(prompts)} prompts")

        # Log batch details if verbosity is high
        if verbosity >= 3:
            print(f"\n[{self.name}] Sending {len(prompts)} prompts to LLM")

        # Execute all calls concurrently
        tasks = [
            self.call_llm(
                prompt=prompt,
                response_model=response_model,
                system_prompt=system_prompt,
                verbosity=verbosity,
            )
            for prompt in prompts
        ]

        responses: List[Any] = await async_gather(
            tasks,
            progress=dict(
                disable=verbosity < 2,
                desc=f"{self.model_name}",
                miniters=max(len(prompts) // 2, 1),
                style="std",  # Force standard tqdm to avoid duplicate output in JupyterLab
            ),
        )

        if verbosity >= 3:
            print(f"\n[{self.name}] Received {len(responses)} responses from LLM")

        return responses

    def _build_extra_body(self) -> Optional[dict]:
        """Build extra_body for litellm with OpenRouter provider preferences."""
        if self.provider_order is None:
            return None
        return {"provider": {"order": self.provider_order, "allow_fallbacks": True}}