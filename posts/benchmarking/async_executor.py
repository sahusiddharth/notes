# custom_async_executor.py
from __future__ import annotations
import asyncio
import time
import logging
from typing import Callable, Any, List, Tuple
from dataclasses import dataclass, field
import nest_asyncio
from tqdm import tqdm

# Apply nest_asyncio to allow nested event loops (e.g., in Jupyter)
nest_asyncio.apply()

logger = logging.getLogger(__name__)


def is_event_loop_running() -> bool:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    else:
        return loop.is_running()


class RateLimiter:
    """
    An asynchronous rate limiter that enforces a minimum interval between calls.
    For example, with max_calls_per_minute=1250, it ensures that calls are spaced by ~0.048 seconds.
    """

    def __init__(self, max_calls_per_minute: int):
        self.interval = 60.0 / max_calls_per_minute
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            wait_time = self.interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_call = time.monotonic()


@dataclass
class AsyncExecutor:
    """
    An asynchronous executor similar in usage to the one in the evaluate function.

    Attributes:
        desc: Description for the progress bar.
        show_progress: Whether to display a progress bar.
        raise_exceptions: Whether to propagate exceptions.
        max_calls_per_minute: API rate limit to enforce.
    """

    desc: str = "Evaluating"
    show_progress: bool = True
    raise_exceptions: bool = False
    max_calls_per_minute: int = 1250
    jobs: List[Tuple[Callable[..., Any], tuple, dict, int]] = field(
        default_factory=list, repr=False
    )
    job_counter: int = 0
    rate_limiter: RateLimiter = field(init=False)

    def __post_init__(self):
        self.rate_limiter = RateLimiter(self.max_calls_per_minute)

    def wrap_callable_with_index(
        self, func: Callable[..., Any], index: int
    ) -> Callable[..., Any]:
        """
        Wraps an asynchronous callable so that it enforces the rate limit,
        and if an error occurs, it waits for an increasing delay (fallback)
        before retrying the function call indefinitely.
        """
        async def wrapped(*args, **kwargs) -> Tuple[int, Any]:
            retry_delay = 10  # initial delay in seconds
            while True:
                try:
                    # Enforce the API rate limit before executing the function
                    await self.rate_limiter.acquire()
                    result = await func(*args, **kwargs)
                    return index, result
                except Exception as e:
                    if self.raise_exceptions:
                        raise e
                    else:
                        logger.error(
                            "Error in job %d: %s. Retrying in %d seconds...",
                            index, e, retry_delay
                        )
                        # Wait asynchronously before retrying
                        await asyncio.sleep(retry_delay)
                        retry_delay += 5  # Increase delay for subsequent retries
        return wrapped

    def submit(self, func: Callable[..., Any], *args, **kwargs):
        """
        Submit an asynchronous job to the executor.
        """
        wrapped_func = self.wrap_callable_with_index(func, self.job_counter)
        self.jobs.append((wrapped_func, args, kwargs, self.job_counter))
        self.job_counter += 1

    async def _run_jobs(self) -> List[Any]:
        tasks = []
        # Create asyncio tasks for each job
        for wrapped_func, args, kwargs, index in self.jobs:
            tasks.append(asyncio.create_task(wrapped_func(*args, **kwargs)))

        results = [None] * len(tasks)
        if self.show_progress:
            pbar = tqdm(total=len(tasks), desc=self.desc)
            for completed in asyncio.as_completed(tasks):
                index, result = await completed
                results[index] = result
                pbar.update(1)
            pbar.close()
        else:
            for completed in asyncio.as_completed(tasks):
                index, result = await completed
                results[index] = result
        return results

    def results(self) -> List[Any]:
        """
        Execute all submitted asynchronous jobs and return their results
        in the order they were submitted.

        Thanks to nest_asyncio, this method can be used inside a Jupyter Notebook.
        """
        # If an event loop is already running, nest_asyncio allows asyncio.run() to work.
        return asyncio.run(self._run_jobs())
