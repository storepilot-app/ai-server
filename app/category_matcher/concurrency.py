from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

from app.category_matcher.config.settings import AI_MAX_CONCURRENT_CPU_TASKS


_cpu_task_slots = BoundedSemaphore(AI_MAX_CONCURRENT_CPU_TASKS)


@contextmanager
def cpu_task_slot() -> Iterator[None]:
    _cpu_task_slots.acquire()
    try:
        yield
    finally:
        _cpu_task_slots.release()
