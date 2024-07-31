import os
from functools import wraps
import time

from llm_retrieval_qa.configs import settings


global timer
timer = settings.timer


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # timer = os.environ.get("TIMER", "false").lower()
        if timer:
            start = time.time()
            result = func(*args, **kwargs)
            print(f"time spent: {time.time() - start} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
