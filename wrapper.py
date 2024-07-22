import os
from functools import wraps
import time


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        timer = os.environ.get("TIMER", "false").lower()
        if timer == "true":
            start = time.time()
            result = func(*args, **kwargs)
            print(f"time spent: {time.time() - start} seconds")
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper
