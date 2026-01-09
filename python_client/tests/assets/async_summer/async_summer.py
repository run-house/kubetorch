import asyncio
import time


async def async_summer(a, b, sleep_time=0.5, return_times=False):
    """Async function that sleeps before returning.

    Used to test async concurrency - if multiple calls start before any finish,
    they are running concurrently on the event loop.

    Args:
        a, b: Numbers to add
        sleep_time: How long to sleep (default 0.5s)
        return_times: If True, return (start_time, end_time, result) tuple
    """
    start_time = time.time()
    await asyncio.sleep(sleep_time)
    result = a + b
    end_time = time.time()

    if return_times:
        return {"start": start_time, "end": end_time, "result": result}
    return result
