import time
from concurrent.futures import ThreadPoolExecutor
import requests

B = "http://127.0.0.1:8000"

def one(i):
    t = time.time()
    r = requests.post(f"{B}/chat",
        json={"message": "你好，请问发货要几天？", "thread_id": f"bench-{i}"}, timeout=120)
    return i, r.status_code, round(time.time() - t, 1)

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, code, cost in ex.map(one, range(1, 5)):
        print(f"  #{i} HTTP {code} 耗时 {cost}s")
print(f">>> 4 个并发总耗时 = {time.time()-t0:.1f}s")