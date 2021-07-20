import time
import doctest
import importlib.util
from pathlib import Path
from tqdm import tqdm


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


if __name__ == "__main__":
    print("[STATUS] Initializing...")
    t_i = time.time()
    path = Path(__file__).parent.resolve()
    test_modules = [
        ("AllData", f"{path}/AllData.py"),
        ("LightCurve", f"{path}/LightCurve.py"),
        ("GP", f"{path}/GP.py"),
        ("GP2D", f"{path}/GP2D.py"),
        ("GP5D", f"{path}/GP5D.py"),
    ]

    for name, path in tqdm(test_modules):
        doctest.testmod(import_module(name, path))  # from document

    print(f"[STATUS] Time Taken: {round(time.time() - t_i)}s")
