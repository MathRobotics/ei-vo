import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
