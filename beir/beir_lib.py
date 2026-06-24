"""Load the pip-installed BEIR library without conflicting with this package."""

import importlib
import site
import sys
from contextlib import contextmanager
from pathlib import Path

from . import PROJECT_ROOT


def _pip_beir_root():
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        if not base:
            continue
        util_path = Path(base) / "beir" / "util.py"
        if util_path.exists():
            return Path(base) / "beir"
    return None


def require_beir_library():
    if _pip_beir_root() is None:
        raise RuntimeError(
            "The 'beir' package is not installed. Install it with:\n"
            "  pip install beir"
        )


def _swap_project_beir():
    saved = {}
    for key in list(sys.modules):
        if key == "beir" or key.startswith("beir."):
            saved[key] = sys.modules.pop(key)
    return saved


def _restore_project_beir(saved):
    sys.modules.update(saved)


@contextmanager
def pip_beir_import_context():
    saved_modules = _swap_project_beir()
    root = str(PROJECT_ROOT.resolve())
    saved_path = sys.path.copy()
    sys.path = [p for p in sys.path if str(Path(p).resolve()) != root]
    try:
        yield
    finally:
        sys.path = saved_path
        _restore_project_beir(saved_modules)


def load_beir_util():
    require_beir_library()
    with pip_beir_import_context():
        try:
            return importlib.import_module("beir.util")
        except ImportError as exc:
            raise RuntimeError(
                "The 'beir' package is not installed. Install it with:\n"
                "  pip install beir"
            ) from exc


def load_generic_data_loader():
    require_beir_library()
    with pip_beir_import_context():
        try:
            module = importlib.import_module("beir.datasets.data_loader")
            return module.GenericDataLoader
        except ImportError as exc:
            raise RuntimeError(
                "The 'beir' package is not installed. Install it with:\n"
                "  pip install beir"
            ) from exc
