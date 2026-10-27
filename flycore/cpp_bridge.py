"""ctypes bridge to the C++ LIF kernel (cpp/lif_controller.cpp).

Build the shared library first, e.g. on Linux/Kaggle:
    g++ -O2 -shared -fPIC -o cpp/libflycore_lif.so cpp/lif_controller.cpp
or with CMake (any platform with a compiler):
    cmake -S cpp -B cpp/build && cmake --build cpp/build --config Release

The bridge exposes the same decide() interface as the Python controllers,
so the experiment runner can benchmark Python vs C++ on identical inputs.
"""
from __future__ import annotations
import ctypes
import os
from pathlib import Path

ACTIONS = ("FORWARD", "LEFT", "RIGHT", "SEEK", "BRAKE")

_CPP_DIR = Path(__file__).resolve().parents[1] / "cpp"
_CANDIDATES = (
    _CPP_DIR / "libflycore_lif.so",
    _CPP_DIR / "flycore_lif.dll",
    _CPP_DIR / "build" / "libflycore_lif.so",
    _CPP_DIR / "build" / "Release" / "flycore_lif.dll",
    _CPP_DIR / "build" / "flycore_lif.dll",
    _CPP_DIR / "build" / "libflycore_lif.dylib",
)


def find_library() -> Path | None:
    env_path = os.environ.get("FLYCORE_LIF_LIB")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    for candidate in _CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def is_available() -> bool:
    return find_library() is not None


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.lif_create.restype = ctypes.c_void_p
    lib.lif_destroy.argtypes = [ctypes.c_void_p]
    lib.lif_reset.argtypes = [ctypes.c_void_p]
    lib.lif_step.restype = ctypes.c_int
    lib.lif_step.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
                             ctypes.c_double, ctypes.c_int, ctypes.c_double]
    return lib


class CppLIFController:
    """Drop-in controller backed by the compiled C++ kernel."""

    ACTIONS = ACTIONS
    name = "lif_cpp"

    def __init__(self, lib_path: Path | None = None):
        path = lib_path or find_library()
        if path is None:
            raise FileNotFoundError(
                "flycore_lif shared library not found; build it first "
                "(see flycore/cpp_bridge.py docstring), or set FLYCORE_LIF_LIB.")
        self._lib = _load(Path(path))
        self._handle = self._lib.lif_create()

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.lif_destroy(handle)
            self._handle = None

    def reset(self):
        self._lib.lif_reset(self._handle)

    def decide(self, ranges, vision_result):
        left, front, right = ranges
        found, goal_error, _ = vision_result
        idx = self._lib.lif_step(self._handle, float(left), float(front), float(right),
                                 1 if found else 0, float(goal_error))
        return ACTIONS[idx]
