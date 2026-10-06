"""The native arena owns rules and observation memory. Python owns sampling and optimization."""

import ctypes
from pathlib import Path

import numpy as np


def library(path=None):
    if path is None:
        root = Path(__file__).resolve().parents[3]
        directory = root / "build" / "nebula"
        path = next((directory / name for name in ("arena.dll", "libarena.dll", "libarena.so", "libarena.dylib")
                     if (directory / name).is_file()), None)
    if path is None:
        raise FileNotFoundError("Build the arena target first, or pass --library PATH")
    native = ctypes.CDLL(str(Path(path).resolve()))
    handle, integer = ctypes.c_void_p, ctypes.c_int
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    bytesArray = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    native.arena_error.argtypes = []
    native.arena_error.restype = ctypes.c_char_p
    native.arena_create.argtypes = [integer, integer, ctypes.c_uint32]
    native.arena_create.restype = handle
    native.arena_destroy.argtypes = [handle]
    native.arena_destroy.restype = None
    native.arena_reset.argtypes = [handle, integer, integer, integer]
    native.arena_read.argtypes = [handle, floats, bytesArray, floats]
    native.arena_step.argtypes = [handle, np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"), floats, bytesArray, bytesArray]
    native.encoder_create.argtypes = [integer]
    native.encoder_create.restype = handle
    native.encoder_destroy.argtypes = [handle]
    native.encoder_destroy.restype = None
    native.encoder_update.argtypes = [handle, integer, integer, integer,
                                     np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS"), floats, bytesArray, floats]
    return native


def check(native, status):
    if status < 0:
        raise RuntimeError(native.arena_error().decode("utf-8", errors="replace"))


def buffers(count, side):
    return (np.empty((count, 38, side, side), dtype=np.float32),
            np.empty((count, 9, side, side), dtype=np.uint8),
            np.empty((count, 2, 512), dtype=np.float32))


class Arena:
    def __init__(self, count, side, seed, path=None):
        self.native = library(path)
        self.handle = self.native.arena_create(count, side, seed)
        if not self.handle:
            check(self.native, -1)
        self.count, self.side = count, side
        self.input = buffers(2 * count, side)
        self.ready = False

    def reset(self, distance, horizon):
        check(self.native, self.native.arena_reset(self.handle, *distance, horizon))
        self.ready = True
        return self.read()

    def read(self):
        if not self.ready:
            raise RuntimeError("Reset the arena before reading observations")
        check(self.native, self.native.arena_read(self.handle, *self.input))
        return self.input

    def step(self, actions):
        actions = np.ascontiguousarray(actions, dtype=np.int32)
        if actions.shape != (2 * self.count,) or np.any(actions < 0) or np.any(actions >= 9 * self.side ** 2):
            raise ValueError("Expected one action index per player")
        rewards = np.empty(2 * self.count, dtype=np.float32)
        terminal = np.empty(self.count, dtype=np.uint8)
        truncated = np.empty(self.count, dtype=np.uint8)
        check(self.native, self.native.arena_step(self.handle, actions, rewards, terminal, truncated))
        return rewards, np.repeat(terminal, 2), np.repeat(truncated, 2)

    def close(self):
        if self.handle:
            self.native.arena_destroy(self.handle)
            self.handle = None
            self.ready = False

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class Encoder:
    def __init__(self, side, rows, cols, player, path=None):
        if not 1 <= rows <= side or not 1 <= cols <= side or player not in (0, 1):
            raise ValueError("Invalid strategy initialization or model input size")
        self.native = library(path)
        self.handle = self.native.encoder_create(side)
        if not self.handle:
            check(self.native, -1)
        self.rows, self.cols, self.player = rows, cols, player
        self.input = buffers(1, side)

    def update(self, values):
        values = np.ascontiguousarray(values, dtype=np.int64)
        if values.shape != (5 + 3 * self.rows * self.cols,) or np.any(values < 0):
            raise ValueError("Invalid observation length or negative value")
        check(self.native, self.native.encoder_update(self.handle, self.rows, self.cols, self.player, values, *self.input))
        return self.input

    def close(self):
        if self.handle:
            self.native.encoder_destroy(self.handle)
            self.handle = None
