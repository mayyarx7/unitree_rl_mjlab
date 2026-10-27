"""rt/lowstate samples from cpp/lowstate_shm's ring buffer in shared memory.

Layout (cpp/lowstate_shm.cpp): a 32-byte header (magic "R1LOWST1", header size, entry size,
capacity, motors, count of entries written), then capacity entries of 168 bytes (seq = index + 1
once complete, tick in ms, arrival wall time, 31 joint angles, IMU quaternion w x y z).
"""

from __future__ import annotations

import mmap
import os
import struct

import numpy as np

_HEADER = struct.Struct("<8sIIIIQ")
ENTRY = np.dtype([("seq", "<u8"), ("tick", "<u4"), ("reserved", "<u4"), ("wall", "<f8"),
                  ("q", "<f4", (31,)), ("quat", "<f4", (4,)), ("pad", "<f4")])
assert ENTRY.itemsize == 168


class LowstateShmSource:
    def __init__(self, name: str = "/r1_lowstate"):
        self.path = "/dev/shm/" + name.lstrip("/")
        self._mm: mmap.mmap | None = None
        self._inode: int | None = None
        self._next = 0      # index of the next entry to take
        self.lost = 0       # entries overwritten before they were read

    def _open(self) -> bool:
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            self._mm = None
            return False
        if self._mm is not None and st.st_ino == self._inode:
            return True
        with open(self.path, "rb") as f:
            self._mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        self._inode = st.st_ino
        magic, header_size, entry_size, capacity, motors, count = _HEADER.unpack_from(self._mm, 0)
        if magic != b"R1LOWST1" or entry_size != ENTRY.itemsize or motors != 31:
            raise RuntimeError(f"{self.path}: not the layout of cpp/lowstate_shm.cpp")
        self._entries = np.frombuffer(self._mm, ENTRY, capacity, header_size)
        self._capacity = capacity
        self._next = count   # a (re)started writer: begin with what comes next
        return True

    def read_new(self) -> np.ndarray:
        """The entries written since the last call, oldest first (fields of ENTRY); empty if none."""
        if not self._open():
            return np.empty(0, ENTRY)
        count = _HEADER.unpack_from(self._mm, 0)[5]
        if count < self._next:          # writer restarted under the same file
            self._next = count
        start = max(self._next, count - self._capacity + 1)
        self.lost += start - self._next
        idx = np.arange(start, count)
        rows = self._entries[idx % self._capacity].copy()
        rows = rows[rows["seq"] == idx + 1]   # skip entries overwritten while copying
        self._next = count
        return rows
