import mmap
import os
import struct
import uuid

import numpy as np
import pytest

from r1_perception.lowstate_source import ENTRY, LowstateShmSource

CAPACITY = 8


class Writer:
    """Writes as cpp/lowstate_shm.cpp does."""

    def __init__(self, name):
        self.path = "/dev/shm/" + name.lstrip("/")
        size = 32 + CAPACITY * ENTRY.itemsize
        with open(self.path, "wb") as f:
            f.write(b"\0" * size)
        self.f = open(self.path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), size)
        struct.pack_into("<8sIIIIQ", self.mm, 0, b"R1LOWST1", 32, ENTRY.itemsize, CAPACITY, 31, 0)
        self.entries = np.frombuffer(self.mm, ENTRY, CAPACITY, 32)
        self.count = 0

    def write(self, tick):
        e = self.entries[self.count % CAPACITY]
        e["seq"] = 0
        e["tick"], e["wall"], e["q"], e["quat"] = tick, tick / 1000.0, np.full(31, tick), (1, 0, 0, 0)
        e["seq"] = self.count + 1
        self.count += 1
        struct.pack_into("<Q", self.mm, 24, self.count)

    def close(self):
        del self.entries
        self.mm.close()
        self.f.close()
        os.unlink(self.path)


@pytest.fixture
def pair():
    name = f"/r1_lowstate_test_{uuid.uuid4().hex[:8]}"
    w = Writer(name)
    yield w, LowstateShmSource(name)
    w.close()


def test_no_file_gives_nothing():
    assert len(LowstateShmSource("/r1_lowstate_missing").read_new()) == 0


def test_new_entries_come_once_in_order(pair):
    w, r = pair
    r.read_new()  # opens at the current end
    for t in (2, 4, 6):
        w.write(t)
    rows = r.read_new()
    assert rows["tick"].tolist() == [2, 4, 6] and rows["q"][1, 0] == 4
    assert len(r.read_new()) == 0
    w.write(8)
    assert r.read_new()["tick"].tolist() == [8]


def test_a_slow_reader_loses_the_oldest_and_counts_them(pair):
    w, r = pair
    r.read_new()
    for t in range(20):
        w.write(t)
    rows = r.read_new()
    assert rows["tick"].tolist() == list(range(13, 20))  # capacity - 1 newest
    assert r.lost == 13
