#!/usr/bin/env python3
"""The R1's height map for the perceptive policy: depth frames and rt/lowstate in, rt/heightmap out.

    .venv/bin/python heightmap_node.py                 # simulator: depth from /dev/shm/r1_depth, DDS on lo
    .venv/bin/python heightmap_node.py --iface eth0    # (robot: needs a depth source for its camera module)

rt/lowstate drives the odometry at up to 200 Hz: the DDS callback only keeps the newest sample
and the main loop takes it every 5 ms. Doing the work in the callback for all of the simulator's
1000 samples a second starved the loop (Python's DDS library holds the GIL while it decodes),
and 200 Hz tracks as well as all samples did on a recorded walk.

Each depth frame (10 Hz) goes into the map and a scan goes out on rt/heightmap (unitree_go
HeightMap_): 17 x 11 values, 0.1 m apart, origin (-0.8, -0.5) m, frame "pelvis_yaw", data rows
along y with x within a row, each the pelvis height minus the ground height in metres; stamp is
the depth frame's capture time. The layout is r1_perception/mapper.py's, Unitree-R1-Perceptive's
training scan. Ctrl+C ends it.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time

import numpy as np

from r1_perception.depth_source import ShmDepthSource
from r1_perception.mapper import SCAN_X, SCAN_Y, HeightMapper

TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HEIGHTMAP = "rt/heightmap"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo", help="DDS network interface (lo: the simulator)")
    ap.add_argument("--shm", default="/r1_depth", help="the simulator's depth shared memory")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    mapper = HeightMapper()
    n = mapper.model.num_motors
    counts = {"lowstate": 0, "depth": 0, "unmatched": 0}
    newest = [None]   # (tick, q, quat, arrival): replaced whole, so the loop never sees half of one

    def on_lowstate(msg) -> None:
        newest[0] = (msg.tick, [m.q for m in msg.motor_state[:n]], list(msg.imu_state.quaternion), time.time())

    ChannelFactoryInitialize(0, args.iface)
    sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
    sub.Init(on_lowstate, 10)
    pub = ChannelPublisher(TOPIC_HEIGHTMAP, HeightMap_)
    pub.Init()
    source = ShmDepthSource(args.shm)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"[heightmap] {TOPIC_LOWSTATE} on {args.iface}, depth from /dev/shm{args.shm}, publishing {TOPIC_HEIGHTMAP}",
          flush=True)
    report_at, latency, last_tick = time.time() + 2.0, [], None
    while not stop.is_set():
        sample = newest[0]
        if sample is not None and sample[0] != last_tick:   # a new physics step / robot sample
            last_tick = sample[0]
            if mapper.on_lowstate(sample[1], sample[2], sample[0] / 1000.0, sample[3]):
                counts["lowstate"] += 1
        frame = source.read()
        if frame is None:
            time.sleep(0.005)
        else:
            depth, k, stamp = frame
            cells = mapper.on_depth(depth, k, stamp)
            scan = mapper.scan() if cells else None
            if scan is None:
                counts["unmatched"] += 1
            else:
                pub.Write(HeightMap_(stamp=stamp, frame_id="pelvis_yaw", resolution=0.1,
                                     width=len(SCAN_X), height=len(SCAN_Y),
                                     origin=[float(SCAN_X[0]), float(SCAN_Y[0])],
                                     data=[float(v) for v in scan]))
                counts["depth"] += 1
                latency.append(time.time() - stamp)
        if time.time() >= report_at:
            pos, fixes = mapper.odometry.position.copy(), mapper.height_fixes[-20:]
            print(f"[heightmap] maps {counts['depth']}, lowstate {counts['lowstate']}, "
                  f"no pose for {counts['unmatched']}; latency {np.median(latency) * 1000 if latency else float('nan'):.0f} ms; "
                  f"odometry {np.round(pos, 3).tolist()}; last height fixes {np.round(np.array(fixes) * 100, 1).tolist()} cm",
                  flush=True)
            report_at, latency = time.time() + 2.0, []
    print("[heightmap] stopped", flush=True)


if __name__ == "__main__":
    main()
