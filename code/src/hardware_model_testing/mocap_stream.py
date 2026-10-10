"""NatNet listener thread that stores the latest rigid-body pose."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
from typing import Any, Callable, Optional

import numpy as np


def _load_natnet_client(sdk_dir: Optional[str]):
    search = []
    if sdk_dir:
        search.append(os.path.abspath(sdk_dir))
    search.append(os.path.dirname(os.path.abspath(__file__)))
    for path in search:
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from NatNetClient import NatNetClient  # type: ignore

            return NatNetClient
        except ImportError:
            continue
    raise ImportError(
        "Could not import NatNetClient. Copy NatNetClient.py next to run_hardware.py "
        "or pass --mocap-sdk-dir pointing at NatNet SDK/Samples (or PythonClient)."
    )


def _silence_natnet_prints(NatNetClient) -> None:
    """SDK 4.x prints 'MoCap Frame: N' on every packet even when print_level=0."""
    orig = getattr(NatNetClient, "_NatNetClient__process_message", None)
    if orig is None or getattr(orig, "_mocap_quiet", False):
        return

    def quiet(self, data, print_level=0):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return orig(self, data, 0)

    quiet._mocap_quiet = True  # type: ignore[attr-defined]
    NatNetClient._NatNetClient__process_message = quiet


class NatNetPoseStream:
    """Background NatNet client; get_raw_pose() returns Motive Y-up position/quat."""

    def __init__(
        self,
        server_ip: str,
        client_ip: str,
        multicast: bool = False,
        sdk_dir: Optional[str] = None,
    ):
        self.server_ip = server_ip
        self.client_ip = client_ip
        self.multicast = multicast
        self.sdk_dir = sdk_dir
        self._lock = threading.Lock()
        self._poses: dict[int, tuple[np.ndarray, np.ndarray, bool, float]] = {}
        self._client: Any = None
        self._running = False

    def _on_rigid_body(self, *args):
        # SDK 3.x: (id, pos, rot, tracking_valid)
        # SDK 4.x: (id, pos, rot)  rot is (qx, qy, qz, qw)
        rb_id = int(args[0])
        pos = np.array(args[1], dtype=np.float64).reshape(3)
        rot = np.array(args[2], dtype=np.float64).reshape(4)
        valid = True if len(args) < 4 else bool(args[3])
        now = time.time()
        packed = (pos, rot, valid, now)
        with self._lock:
            self._poses[rb_id] = packed
            self._poses[rb_id & 0xFFFF] = packed

    def start(self) -> None:
        NatNetClient = _load_natnet_client(self.sdk_dir)
        _silence_natnet_prints(NatNetClient)
        client = NatNetClient()
        listener: Callable = self._on_rigid_body

        if hasattr(client, "set_server_address"):
            client.set_server_address(self.server_ip)
            client.set_client_address(self.client_ip)
            if hasattr(client, "set_use_multicast"):
                client.set_use_multicast(self.multicast)
            if hasattr(client, "set_print_level"):
                client.set_print_level(0)
            if hasattr(client, "rigid_body_listener"):
                client.rigid_body_listener = listener
            else:
                client.rigidBodyListener = listener
            ok = client.run() if hasattr(client, "run") else True
            if ok is False:
                raise RuntimeError(
                    "NatNetClient.run() failed. Check Motive streaming, IPs, unicast/multicast."
                )
        else:
            client.serverIPAddress = self.server_ip
            client.localIPAddress = self.client_ip
            client.rigidBodyListener = listener
            if hasattr(client, "run"):
                client.run()
            else:
                raise RuntimeError("This NatNetClient.py has no run(); use the SDK 4.x PythonClient.")

        self._client = client
        self._running = True
        print(
            f"NatNet started (server={self.server_ip}, client={self.client_ip}, "
            f"multicast={self.multicast})"
        )

    def seen_ids(self) -> list[int]:
        with self._lock:
            return sorted(self._poses.keys())

    def get_raw_pose(self, rigid_body_id: int) -> Optional[tuple[np.ndarray, np.ndarray, bool, float]]:
        want = int(rigid_body_id)
        with self._lock:
            pose = self._poses.get(want)
            if pose is not None:
                return pose
            return self._poses.get(want & 0xFFFF)

    def wait_for_pose(self, rigid_body_id: int, timeout_s: float = 8.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pose = self.get_raw_pose(rigid_body_id)
            if pose is not None:
                return pose
            time.sleep(0.05)
        seen = self.seen_ids()
        raise TimeoutError(
            f"No NatNet rigid body id={rigid_body_id} after {timeout_s:.1f}s "
            f"(seen ids={seen}). Use Motive Streaming ID, e.g. --mocap-rb-id 81."
        )

    def shutdown(self) -> None:
        self._running = False
        client = self._client
        if client is None:
            return
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if hasattr(client, "shutdown"):
                    client.shutdown()
                elif hasattr(client, "stop"):
                    client.stop()
        except Exception:
            pass
