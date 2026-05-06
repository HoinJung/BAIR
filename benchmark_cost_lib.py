"""Shared resource monitoring for computational-cost benchmarks."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil


class ResourceMonitor:
    def __init__(self, gpu_id: int, interval: float = 0.2):
        self.gpu_id = gpu_id
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.samples: List[dict] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.process.cpu_percent(interval=None)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _process_tree(self):
        procs = [self.process]
        try:
            procs.extend(self.process.children(recursive=True))
        except psutil.Error:
            pass
        return procs

    def _gpu_sample(self, pids: set[int]) -> Dict[str, float]:
        proc_mem = 0.0
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                    mem = float(parts[1])
                except ValueError:
                    continue
                if pid in pids:
                    proc_mem += mem
        except Exception:
            proc_mem = np.nan

        try:
            util_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            util = float(util_out.splitlines()[0].strip())
        except Exception:
            util = np.nan
        return {"process_gpu_mem_used_mb": proc_mem, "global_gpu_util_pct": util}

    def _run(self):
        while not self.stop_event.is_set():
            rss = 0
            cpu = 0.0
            procs = self._process_tree()
            pids = set()
            for proc in procs:
                try:
                    pids.add(proc.pid)
                    rss += proc.memory_info().rss
                    cpu += proc.cpu_percent(interval=None)
                except psutil.Error:
                    continue
            vm = psutil.virtual_memory()
            gpu = self._gpu_sample(pids)
            self.samples.append(
                {
                    "time": time.time(),
                    "rss_mb": rss / (1024**2),
                    "cpu_percent": cpu,
                    "system_ram_used_mb": vm.used / (1024**2),
                    "system_ram_percent": vm.percent,
                    **gpu,
                }
            )
            self.stop_event.wait(self.interval)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        df = pd.DataFrame(self.samples)
        return {
            "peak_process_rss_mb": float(df["rss_mb"].max()),
            "mean_process_rss_mb": float(df["rss_mb"].mean()),
            "peak_cpu_percent": float(df["cpu_percent"].max()),
            "mean_cpu_percent": float(df["cpu_percent"].mean()),
            "peak_system_ram_used_mb": float(df["system_ram_used_mb"].max()),
            "mean_system_ram_percent": float(df["system_ram_percent"].mean()),
            "peak_process_gpu_mem_mb": float(df["process_gpu_mem_used_mb"].max()),
            "mean_process_gpu_mem_mb": float(df["process_gpu_mem_used_mb"].mean()),
            "peak_global_gpu_util_pct": float(df["global_gpu_util_pct"].max()),
            "mean_global_gpu_util_pct": float(df["global_gpu_util_pct"].mean()),
        }
