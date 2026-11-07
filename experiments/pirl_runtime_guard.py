"""CPU-only resource gates and restart accounting for the E7 queue."""
from pathlib import Path
import json
import re

GIB = 1024 ** 3


def memory_sample(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    """Use the tighter host/container headroom; retain raw and Ray working-set usage."""
    info = {line.split(':')[0]: int(line.split()[1]) * 1024
            for line in (proc / 'meminfo').read_text().splitlines() if ':' in line}
    total, available = info['MemTotal'], info['MemAvailable']
    raw = working = None; limit = total
    current = cgroup / 'memory.current'
    if current.exists():
        raw = int(current.read_text())
        stats = dict(line.split() for line in (cgroup / 'memory.stat').read_text().splitlines())
        working = max(0, raw - int(stats.get('inactive_file', 0)) - int(stats.get('active_file', 0)))
        max_text = (cgroup / 'memory.max').read_text().strip()
        if max_text != 'max': limit = min(limit, int(max_text))
        available = min(available, max(0, limit - working))
    used = max(total - info['MemAvailable'], working or 0)
    return {'total_gib':limit/GIB, 'available_gib':available/GIB,
            'used_gib':used/GIB, 'pressure_fraction':max(1-available/limit, used/limit),
            'host_available_gib':info['MemAvailable']/GIB,
            'cgroup_current_gib':raw/GIB if raw is not None else None,
            'cgroup_working_gib':working/GIB if working is not None else None}


def evaluation_allowed(gpu, ram, policy, pending=0, training=False, gpu_peak=0, ram_peak=0, iterations=0):
    if training and iterations < policy['overlap_after_training_iterations']: return False
    gpu_needed = policy['eval_reserve_gib'] * (1 + pending) + policy['minimum_headroom_gib']
    ram_needed = policy['eval_ram_reserve_gib'] * (1 + pending) + policy['minimum_ram_headroom_gib']
    if gpu['total_gib'] - gpu['used_gib'] < gpu_needed or ram['available_gib'] < ram_needed: return False
    if training:
        return (gpu_peak + gpu_needed <= gpu['total_gib']
                and ram_peak + ram_needed <= ram['total_gib'] * .90)
    return True


def should_defer(gpu, ram, policy):
    return (gpu['total_gib'] - gpu['used_gib'] < 8
            or ram['available_gib'] < policy['defer_ram_headroom_gib']
            or ram['pressure_fraction'] >= .85)


FATAL_MARKERS = ('ray.exceptions.OutOfMemoryError:', 'torch.OutOfMemoryError:',
                 'CUDA out of memory.', 'Workers (tasks / actors) killed due to memory pressure',
                 'Connection closed by peer', 'ray.exceptions.ActorDiedError:',
                 'ray.exceptions.WorkerCrashedError:', 'Pre-update logprob mismatch exceeds')


class FatalLogScanner:
    def __init__(self): self.offset = 0; self.tail = ''
    def scan(self, path):
        if not path.exists(): return None
        with path.open('rb') as f:
            f.seek(self.offset); data = f.read(); self.offset = f.tell()
        text = self.tail + data.decode(errors='replace'); self.tail = text[-1024:]
        for line in text.splitlines():
            if any(marker in line for marker in FATAL_MARKERS):
                return re.sub(r'\x1b\[[0-9;]*m', '', line)[-1500:]
        return None


def lineage_records(records, attempts):
    """Exclude updates rolled back by each resume, while retaining original logs."""
    for attempt in attempts:
        cutoff, started = attempt['retained_until_unix'], attempt['started_unix']
        records = [r for r in records if r['time_unix'] <= cutoff or r['time_unix'] >= started]
    return records


def metric_segments(data, attempts):
    """Return segments and the rollback point applied before each segment."""
    offset = 0; rollback = None
    for attempt in attempts:
        end = attempt['metrics_start_bytes']
        yield rollback, data[offset:end].decode(errors='replace')
        offset = end; rollback = attempt['resume_step']
    yield rollback, data[offset:].decode(errors='replace')


def process_tree(root_pid):
    """Capture only descendants of our launched driver, with PID-reuse identity."""
    records = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            records[int(p.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError): continue
    found = {root_pid}; changed = True
    while changed:
        extra = {pid for pid, (parent, _) in records.items() if parent in found} - found
        changed = bool(extra); found.update(extra)
    return {pid: records[pid][1] for pid in found if pid in records}


def identity_alive(pid, started):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] == started and fields[0] != 'Z'
    except (OSError, IndexError): return False
