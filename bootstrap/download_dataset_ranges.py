"""Download pinned public Clean data using bounded ranges and content hashes."""
import argparse
import concurrent.futures
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

import requests
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path, etag):
    if re.fullmatch('[0-9a-f]{64}', etag):
        result = hashlib.sha256()
    elif re.fullmatch('[0-9a-f]{40}', etag):
        result = hashlib.sha1()
        result.update(f'blob {path.stat().st_size}\0'.encode())
    else:
        raise ValueError('Unsupported content hash')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download_file(base, manifest, item, workers, chunk, endpoint):
    relative, size, etag = item['path'], item['size'], item['etag']
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = base / '.cache/huggingface/download' / (relative + '.metadata')
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size and digest(target, etag) == etag:
        metadata.write_text(f"{manifest['revision']}\n{etag}\n{time.time()}\n")
        print('ALREADY_VERIFIED', relative, flush=True)
        return
    part = target.with_name(target.name + '.range.partial')
    receipt = metadata.with_suffix('.ranges.json')
    if receipt.exists() and part.exists():
        state = json.loads(receipt.read_text())
        if (state['revision'], state['etag'], state['size'], state['chunk']) != (manifest['revision'], etag, size, chunk):
            raise ValueError('Range receipt does not match pinned file')
    else:
        # Hub 2.x uses a hash of the filename, followed by the content ETag.
        candidates = list(metadata.parent.glob(f'*.{etag}.*.incomplete'))
        if not candidates:
            candidates = list(metadata.parent.glob(f'*.{etag}.incomplete'))
        prefix = max(candidates, key=lambda p: p.stat().st_size) if candidates else None
        if prefix:
            shutil.copyfile(prefix, part)
        else:
            part.touch()
        prefix_size = part.stat().st_size
        if prefix_size > size:
            raise ValueError('Cached prefix exceeds file size')
        state = dict(revision=manifest['revision'], etag=etag, size=size,
                     prefix=prefix_size, chunk=chunk, done=[])
        atomic_json(receipt, state)
    fd = os.open(part, os.O_RDWR)
    os.ftruncate(fd, size)
    done = set(state['done'])
    ranges = [(start, min(start + chunk, size)) for start in range(state['prefix'], size, chunk) if start not in done]
    mutex = threading.Lock()
    local = threading.local()
    url = hf_hub_url(manifest['repo'], relative, repo_type='dataset',
                     revision=manifest['revision'], endpoint=endpoint)
    print('START', relative, 'prefix_bytes', state['prefix'], 'pending_chunks', len(ranges), flush=True)

    def fetch(bounds):
        if not hasattr(local, 'session'):
            local.session = requests.Session()  # Public files; no account token is sent.
        start, end = bounds
        for attempt in range(8):
            try:
                with local.session.get(url, headers={'Range': f'bytes={start}-{end-1}'},
                                       timeout=(15, 45), stream=True) as response:
                    response.raise_for_status()
                    expected = f'bytes {start}-{end-1}/{size}'
                    if not (response.status_code == 206 and response.headers.get('Content-Range') == expected):
                        if not (response.status_code == 200 and start == 0 and end == size):
                            raise ValueError('Server did not honor byte range')
                    blocks, length = [], 0
                    for block in response.iter_content(1024 * 1024):
                        length += len(block)
                        if length > end - start:
                            raise ValueError('Response exceeds requested range')
                        blocks.append(block)
                    if length != end - start:
                        raise ValueError('Incomplete range')
                    data = b''.join(blocks)
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], start + offset)
                with mutex:
                    state['done'].append(start)
                    atomic_json(receipt, state)
                    completed = state['prefix'] + sum(min(chunk, size - x) for x in state['done'])
                    print('PROGRESS', relative, f'{completed}/{size}', f'{completed / size:.1%}', flush=True)
                return
            except Exception as exc:
                print('RETRY', relative, start, attempt + 1, type(exc).__name__, flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(15, attempt * 2 + 1))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fetch, ranges):
                pass
        os.fsync(fd)
    finally:
        os.close(fd)
    if digest(part, etag) != etag:
        raise ValueError('Full content hash mismatch: ' + relative)
    part.replace(target)
    metadata.write_text(f"{manifest['revision']}\n{etag}\n{time.time()}\n")
    print('FILE_VERIFIED', relative, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-mib', type=int, default=8)
    parser.add_argument('--endpoint', default='https://huggingface.co')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.chunk_mib <= 32:
        parser.error('workers must be 1..16; chunk-mib must be 1..32')
    manifest = next(m for m in json.loads((ROOT / 'bootstrap/downloads.lock.json').read_text())
                    if m['destination'] == 'datasets/RoboTwin2.0')
    base = ROOT / manifest['destination']
    base.mkdir(parents=True, exist_ok=True)
    with (base / '.range-downloader.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = HfApi(endpoint='https://huggingface.co', token=False).repo_info(
            manifest['repo'], repo_type='dataset', revision=manifest['revision'], files_metadata=True)
        if info.sha != manifest['revision']:
            raise ValueError('Remote revision mismatch')
        remote = {f.rfilename: f for f in info.siblings}
        files = []
        for f in manifest['files']:
            if not any(fnmatch.fnmatchcase(f['path'], pattern) for pattern in manifest['patterns']):
                continue
            item = remote[f['path']]
            files.append(dict(path=f['path'], size=item.size,
                              etag=item.lfs.sha256 if item.lfs else item.blob_id))
        # Prepare demonstrations first, then the three shared simulation archives.
        files.sort(key=lambda f: (not f['path'].startswith('dataset/'), f['path']))
        atomic_json(base / 'download-manifest.json', manifest)
        atomic_json(ROOT / '.runtime/bootstrap/clean-range-files.json', files)
        for item in files:
            download_file(base, manifest, item, args.workers, args.chunk_mib * 1024 * 1024, args.endpoint)
        (base / 'DOWNLOAD_COMPLETE').write_text(manifest['revision'] + '\n')
        print('CLEAN_AND_ASSETS_COMPLETE_CONTENT_HASH_VERIFIED', len(files), flush=True)


if __name__ == '__main__':
    main()
