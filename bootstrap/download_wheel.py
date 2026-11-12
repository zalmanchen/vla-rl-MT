"""Fetch a locked wheel in resumable ranges with official registry SHA-256."""
import argparse
import concurrent.futures
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

import requests
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
INDEX = 'https://download.pytorch.org/whl/cu128/torch/'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default='torch', help='Package name pinned in environment.freeze.txt')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--proxy', help='Explicit proxy URL; omit to inherit environment')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be 1..16')
    locked = dict(line.split('==', 1) for line in (ROOT / 'bootstrap/environment.freeze.txt').read_text().splitlines() if '==' in line)
    if args.package not in locked:
        parser.error('Package must be pinned in environment.freeze.txt')
    version = locked[args.package]
    wheels = ROOT / '.runtime/wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    receipts = ROOT / '.runtime/bootstrap'
    receipts.mkdir(parents=True, exist_ok=True)

    def session():
        client = requests.Session()
        if args.proxy:
            client.trust_env = False
            client.proxies = {'http': args.proxy, 'https': args.proxy}
        return client

    with (wheels / f'.{args.package}-download.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidates = []

        class IndexParser(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag != 'a':
                    return
                href = dict(attrs).get('href', '')
                name = unquote(urlsplit(href).path.rsplit('/', 1)[-1])
                if name.startswith(f'torch-{version}-cp311-cp311-') and 'linux' in name and 'x86_64' in name:
                    url = urljoin(INDEX, href)
                    checksum = parse_qs(urlsplit(url).fragment).get('sha256', [''])[0]
                    if not re.fullmatch('[0-9a-f]{64}', checksum):
                        raise ValueError('Official wheel has no SHA-256')
                    candidates.append(dict(filename=name, url=url.split('#')[0], sha256=checksum))

        with session() as client:
            if args.package == 'torch':
                response = client.get(INDEX, timeout=(15, 45))
                response.raise_for_status()
                IndexParser().feed(response.text)
                if len(candidates) != 1:
                    raise ValueError('Expected exactly one official wheel for pinned version')
                wheel = candidates[0]
                with client.get(wheel['url'], headers={'Range': 'bytes=0-0'}, stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get('Content-Range', '')
                    if response.status_code != 206 or not re.fullmatch(r'bytes \d+-\d+/\d+', content_range):
                        raise ValueError('Official server does not support byte ranges')
                    wheel['size'] = int(content_range.split('/')[1])
            else:
                response = client.get(f'https://pypi.org/pypi/{args.package}/{version}/json', timeout=(15, 45))
                response.raise_for_status()
                supported = {tag: i for i, tag in enumerate(sys_tags())}
                choices = []
                for item in response.json()['urls']:
                    if item['packagetype'] != 'bdist_wheel':
                        continue
                    _, _, _, tags = parse_wheel_filename(item['filename'])
                    matches = [supported[tag] for tag in tags if tag in supported]
                    if matches:
                        choices.append((min(matches), item['filename'], item))
                if not choices:
                    raise ValueError('No compatible wheel on official PyPI for pinned package')
                item = min(choices, key=lambda choice: choice[:2])[2]
                wheel = dict(filename=item['filename'], url=item['url'],
                             sha256=item['digests']['sha256'], size=item['size'])
                if not re.fullmatch('[0-9a-f]{64}', wheel['sha256']):
                    raise ValueError('Official wheel has no SHA-256')
        write_json(receipts / f'{args.package}-wheel.json', wheel)
        target = wheels / wheel['filename']
        if target.is_file() and sha256(target) == wheel['sha256']:
            print('WHEEL_ALREADY_VERIFIED', target, flush=True)
            return
        part = target.with_suffix('.whl.partial')
        receipt = receipts / f'{args.package}-wheel-ranges.json'
        chunk = 8 * 1024 * 1024
        if part.exists() and receipt.exists():
            state = json.loads(receipt.read_text())
            if (state['sha256'], state['size'], state['chunk']) != (wheel['sha256'], wheel['size'], chunk):
                raise ValueError('Saved ranges do not match official wheel')
        else:
            part.touch(exist_ok=True)
            if part.stat().st_size > wheel['size']:
                raise ValueError('Saved prefix too large')
            state = dict(sha256=wheel['sha256'], size=wheel['size'], chunk=chunk,
                         prefix=part.stat().st_size, done=[])
            write_json(receipt, state)
        completed = set(state['done'])
        ranges = [(start, min(start + chunk, wheel['size']))
                  for start in range(state['prefix'], wheel['size'], chunk) if start not in completed]
        print('WHEEL_START', wheel['filename'], 'prefix_bytes', state['prefix'], 'pending_chunks', len(ranges), flush=True)
        fd = os.open(part, os.O_RDWR)
        os.ftruncate(fd, wheel['size'])
        mutex, local = threading.Lock(), threading.local()

        def fetch(bounds):
            if not hasattr(local, 'client'):
                local.client = session()
            start, end = bounds
            for attempt in range(8):
                try:
                    started = time.monotonic()
                    with local.client.get(wheel['url'], headers={'Range': f'bytes={start}-{end-1}'},
                                          timeout=(10, 15), stream=True) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get('Content-Range') != f"bytes {start}-{end-1}/{wheel['size']}":
                            raise ValueError('Incorrect byte range')
                        data = bytearray()
                        for block in response.iter_content(16 * 1024):
                            if time.monotonic() - started > 60:
                                raise TimeoutError('Range transfer exceeded 60 seconds')
                            data.extend(block)
                            if len(data) > end - start:
                                raise ValueError('Response exceeded requested range')
                        if len(data) != end - start:
                            raise ValueError('Incomplete byte range')
                    offset = 0
                    while offset < len(data):
                        offset += os.pwrite(fd, data[offset:], start + offset)
                    with mutex:
                        state['done'].append(start)
                        write_json(receipt, state)
                        done = state['prefix'] + sum(min(chunk, wheel['size'] - x) for x in state['done'])
                        print('WHEEL_PROGRESS', f"{done}/{wheel['size']}", f"{done / wheel['size']:.1%}", flush=True)
                    return
                except Exception as exc:
                    print('WHEEL_RETRY', start, attempt + 1, type(exc).__name__, flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(15, attempt * 2 + 1))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                for _ in pool.map(fetch, ranges):
                    pass
            os.fsync(fd)
        finally:
            os.close(fd)
        if sha256(part) != wheel['sha256']:
            raise ValueError('Official wheel SHA-256 mismatch; installation refused')
        part.replace(target)
        print('WHEEL_COMPLETE_SHA256_VERIFIED', target, flush=True)


if __name__ == '__main__':
    main()
