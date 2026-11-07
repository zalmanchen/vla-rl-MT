"""Non-destructive report/Hydra organization with a complete relocation manifest."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import yaml
from research_artifacts import ROOT, RESEARCH, LOCATIONS, relocate_html, relocate_markdown


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def open_output_paths():
    paths = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            descriptors = list((proc / 'fd').iterdir())
        except (PermissionError, FileNotFoundError):
            continue
        for fd in descriptors:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith(str(ROOT / 'outputs') + '/'):
                paths.add(target)
    return paths


def planned_moves():
    moves, skipped = [], []
    for old, new in LOCATIONS.items():
        source, dest = RESEARCH / old, RESEARCH / new
        if source.is_symlink() or not source.exists():
            continue
        if dest.exists():
            raise FileExistsError(f'Refusing to overwrite {dest}')
        moves.append((source, dest, 'research'))
    opened = open_output_paths()
    for config in sorted((ROOT / 'outputs').glob('20??-??-??/*/.hydra/config.yaml')):
        folder = config.parent.parent
        if any(path.startswith(str(folder) + '/') for path in opened):
            skipped.append({'path': str(folder.relative_to(ROOT)), 'reason': 'open by a running process'})
            continue
        cfg = yaml.safe_load(config.read_text())
        log_path = Path(cfg['runner']['logger']['log_path'])
        prefix = ROOT / '.runtime/experiments'
        if not log_path.is_relative_to(prefix):
            skipped.append({'path': str(folder.relative_to(ROOT)), 'reason': 'unrecognized experiment owner'})
            continue
        experiment = log_path.relative_to(prefix).parts[0]
        dest = ROOT / 'outputs/archive/hydra' / experiment / f'{folder.parent.name}-{folder.name}'
        if dest.exists():
            raise FileExistsError(dest)
        moves.append((folder, dest, 'hydra'))
    return moves, skipped


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    moves, skipped = planned_moves()
    if not args.apply:
        print(json.dumps({'moves': [{'from': str(a.relative_to(ROOT)), 'to': str(b.relative_to(ROOT)), 'kind': k}
                                    for a, b, k in moves], 'skipped': skipped}, ensure_ascii=False, indent=2))
        return
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    archive = RESEARCH / 'archive' / f'layout-{stamp}'; archive.mkdir(parents=True)
    manifest = {'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'moves': [],
                'skipped': skipped, 'backups': [], 'rewritten_documents': [],
                'policy': 'No checkpoint, dataset, model, raw episode or active run directory is deleted or relocated.'}
    def backup(path):
        dest = archive / 'original' / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        manifest['backups'].append({'source': str(path.relative_to(ROOT)), 'backup': str(dest.relative_to(ROOT)), 'sha256': digest(path)})
    for path in [RESEARCH / 'progress.html', ROOT / 'experiments/render_progress.py', ROOT / 'experiments/README.md']:
        if path.exists(): backup(path)
    for source, dest, kind in moves:
        files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
        records = [{'relative': str(p.relative_to(source)) if source.is_dir() else '',
                    'bytes': p.stat().st_size, 'sha256_before': digest(p)} for p in files]
        original_text = None
        if source.is_file() and source.suffix in ['.html', '.md']:
            original_text = source.read_text(); backup(source)
        # Re-check open Hydra handles immediately before the rename.
        if kind == 'hydra' and any(p.startswith(str(source) + '/') for p in open_output_paths()):
            manifest['skipped'].append({'path': str(source.relative_to(ROOT)), 'reason': 'became active during organization'})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source == RESEARCH / 'e1-results.json':
            # Publish the compatibility alias atomically for the live PACE reader.
            shutil.copy2(source, dest)
            temporary_alias = source.with_name('.e1-results-alias.tmp')
            temporary_alias.symlink_to(os.path.relpath(dest, source.parent))
            temporary_alias.replace(source)
        else:
            source.replace(dest)
        if original_text is not None:
            text = (relocate_html(original_text, str(source.relative_to(RESEARCH)), dest)
                    if source.suffix == '.html' else relocate_markdown(original_text, source, dest))
            dest.write_text(text)
        for row in records:
            path = dest / row['relative'] if row['relative'] else dest
            row['sha256_after'] = digest(path)
            if original_text is None:
                assert row['sha256_before'] == row['sha256_after'], path
        manifest['moves'].append({'from': str(source.relative_to(ROOT)), 'to': str(dest.relative_to(ROOT)), 'kind': kind, 'files': records})
    # The running PACE downloader still reads this legacy baseline path.
    alias = RESEARCH / 'e1-results.json'
    if not alias.exists():
        alias.symlink_to('results/e1/e1-results.json')
    manifest['compatibility_aliases'] = [{'path': '.research/e1-results.json', 'target': 'results/e1/e1-results.json', 'reason': 'running PACE controller'}]
    for path in [RESEARCH / 'takeaway.md', RESEARCH / '汇报.html']:
        if not path.exists(): continue
        backup(path); old = path.read_text()
        new = (relocate_html(old, path.name, path) if path.suffix == '.html'
               else relocate_markdown(old, path, path))
        path.write_text(new)
        manifest['rewritten_documents'].append({'path': str(path.relative_to(ROOT)), 'sha256_after': digest(path)})
    (archive / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (RESEARCH / 'organization-latest.json').write_text(json.dumps({'manifest': str((archive / 'manifest.json').relative_to(ROOT))}, indent=2))
    print(json.dumps({'manifest': str(archive / 'manifest.json'), 'moved_groups': len(manifest['moves']), 'preserved_active': skipped}, ensure_ascii=False))


if __name__ == '__main__':
    main()
