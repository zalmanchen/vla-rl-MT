"""Canonical destinations for research reports; training artifacts stay in place."""
import html
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / '.research'
GROUPS = {
    'results/e1': ['e1-results.json', 'e1-results.csv'],
    'archive/e1-before-resume': ['e1-results-initial.json', 'e1-results-initial.csv'],
    'results/e3': ['motus-window-diagnostic.json'],
    'results/e4': ['e4-analysis.html', 'e4-analysis.json', 'e4-analysis.csv', 'offline-rl.html',
                   'offline-rl-pipeline-status.json', 'offline-batch-status.json', 'replay-evaluation-status.json'],
    'results/e5': ['pirl-progress.html', 'pirl-pilot-results.json', 'pirl-training-metrics.json',
                   'pirl-training-diagnostics.png', 'pirl-training-diagnostics.svg'],
    'results/e6': ['stability-progress.html', 'stability-progress.json', 'e6-progress-analysis.json'],
    'results/e7': ['e7-analysis.html', 'e7-analysis.json', 'e7-analysis.csv', 'e7-paired-scenes.csv', 'e7-analysis.png', 'e7-analysis.svg'],
    'diagnostics/model': ['motus-transform-audit.json', 'motus-checkpoint-metadata-audit.json'],
    'diagnostics/recap': ['e4-initialization-audit.json', 'e4-recap-paper-audit.json'],
    'literature': ['motus-paper-reference.json', 'phenomena-paper-audit.html', 'phenomena-paper-audit.json',
                   'briefing-sources.json', 'training-regime-evidence.json', 'rl-algorithm-selection.json'],
    'planning': ['paper.md', 'proposal.md', 'post-training-plan.html', 'post-training-plan.json',
                 'unified-posttraining-assessment.html', 'unified-posttraining-assessment.json'],
}
LOCATIONS = {name: f'{folder}/{name}' for folder, names in GROUPS.items() for name in names}
LOCATIONS['literature-review-20261106'] = 'literature/papers-20261106'


def canonical_name(name):
    for old in sorted(LOCATIONS, key=len, reverse=True):
        if name == old or name.startswith(old + '/'):
            return LOCATIONS[old] + name[len(old):]
    return name


def artifact_path(name):
    path = RESEARCH / canonical_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def relocate_url(url, old_file, new_file):
    url = html.unescape(url)
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        return url
    target = Path(os.path.abspath(Path(old_file).parent / unquote(parts.path)))
    if target.is_relative_to(RESEARCH):
        target = RESEARCH / canonical_name(str(target.relative_to(RESEARCH)))
    relative = quote(os.path.relpath(target, Path(new_file).parent), safe='/._-')
    if parts.path.endswith('/'):
        relative += '/'
    return urlunsplit(('', '', relative, parts.query, parts.fragment))


def relocate_html(text, legacy_name, destination=None):
    old_file = RESEARCH / legacy_name
    new_file = Path(destination) if destination else artifact_path(legacy_name)
    pattern = r'(?P<prefix>\b(?:href|src)\s*=\s*)(?P<quote>[\"\x27])(?P<url>.*?)(?P=quote)'
    def replace(match):
        url = relocate_url(match['url'], old_file, new_file)
        return match['prefix'] + match['quote'] + html.escape(url, quote=True) + match['quote']
    result = re.sub(pattern, replace, text)
    if legacy_name in LOCATIONS and legacy_name.endswith('.html') and 'id="research-stage-note"' not in result:
        overview = html.escape(relocate_url('progress.html', old_file, new_file), quote=True)
        curves = html.escape(relocate_url('five-task-curves.html', old_file, new_file), quote=True)
        note = ('<aside id="research-stage-note" style="font:19px/1.8 system-ui;padding:18px;'
                'margin:20px 0;background:#e6f1f7;color:#193245;border-radius:8px">'
                '<strong>历史阶段记录</strong>：本页保留该阶段的结果与当时计划。'
                f'当前结论与安排见 <a href="{overview}">实验总览</a> 和 '
                f'<a href="{curves}">E7实时曲线</a>。</aside>')
        if re.search(r'<body\b[^>]*>', result, flags=re.I):
            result = re.sub(r'(<body\b[^>]*>)', lambda m: m[0] + note, result, count=1, flags=re.I)
        else:
            result = re.sub(r'(?=<h1\b)', lambda m: note, result, count=1, flags=re.I)
    return result


def relocate_markdown(text, old_file, new_file):
    return re.sub(r'\]\(([^\s)]+)\)',
                  lambda m: '](' + relocate_url(m[1], old_file, new_file) + ')', text)
