"""Restart the first E5 controller once after the corrected launch syntax error."""
import json,subprocess,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'.runtime/experiments/e5-pirl'
old=json.loads((RUN/'controller.json').read_text())['pid']
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
while alive(old):time.sleep(15)
# This restart is specifically for the initial pre-training syntax failure.
state=json.loads((RUN/'status.json').read_text())
if state['phase']!='队列需检查':raise SystemExit('Previous queue did not fail; no automatic duplicate launch')
with (RUN/'controller.log').open('a') as log:
 p=subprocess.Popen([str(ROOT/'.runtime/venv-tools/bin/python'),'-u',str(ROOT/'experiments/pirl_batch_controller.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(RUN/'controller.json').write_text(json.dumps({'pid':p.pid,'started_unix':time.time(),'superseded_pid':old,'reason':'initial pre-training extension syntax corrected'},indent=2))
print('restarted',p.pid,flush=True)
