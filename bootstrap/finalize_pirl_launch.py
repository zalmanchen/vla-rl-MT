"""Reload the E5 driver after baseline children finish, preserving all their results."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'.runtime/experiments/e5-pirl'
receipt=json.loads((RUN/'controller-reload.json').read_text());old=receipt['pid']
plan=json.loads((RUN/'plan.json').read_text());deadline=time.time()+1800
while True:
 complete=True
 for task in plan['tasks']:
  records=[json.loads(p.read_text()) for p in (RUN/'zero-update-evaluation/clean'/task).glob('episode_*.json')]
  complete &= len(records)==20 and all(x['status']=='completed' for x in records)
 if complete:break
 if time.time()>deadline:raise TimeoutError('Baseline children not complete; old controller still suspended')
 time.sleep(10)
# The old driver holds no training job: it was suspended during baseline evaluation.
os.kill(old,signal.SIGTERM);os.kill(old,signal.SIGCONT)
for _ in range(30):
 try:
  if Path(f'/proc/{old}/stat').read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(1)
with (RUN/'controller.log').open('a') as log:
 p=subprocess.Popen([str(ROOT/'.runtime/venv-tools/bin/python'),'-u',str(ROOT/'experiments/pirl_batch_controller.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
(RUN/'controller.json').write_text(json.dumps({'pid':p.pid,'started_unix':time.time(),'superseded_pid':old,'reason':'final code frozen before formal training'},indent=2));print('formal queue controller',p.pid,flush=True)
