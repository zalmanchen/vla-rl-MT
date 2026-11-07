"""Resume-safe piRL training and paired/retention evaluation using pinned RLinf."""
import concurrent.futures,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'.runtime/experiments/e5-pirl'

def write(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def status(phase,detail='',**kw):
 write(RUN/'status.json',{'phase':phase,'detail':detail,'updated_unix':time.time(),'pid':os.getpid(),**kw})
 subprocess.run([sys.executable,str(ROOT/'experiments/render_pirl_progress.py')],cwd=ROOT,check=True)
def launch(script,args,logfile,timeout=24*3600):
 command=['bash','-c','source bootstrap/env.sh; export PYTHONPATH="$PROJECT_ROOT/experiments:$PYTHONPATH"; export RLINF_EXT_MODULE=pirl_extension; export OMP_NUM_THREADS=4; export HYDRA_FULL_ERROR=1; exec python -u "$@"','--','experiments/'+script,*args]
 if script=='eval_pirl.py':command[2]=command[2].replace('export PYTHONPATH="$PROJECT_ROOT/experiments:', 'export PYTHONPATH="$PROJECT_ROOT/.runtime/reference-transformers:$PROJECT_ROOT/experiments:')
 logfile.parent.mkdir(parents=True,exist_ok=True)
 with logfile.open('a') as log:
  child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(logfile.with_suffix('.process.json'),{'pid':child.pid,'command':command,'started_unix':time.time()})
  deadline=time.time()+timeout
  while child.poll() is None:
   if time.time()>deadline:
    os.killpg(child.pid,signal.SIGTERM);raise TimeoutError(str(logfile))
   subprocess.run([sys.executable,str(ROOT/'experiments/render_pirl_progress.py')],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   time.sleep(20)
  if child.returncode:raise RuntimeError(f'{script} returned {child.returncode}: {logfile}')
def checkpoint(job,step):return RUN/'jobs'/job['name']/job['name']/f'checkpoints/global_step_{step}/actor'
def evaluate(job,tasks,run,checkpoint_path='base',episodes=100):
 def one(task):
  dest=run/'clean'/task
  complete=[json.loads(p.read_text()) for p in dest.glob('episode_*.json')]
  if len(complete)>=episodes and all(x['status']=='completed' for x in complete):return
  launch('eval_pirl.py',['--task',task,'--regime','clean','--episodes',str(episodes),'--run',str(run),'--seed-protocol','fixed_eval','--backend','reference','--config',job['config'],'--checkpoint',str(checkpoint_path)],run/'logs'/f'{task}.log',timeout=3*3600)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for future in [pool.submit(one,t) for t in tasks]:future.result()
 results=[]
 for task in tasks:
  records=[json.loads(p.read_text()) for p in (run/'clean'/task).glob('episode_*.json')]
  assert len(records)==episodes and all(x['status']=='completed' for x in records)
  base=ROOT/'.runtime/experiments/e1-reference-clean/clean'/task
  before=[json.loads((base/f"episode_{x['seed']}.json").read_text()) for x in records]
  results.append({'task':task,'episodes':episodes,'successes':sum(x['success'] for x in records),'original_reference_successes_same_seeds':sum(x['success'] for x in before),'new_wins':sum(x['success'] and not b['success'] for x,b in zip(records,before)),'new_losses':sum(not x['success'] and b['success'] for x,b in zip(records,before))})
 write(run/'results.json',results)
def main():
 pilot=RUN/'pilot-plan.json'
 if pilot.exists() and not json.loads(pilot.read_text()).get('formal_queue_enabled',False):
  raise SystemExit('Full matrix deferred by pilot-plan.json; use pirl_pilot_controller.py for quick diagnostics.')
 lock=(RUN/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan=json.loads((RUN/'plan.json').read_text());deadline=time.time()+1800
 status('等待数值与环境验收','首个官方PPO单次更新已完成；验收失败不会自动放行大规模训练。')
 while True:
  audits=[RUN/'zero-update-audit.json',RUN/'multitask-env-smoke.json',RUN/'weight-roundtrip-audit.json',RUN/'reward-mask-audit.json',RUN/'checkpoint-reference-smoke.json',RUN/'terminal-integration-audit.json']
  if all(p.exists() for p in audits) and (RUN/'replay-smoke/SMOKE_PASSED').exists():
   values=[json.loads(p.read_text()) for p in audits]
   if any(v['status']!='pass' for v in values):raise RuntimeError('Prerequisite audit failed')
   break
  if time.time()>deadline:raise TimeoutError('Prerequisite audit did not complete in 30 minutes')
  time.sleep(15)
 gpu=(ROOT/'.runtime/experiments/gpu-evaluation.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
 jobs=[j for j in plan['jobs'] if j['stage']!='smoke'];priority=['adjust_bottle','stack_blocks_two','place_container_plate','mt5','click_bell','place_empty_cup']
 jobs.sort(key=lambda j:(j['seed'],priority.index('mt5' if j['stage']=='multi' else j['tasks'][0])))
 zero=RUN/'zero-update-evaluation'
 status('零更新闭环评估','相同五任务各20场景；量化新入口相对原始参考实现的差异。')
 evaluate(jobs[0],plan['tasks'],zero,episodes=20)
 for i,job in enumerate(jobs):
  out=RUN/'jobs'/job['name'];out.mkdir(parents=True,exist_ok=True)
  if not (out/'TRAINING_FINISHED').exists():
   status('在线PPO训练',job['name'],job_index=i+1,total_jobs=len(jobs))
   write(out/'status.json',{'phase':'training','started_unix':time.time()})
   args=['--config-path',str(RUN/'configs'),'--config-name',job['name']]
   checkpoints=list((out/job['name']/'checkpoints').glob('global_step_*'))
   if checkpoints:
    latest=max(checkpoints,key=lambda p:int(p.name.split('_')[-1]))
    args.append('runner.resume_dir='+str(latest))
   for micro in [4,2,1]:
    try:
     launch('run_pirl_job.py',args+[f'actor.micro_batch_size={micro}'],out/f'training-micro{micro}.log')
     break
    except RuntimeError:
     text=(out/f'training-micro{micro}.log').read_text(errors='replace')[-50000:]
     if 'out of memory' not in text.lower() or micro==1:raise
     status('减小微批次后重试',job['name']+f' micro_batch={micro}')
   if not (checkpoint(job,100)/'model_state_dict/full_weights.pt').exists():raise RuntimeError('Final checkpoint missing')
   (out/'TRAINING_FINISHED').write_text(str(time.time()))
   interactions=out/'online-chunks/interactions.jsonl'
   counts={}
   if interactions.exists():
    for line in interactions.read_text().splitlines():
     x=json.loads(line);t=counts.setdefault(x['task'],{'requested_actions':0,'executed_actions':0,'chunks':0})
     t['requested_actions']+=x['requested_actions'];t['executed_actions']+=x['executed_actions'];t['chunks']+=1
   write(out/'interaction-budget.json',counts)
  status('训练任务配对评估',job['name']);write(out/'status.json',{'phase':'evaluating trained tasks'})
  evaluate(job,job['tasks'],out/'evaluation',checkpoint(job,100))
  for step in [10,20,50]:
   status('中间检查点曲线评估',job['name']+f' step {step}')
   evaluate(job,job['tasks'],out/f'curve-{step}',checkpoint(job,step),episodes=20)
  write(out/'status.json',{'phase':'trained-task evaluation complete'})
  control=out/'success-replay'
  control_steps=job['rollout_iterations']*5*len(job['tasks'])
  status('同经验成功重放对照',job['name'])
  launch('train_pirl_success_replay.py',['--config',job['config'],'--source',str(out/'online-chunks'),'--output',str(control),'--steps',str(control_steps),'--micro-batch','4'],control/'training.log')
  evaluate(job,job['tasks'],control/'evaluation',control)
  write(out/'status.json',{'phase':'PPO and matched replay trained-task evaluation complete'})
 # Retention is a separate stage: no missing tasks are counted as failures.
 tasks=list(json.loads((ROOT/'.research/e1-results.json').read_text())['contract']['tasks'])
 for job in jobs:
  out=RUN/'jobs'/job['name'];status('22任务能力保持评估',job['name'])
  evaluate(job,tasks,out/'evaluation',checkpoint(job,100))
  evaluate(job,tasks,out/'success-replay/evaluation',out/'success-replay')
  write(out/'status.json',{'phase':'complete: PPO and replay 22-task evaluation'})
 status('单任务／五任务／能力保持队列完成','长程连续任务仍为独立待完成阶段。')
if __name__=='__main__':
 try:main()
 except Exception as e:
  status('队列需检查',str(e));raise
