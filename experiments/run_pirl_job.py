"""Launch the pinned RLinf embodied runner with project-owned experiment configs."""
import os,runpy
from pathlib import Path
import pirl_extension
pirl_extension.register()
if __name__=='__main__':
 runpy.run_path(str(Path(os.environ['REPO_PATH'])/'examples/embodiment/train_embodied_agent.py'),run_name='__main__')
