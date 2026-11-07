"""Run the pinned trainer with measurement-only E6 extensions."""
import os
from pathlib import Path
import runpy
import pirl_diagnostic_extension

if __name__ == '__main__':
    pirl_diagnostic_extension.register()
    runpy.run_path(str(Path(os.environ['REPO_PATH']) / 'examples/embodiment/train_embodied_agent.py'), run_name='__main__')
