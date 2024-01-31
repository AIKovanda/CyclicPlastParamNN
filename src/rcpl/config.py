from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CONFIGS_DIR = BASE_DIR / 'configs'

DATA_DIR = BASE_DIR / 'data'
TASKS_DIR = DATA_DIR / 'tasks'
RUNS_DIR = DATA_DIR / 'runs'

MEASURED_EXP_DIR = DATA_DIR / 'epsp_stress' / 'measured'
GENERATED_EXP_DIR = DATA_DIR / 'epsp_stress' / 'generated'
