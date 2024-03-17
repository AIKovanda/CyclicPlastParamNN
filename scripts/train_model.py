import sys

from taskchain.task import Config

from rcpl import config

DEVICE = 'cuda'

relative_path = sys.argv[1]
config_path = config.CONFIGS_DIR / relative_path

conf = Config(
    config.TASKS_DIR,  # where Taskchain data should be stored
    config_path,
    context={'device': DEVICE, 'run_name': relative_path},
    global_vars=config,  # set global variables
)
chain = conf.chain()
chain.set_log_level('CRITICAL')

print(chain.dataset_info.value)

_ = chain.train_model.value
# _ = chain.train_model.force().value

print(f"Calculating metrics for {sys.argv[1]}")
_ = chain.model_metrics.force().value
