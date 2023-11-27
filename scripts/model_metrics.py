import sys

from taskchain.task import Config

from rcpl import config

DEVICE = 'cuda'

config_path = config.CONFIGS_DIR / sys.argv[1]

conf = Config(
    config.TASKS_DIR,  # where Taskchain data should be stored
    config_path,
    context={'device': DEVICE},
    global_vars=config,  # set global variables
)
chain = conf.chain()
chain.set_log_level('CRITICAL')

print(chain.dataset_info.value)

print(chain.train_model)
_ = chain.train_model.force().value
# input("Press Enter to continue...")

print("Calculating metrics for ", sys.argv[1])
_ = chain.model_metrics.force().value
print(_)
