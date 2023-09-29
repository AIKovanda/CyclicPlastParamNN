import json
import sys

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from taskchain.task import Config

from rcpl import config
from rcpl.config import BASE_DIR
from rcpl.rcpl import Experiment

DEVICE = 'cuda'

config_path = config.CONFIGS_DIR / 'model' / sys.argv[1]

conf = Config(
    config.TASKS_DIR,  # where Taskchain data should be stored
    config_path,
    context={'device': DEVICE},
    global_vars=config,  # set global variables
)
chain = conf.chain()
chain.set_log_level('CRITICAL')
# chain.train_model.force().value
net = chain.trained_model.value

torch_exp = torch.unsqueeze(torch.from_numpy(chain.real_experiment.value['2023-08-27']), 0).float().to(DEVICE)

net.eval()
with torch.no_grad():
    theta_hat = net(torch_exp)

if chain.model_info.value['takes_epsp']:
    experiment = Experiment(epsp=chain.real_experiment.value['2023-08-27'][1])
else:
    experiment = chain.experiment.value

sig = chain.get_random_pseudo_experiment.value(
    scaled_params=theta_hat[0].cpu().numpy(),
    experiment=experiment,
)[0]

figs_dir = BASE_DIR / 'out'
save_dir = figs_dir / chain.model_info.value["run_dir"]
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / f'{chain.model_info.value["run_name"]}_params.json', 'w') as f:
    json.dump(chain.unscale_params_torch.value(theta_hat).cpu().numpy().tolist()[0], f)

with sns.axes_style("darkgrid"):
    plt.figure(figsize=(20, 4))
    sns.lineplot(chain.real_experiment.value['2023-08-27'][0], label='Experimental data', color='red')
    # sns.lineplot(df_sim[:200]['sig'], label='sim')
    plt.xlabel('Pseudo time $t$')
    plt.ylabel('Tension stress $\\sigma$ (MPa)')
    sns.lineplot(sig, label='Simulation')
    plt.tight_layout()
    plt.savefig(save_dir / f'{chain.model_info.value["run_name"]}_exp_sim.pdf', bbox_inches='tight', pad_inches=0)

validate_crlb = chain.validate_crlb.value['2023-08-27']

with sns.axes_style('ticks'):
    with torch.no_grad():
        points_a = validate_crlb['crlb']['relative_std']
        labels = ['k_0', '\\kappa_1', '\\kappa_2', 'c_1', 'c_2', 'c_3', 'c_4', 'a_1', 'a_2', 'a_3', 'a_4']

        plt.figure(figsize=(10, 5))
        plt.plot(points_a, 'o', label='crlb')
        plt.plot(validate_crlb['theta_relative_std'], 'o', label='model_var')

        for i, (x, y) in enumerate(enumerate(points_a)):
            plt.text(x + 0.1, y, f'${labels[i]}$', fontsize=15, ha='left', va='center')

        plt.yscale('log')
        plt.title('STD / Theta for $\\sigma^2=1$')

        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.xlabel('Parameters')
        plt.ylabel('Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f'{chain.model_info.value["run_name"]}_crlb.pdf', bbox_inches='tight', pad_inches=0)
