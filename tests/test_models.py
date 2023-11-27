import re

import numpy as np
import pytest
import torch
import yaml

from rcpl.config import DATA_DIR
from rcpl.cpl_models.maf import MAFCPLModelFactory
from rcpl.cpl_models.maftr import MAFTr, MAFTrCPLModelFactory
from rcpl.experiment import Experiment

GENERATED_DIR = DATA_DIR / 'epsp_stress' / 'generated'
MEASURED_DIR = DATA_DIR / 'epsp_stress' / 'measured'


FACTORIES = {
    'maftr_factory': MAFTrCPLModelFactory(**yaml.unsafe_load('''
          params_bound:
            k0: [15, 250]
            κ1: [100, 10000]
            κ2: [0.00666666, 0.033333]
            # κ3: [ 100, 1000 ]
            # κ4: [ 0.02, 0.05 ]
            # κ5: [ 10, 200 ]
            # κ6: [ 0.02, 0.05 ]
            c1: [ 1000, 50000 ]
            c2: [ 10, 5000 ]
            c3: [ 10, 2000 ]
            c4: [ 10, 2000 ]
            a1: [ 0.000000001, 200 ]
            a2: [ 0.000000001, 200 ]
            a3: [ 0.000000001, 200 ]
            a4: [ 0.000000001, 200 ]
            aL: [ 0, 500 ]
          apriori_distribution_params:
            a_sum_bound: [100, 400]
            log_shift: 100
            scale_c: 1
          model_kwargs:
            dim: 4
            kappa_dim: 2
    ''')),
    'maftr_factory2': MAFTrCPLModelFactory(**yaml.unsafe_load('''
          params_bound:
            k0: [15, 250]
            κ1: [100, 10000]
            κ2: [0.00666666, 0.033333]
            κ3: [ 100, 1000 ]
            κ4: [ 0.02, 0.05 ]
            κ5: [ 10, 200 ]
            κ6: [ 0.02, 0.05 ]
            c1: [ 1000, 50000 ]
            c2: [ 10, 5000 ]
            c3: [ 10, 2000 ]
            c4: [ 10, 2000 ]
            a1: [ 0.000000001, 200 ]
            a2: [ 0.000000001, 200 ]
            a3: [ 0.000000001, 200 ]
            a4: [ 0.000000001, 200 ]
            aL: [ 0, 500 ]
          apriori_distribution_params:
            a_sum_bound: [100, 400]
            log_shift: 100
            scale_c: 1
          model_kwargs:
            dim: 4
            kappa_dim: 6
    ''')),
    'maf_factory': MAFCPLModelFactory(**yaml.unsafe_load('''
      params_bound:
        k0: [15, 250]
        κ1: [100, 10000]
        κ2: [0.00666666, 0.033333]
        c1: [1000, 10000]
        c2: [50, 2000]
        c3: [50, 2000]
        c4: [50, 2000]
        a1: [0.000000001, 350]
        a2: [0.000000001, 350]
        a3: [0.000000001, 350]
        a4: [0.000000001, 350]
      apriori_distribution_params:
        a_sum_bound: [100, 400]
        log_shift: 100
        scale_c: 1
      model_kwargs:
        dim: 4
        kappa_dim: 2
    ''')),
}


@pytest.mark.parametrize("fac_name,fac", FACTORIES.items())
def test_torch(fac_name, fac):
    m_exp = Experiment(json_path=DATA_DIR / 'epsp_stress' / 'measured' / '2023-11-23.json')
    m_epsp = m_exp.get_signal_representation(('geom', 17, 20), ['epsp'])

    for i in range(1000):
        batch_size = 1#(i % 64) + 1
        theta_batch = [fac.make_random_theta() for _ in range(batch_size)]
        epsp_batch = [m_epsp + np.random.rand(np.prod(m_epsp.shape)) / 100000 for _ in range(batch_size)]

        np_stress = [fac.make_model(theta).predict_stress(epsp) for theta, epsp in zip(theta_batch, epsp_batch)]

        torch_model = fac.make_model(torch.from_numpy(np.stack(theta_batch)))
        torch_stress = torch_model.predict_stress_torch_batch(torch.from_numpy(np.stack(epsp_batch)))

        if np.mean(np.abs(np.array(np_stress) - torch_stress.detach().numpy())) > 0.05:
            np.save('np_stress.npy', np.array(np_stress))
            np.save('torch_stress.npy', torch_stress.detach().numpy())
            np.save('epsp_batch.npy', np.stack(epsp_batch))
            np.save('theta_batch.npy', np.stack(theta_batch))
        assert np.mean(np.abs(np.array(np_stress) - torch_stress.detach().numpy())) < 0.05, (i, np.mean(np.abs(np.array(np_stress) - torch_stress.detach().numpy())))


@pytest.mark.parametrize("exp_path", list(GENERATED_DIR.glob('*maftr*.json')))
def test_maftr_model(exp_path):
    measured_experiment_path = MEASURED_DIR / re.sub(r'\.maftr(\d+)', '', exp_path.name)
    m_exp = Experiment(json_path=measured_experiment_path)
    m_exp._load_json()
    exp = Experiment(json_path=exp_path)
    exp._load_json()
    num_model = MAFTr(theta=np.array(exp.meta['theta']), **exp.meta['model_kwargs'])

    stress = num_model.predict_stress(m_exp.get_signal_representation(exp.meta['representation'], ['epsp']))

    mean_abs_dif = np.mean(np.abs(stress - exp.get_signal_representation(exp.meta['representation'], ['stress'])))
    assert mean_abs_dif < 0.05, f'Mean absolute difference is {mean_abs_dif} for {exp_path.name}'
