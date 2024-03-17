# Cyclic Plastic Loading parameter estimation

This repository contains the code to estimate parameter of a material model
in the cyclic plastic loading. 

# Installation

```bash
pip install -e .
```

# Usage

Simple examples can be found in the `notebooks` directory.

### Custom model

This whole pipeline is not model specific and can be relatively
easily adapted to any other model. To use your custom model, you need to
do the following:

1. Adjust the `src/rcpl/material_model/custom_model.py` file to your needs.
2. Make a config yaml file based on `configs/dataset/custom_model/custom_model1.yaml`.
3. Make a config of a parameter estimation model based on `configs/custom_model/gru.yaml`.
4. Run the model training with the following command:

```bash
python scripts/train_model.py custom_model/gru.yaml
```
or alternatively, adjust the notebooks to your needs.

# Data

Processed data from 3 experiments is included in the `data/epsp_stress/measured` directory.
