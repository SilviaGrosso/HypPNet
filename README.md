# When Curvature Counts: Hyperbolic Geometry in Prototype-Based Image Classification


This repository is released for reproducibility. It contains the official implementation of **HypPNet**, the hyperbolic prototype learning model presented in the paper *"When Curvature Counts: Hyperbolic Geometry in Prototype-Based Image Classification"*.

All models were trained for `200 epochs` using `SGD` with a `learning rate` of `0.1`, `weight decay` of `0.001`, `momentum` of `0.9`, and a `linear learning rate scheduler` with decays at epochs `60`, `120`, and `160`. 
The backbone was a `ResNet18` trained from scratch. 
For HypPNet and ECL<sup>p</sup>, prototypes were updated using separate optimizers (`RSGD` for HypPNet, `SGD` for ECL<sup>p</sup>) with a `learning rate`of `0.001`, `momentum`of `0.9`, and `weight decay` set to `0`. 
We performed a hyperparameter tuning on the `temperature`, resulting in an overall best value of `0.1`, except for NF, for which the temperature was set to `0.08`. 
The regularization slope `lambda` was fixed to `0.1` for the smallest embedding dimension and `0.01` for larger ones, as it depends on the embedding dimensionality. 


## Installation

Before running the code, install the requirements.txt 

```
# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the packages from requirements.txt
pip install -r requirements.txt
```

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among CUB-2011 or Aircraft is not downloaded yet, please run the respective python file. 

Once the dataset is downloaded, the command to run an experiment is:
```
python main.py -config configs/config.yaml -device cpu
```

