# When Curvature Counts: Hyperbolic Geometry in Prototype-Based Image Classification


This repository is released for reproducibility. It contains the official implementation of **HypPNet**, the hyperbolic prototype learning model presented in the paper *"When Curvature Counts: Hyperbolic Geometry in Prototype-Based Image Classification"*.


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

The configuration files in configs allow for easy and immediate reproducibility. In case a dataset among cub2011 or aircraft is not downloaded yet, please run the respective python file. 

Once the dataset is downloaded, the command to run an experiment is:
```
python main.py -config configs/config.yaml -device cpu
```

