# LLM_activation_pruning
Code for pruning of input activations to speed up of LLMs

## Environment
To set up the environment, use the Docker container for PyTorch 2.6.0 with CUDA 12.4. 
Additionally, you need to install the following packages:
```
pip install transformers==4.51.3
pip install lm_eval==0.4.9
```
Install any other required packages as needed.

## Parameters
All parameters for the experiments are contained in the configuration file located at 
`./act_prune/config/config.yaml`

## Experiments
To conduct an experiment, simply run the code from the directory `./act_prune`:

```
python main.py
```
