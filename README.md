# Universal Effectiveness of High-Depth Circuits in Variational Eigenproblems

This repository provides the source codes to reproduce the results of [arXiv:2010.00157](https://arxiv.org/abs/2010.00157),
which explores the effectiveness of high-depth, noiseless, parameteric quantum circuits by challenging their capability 
to simulate the ground states of quantum many-body Hamiltonians.
Please refer our paper to find the details of experimental results and discussions. 


## Prerequisites

The code mainly depends on `jax` and `qutip`. 
We lists the environments that our experiment performed on.
 - python 3.7
 - CUDA 10.1
 - Python Packages:
   + `jax==0.1.75`
   + `jaxlib==0.1.52`
   + `qutip==4.5.2`
   + `wandb==0.9.6`
   
The required python packages are listed in `requirements.txt` and you can install the packages with the following command.
```bash
  $ pip install -r requirements.txt
``` 

Basically the experimental results are managed through `wandb` package.
Please refer the [installation guide](https://docs.wandb.ai/quickstart) for setting up the wandb. 


## How to Run

There are three main experiments to evaluate the circuit's capability.
  - `expressibility.py`: Random states.
  - `ising_model.py`: Ground state of Ising model.
  - `SYK4_model.py`: Ground state of Sachdev-Ye-Kitaev (SYK) model.

Also, there are the barren plateau phenomena experiments for Ising and SYK models.
  - `ising_bp.py`: Ground state of Ising model.
  - `SYK4_bp.py`: Ground state of Sachdev-Ye-Kitaev (SYK) model.

To run the experiments, simply run the script below.  
```bash
  $ python expressibility.py --n-qubits 8 --n-layers 56 --lr 0.05 --train-steps 1000 --seed 1
  $ python ising_model.py --n-qubits 8 --n-layers 56 --g 2 --h 0 --lr 0.01 --train-steps 1000 --seed 1
  $ python SYK_hamiltonian.py --n-qubits 8 --n-layers 56 --lr 0.1 --train-steps 3000 --seed 1 --seed-SYK 1
  $ python ising_bp.py --n-qubits 8 --max-n-layers 300 --g 2 --h 0 --sample-size 1000 --seed 1
  $ python SYK_bp.py --n-qubits 8 --max-n-layers 300 --sample-size 100 --seed 1 --seed-SYK 1
```
Here, it is necessary to provide the seed value for the reproducibility.
You can find two different seeds from the options. 
`--seed-SYK` is for determining the SYK model and `--seed` is for all the other random values. 
After running, you can find the resulting files under the experiment directory `./results/{datetime}_{exp_name}`.
