# Proximal Policy Optimization with Adaptive Generalized Advantage Estimate: Critic-Aware Refinements  



This repository contains the official PyTorch implementation of our paper:

Proximal Policy Optimization with Adaptive Generalized Advantage Estimate: Critic-Aware Refinements  
Naemeh Mohammadpour, Meysam Fozi, Mohammad Mehdi Ebadzadeh, Ali Azimi, Ali Kamali  
Amirkabir University of Technology (AUT), 2025  

---

## Overview

Proximal Policy Optimization (PPO) is one of the most widely used reinforcement learning (RL) algorithms
due to its balance of stability and learning efficiency.However, PPO is highly sensitive to hyperparameters
such as λ in Generalized Advantage Estimation (GAE), which manages the bias-variance trade-off.

In this project, we propose two refinements:

1. Adaptive GAE (Dynamic λ scheduling)  
   - λ is adjusted dynamically based on critic loss trends.  
   - If the critic improves → λ decreases (reducing variance).  
   - If the critic worsens → λ increases (reducing bias).  
   - Eliminates costly manual hyperparameter tuning.  

2. Policy Update Delay (PUD)  
   - Inspired by TD3, the actor is updated less frequently than the critic.  
   - Improves training stability and reduces policy oscillations.  

Together, these refinements produce more adaptive, stable, and generalizable agents with minimal computational overhead.

---

## Features

- PPO implementation based on [CleanRL](https://github.com/vwxyzjn/cleanrl).  
- Dynamic λ adjustment for bias-variance control.  
- Delayed actor updates for improved stability.  
- Compatible with OpenAI Gymnasium and DeepMind Control Suite.  
- TensorBoard logging for training metrics.  
- Model saving and evaluation support.  

---

## Results

Evaluated on standard continuous control benchmarks:

- OpenAI Gym (MuJoCo): Ant-v4, Humanoid-v4, HalfCheetah-v4  
- DeepMind Control Suite: Quadruped-Walk  

Key findings:

- Adaptive GAE consistently improves stability and return.  
- PUD alone may underperform, but Adaptive GAE + PUD yields stronger results.  
- Significant improvements in Ant-v4 and Humanoid-v4.  
- In HalfCheetah-v4, critic instability limits λ’s adaptiveness.  



## Installation

Use the requirements from this repo:

```bash
pip install -r requirements.txt

````


## Usage

Run training with default settings (Humanoid-v4, 2M timesteps):

```bash
python ppo_adaptive.py
```

## Custom runs

```bash
# Run Ant-v4 for 4M timesteps
python ppo_adaptive.py --env_id Ant-v4 --total_timesteps 4000000  

# Save trained model
python ppo_adaptive.py --save_model True  

# Enable TensorBoard logging
tensorboard --logdir runs
```

---

## Logging

Metrics tracked with TensorBoard:

* Episodic return and length
* Policy loss, value loss, entropy
* KL divergence and clip fraction
* Explained Variance (EV)
* Steps per second (SPS)

---

## Key Hyperparameters

| Hyperparameter    | Value / Range | Description                     |
| ----------------- | ------------- | ------------------------------- |
| learning\_rate    | 3e-4          | Adam optimizer LR               |
| gamma             | 0.99          | Discount factor                 |
| gae\_lambda       | \[0.70, 0.99] | Adaptively updated              |
| m                 | 0.09 → 0.005  | Step size for λ updates         |
| policy\_frequency | 2             | Critic updates per actor update |
| clip\_coef        | 0.2           | PPO clipping                    |
| vf\_coef          | 0.5           | Value loss coefficient          |
| max\_grad\_norm   | 0.5           | Gradient clipping               |

---

## Citation

If you use this code, please cite our work:

```bibtex
@article{mohammadpour2025ppo,
  title={Proximal Policy Optimization with Adaptive Generalized Advantage Estimate: Critic-Aware Refinements},
  author={Mohammadpour, Naemeh and Fozi, Meysam and Ebadzadeh, Mohammad Mehdi and Azimi, Ali and Kamali, Ali},
  journal={Journal of Modeling in Mechanics},
  year={2025},
  publisher={University of Guilan}
}
```

---

## Acknowledgments

* Based on [CleanRL](https://github.com/vwxyzjn/cleanrl).
* Inspired by [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347), [GAE (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438), and [TD3 (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477).

---


