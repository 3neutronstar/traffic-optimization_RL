# traffic-control_RL
Reinforcement Learning based traffic-control

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python run.py simulate
``` 
Run in RL algorithm DQN (default device: cpu)
```shell script
    python run.py train --gpu False
``` 
If you want to use other algorithm, use this code (ppo, REINFORCE, a2c) 

```shell script
    python run.py train --algorithm ppo --gpu False
``` 
Check the RL performance that based on FRAP model [FRAP Paper]https://arxiv.org/abs/1905.04722
```shell script
    python run.py train --model frap
``` 
- check the result
Tensorboard
```shell script
    tensorboard --logdir training_data
``` 
Hyperparameter in json, model is in `training_data/model` directory.
