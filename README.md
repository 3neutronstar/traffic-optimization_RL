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
Didn't check that it learns well. (Prototype)
- check the result
Tensorboard
```shell script
    tensorboard --logdir training_data
``` 
Hyperparameter in json, model is in `training_data/model` directory.

## Old version of Learning Process
In algorithm_train.py
### Learning Process
- Agent
Traffic Light System (Intersection)
- State
Vehicle Movement Demand(2spaces per each inEdge, total 8 spaces) <br/>
Phase (4 or 8 spaces) choose by `--phase [4 or 8]`<br/>

Total 12 or 16 state spaces <br/>

Only in FRAP model -> 16 state spaces

- Action
Phase (4 or 8 spaces) each 20s <br/>
After action, all yellow light turn on for 5s
