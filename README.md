# traffic-control_RL (Discrete Action Space)
Reinforcement Learning based traffic-control

### Prerequisite
- python 3.7.9 above
- pytorch 1.7.1 above
- tensorboard 2.0.0 above

### How to use
check the condition state (throughput)
```shell script
    python ./Experiment/run.py simulate
``` 
Run in RL algorithm DQN (default device: cpu)
```shell script
    python ./Experiment/run.py train --gpu False
``` 
If you want to use other algorithm, use this code (ppo,super_dqn, ~~REINFORCE, a2c~~) 

```shell script
    python ./Experiment/run.py train --algorithm ppo
``` 
Check the RL performance that based on FRAP model [FRAP Paper]https://arxiv.org/abs/1905.04722
```shell script
    python ./Experiment/run.py train --model frap
``` 
Didn't check that it learns well. (Prototype)
- check the result
Tensorboard
```shell script
    tensorboard --logdir ./Experiment/training_data
``` 
Hyperparameter in json, model is in `./Experiment/training_data/[time you run]/model` directory.

- replay the model
```shell script
    python ./Experiment/run.py test --replay_name /replay_data in training_data dir/ --replay_epoch NUM
```


## New version of Learning Process(Discrete)
NxN intersecion
### Single Agent DQN
- Experiment
    1) Every 160s(depend on COMMON_PERIOD)
    2) Controls the each phase length that phases are in intersection system

- Agent
    1) Traffic Light Systems (Intersection)

- State
    1) Vehicle Movement Demand(in FRAP only) or Queue Length(2 spaces per each inEdge, total 8 spaces)for each end of phase: 4 phase==> 32spaces <br/>
    -> each number of vehicle is divided by max number of vehicles in an edge.(Normalize)
    2) Phase Length(If the number of phase is 4, spaces is composed of 4) <br/>
    -> (up,right,left,down) is divided by max period (Normalize)
    3) Searching method
        (1) Before phase ends, receive the inflow vehicles

- Action (per each COMMON_PERIOD of intersection)
    1) Tuple of +,- of each phases (13)
    2) Length of phase


- Reward
    1) Max Pressure Control Theory
    2) Penalty if phase exceeds its max length

### Decentralized DQN
- Experiment
    1) Every 160s(depend on COMMON_PERIOD)
    2) Controls the each phase length that phases are in intersection system

- Agents
    1) Traffic Light Systems (Intersection)
    2) Have their own offset value
    3) Update itself asynchronously (according to offset value and COMMON_PERIOD value)

- State
    1) Queue Length(2 spaces per each inEdge, total 8 spaces) <br/>
    -> each number of vehicle is divided by max number of vehicles in an edge.(Normalize, TODO)
    2) Phase Length(If the number of phase is 4, spaces are composed of 4) <br/>
    -> (up,right,left,down) is divided by max period (Normalize)
    3) Searching method
        (1) Before phase ends, receive all the number of inflow vehicles

- Action (per each COMMON_PERIOD of intersection)
    1) Tuple of +,- of each phases (13)
    2) Length of phase time changes
    -> minimum value exists and maximum value exists

- Next State
    1) For agent, next state will be given after 160s.
    2) For environment, next state will be updated every 1s.

- Reward
    1) Max Pressure Control Theory (Reward = -pressure=-(inflow-outflow))


## New version of Learning Process(Continuous)
NxN intersecion `Experiment` directory.
### Decentralized DDPG
- Experiment
    1) Every 160s(COMMON_PERIOD)
    2) Controls the phase length

- Agent
    1) Traffic Light System (Intersection)

- State
    1) Vehicle Movement Demand(in FRAP only) or Queue Length(2 spaces per each inEdge, total 8 spaces) <br/>
    -> each number of vehicle is divided by max number of vehicles in an edge.(Normalize)
    2) Phase Length(If the number of phase is 4, spaces are composed of 4) <br/>
    -> (up,right,left,down) is divided by max period (Normalize)

- Action (per each COMMON_PERIOD of intersection)
    1) Demand of each phase (in here 4 phase) -> multi-agent
    2) Between two phases, have 3 seconds for phase of all yellow movement signals. 

- Reward
    1) Max Pressure Control Theory
    2) Penalty if phase exceeds its max length


## Old version of Learning Process
3x3 intersection with singleagent and multiagent system `Discrete/` directory.

### How to use
check the condition state (throughput)
```shell script
    python ./Discrete/run.py simulate
``` 
Run in RL algorithm DQN (default device: cpu)
```shell script
    python ./Discrete/run.py train --gpu False
``` 
If you want to use other algorithm, use this code (ppo, ~~REINFORCE, a2c~~) 

```shell script
    python ./Discrete/run.py train --algorithm ppo --gpu False
``` 
Check the RL performance that based on FRAP model [FRAP Paper]https://arxiv.org/abs/1905.04722
```shell script
    python ./Discrete/run.py train --model frap
``` 
Didn't check that it learns well. (Prototype)
- check the result
Tensorboard
```shell script
    tensorboard --logdir ./Discrete/training_data
``` 
Hyperparameter in json, model is in `training_data/model` directory.

### Learning Process
- Agent
Traffic Light System (Intersection)
- State
Vehicle Movement Demand(2 spaces per each inEdge, total 8 spaces) <br/>
Phase (4 or 8 spaces) choose by `--phase [4 or 8]`<br/>

Total 12 or 16 state spaces <br/>

Only in FRAP model -> 16 state spaces

- Action
Phase (4 or 8 spaces) each 20s <br/>
After action, all yellow light turn on for 5s

## Utils
gen_tllogic.py
```shell script
python /path/to/repo/util/gen_tllogic.py --file [xml]
```
graphcheck.py
```shell script
python /path/to/repo/util/gen_tllogic.py file_a file_b --type [edge or lane] --data speed
```
    - check the tensorboard
    `tensorboard --logdir tensorboard`
