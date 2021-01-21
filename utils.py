import os,json
from gen_net import configs
def save_params(configs, time_data):
    with open(os.path.join(configs['current_path'], 'training_data', '{}_{}.json'.format(configs['file_name'], time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(configs, file_name):
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def update_tensorboard(writer,epoch,env,agent,arrived_vehicles):
    env.update_tensorboard(writer,epoch)
    agent.update_tensorboard(writer,epoch)
    writer.add_scalar('episode/arrived_num', arrived_vehicles,
                        configs['max_steps']*epoch)  # 1 epoch마다
    writer.flush()