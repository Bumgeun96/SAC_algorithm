import argparse
import gymnasium as gym
from gymnasium.utils.save_video import save_video
import torch
import numpy as np
import random
from datetime import datetime
import pickle
from sac_agent import Agent

parser = argparse.ArgumentParser(description="")
# parser.add_argument("-env", type=str,default="Walker2d-v3", help="Environment name")
parser.add_argument("-env", type=str,default="Hopper-v3", help="Environment name")
parser.add_argument("-info", type=str, help="Information or name of the run")
parser.add_argument("-n_epi", type=int, default=5000000, help="The amount of training episodes, default is 5000")
parser.add_argument("-n_step", type=int, default=1000, help="The amount of steps per an episode, default is 1000")
parser.add_argument("-terminal_steps", type=int, help="The amount of training steps, default is none")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_value", type=float, default=3e-4, help="Learning rate of adapting the network weights, default is 1e-3")
parser.add_argument("-lr_critic", type=float, default=3e-4, help="Learning rate of adapting the network weights, default is 1e-3")
parser.add_argument("-lr_actor", type=float, default=3e-4, help="Learning rate of adapting the network weights, default is 1e-3")
parser.add_argument("-fixed_alpha", type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-memory_size", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-tau", type=float, default=0.005, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("-reward_scale", type=float, default=5, help="reward scale, default is 5")
args = parser.parse_args()

def train(env, agent, n_episodes, max_step,training_steps):
    total_step = 0
    random_seed = random.randint(0,200)
    env.reset(seed = random_seed)
    Eval = False
    if training_steps == None:
        n_episodes = 999999999
    for epi in range(1,n_episodes+1):
        state = env.reset()
        state = state[0].reshape((1,state_size))
        score = 0
        for step in range(1,max_step+1):
            total_step += 1
            action = agent.act(state).numpy()
            action = np.clip(action*action_high, action_low, action_high)
            next_state, reward, done, info,_ = env.step(action)
            if step == max_step:
                done = True
            next_state = next_state.reshape((1,state_size))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if total_step % 1000 == 0:
                torch.save(agent.actor.state_dict(),"./model/("+args.env+")actor.pt")
                print('========================================')
                now = datetime.now()
                print('[',now.hour,':',now.minute,':',now.second,']','steps:',total_step + step)
                Eval = True
            if done:
                # try:
                #     print('returns:',score[0],'step:',step)
                # except:
                #     print('returns:',score,'step:',step)
                if Eval:
                    eval(random_seed)
                    Eval = False
                break
        if total_step == training_steps:
            break
def eval(seed):
    eval_env = gym.make(env_name,render_mode="rgb_array_list")
    eval_agent.actor.load_state_dict(torch.load("./model/("+args.env+")actor.pt"))
    state=eval_env.reset(seed = seed)[0].reshape((1,state_size))
    returns = 0
    step = 0
    while True:
        step += 1
        action = eval_agent.deterministic_act(state).numpy()
        action = np.clip(action*action_high, action_low, action_high)
        next_state, reward, done, info,_ = eval_env.step(action)
        next_state = next_state.reshape((1,state_size))
        state = next_state
        returns += reward
        if done or step == 1000:
            RETURNS.append(returns)
            print('returns:',returns,'step:',step)
            with open('./figures/returns_'+args.env+'(seed:'+str(args.seed)+').pickle',"wb") as fw:
                    pickle.dump(RETURNS,fw)
            # save_video(eval_env.render(),
            #            "videos",
            #            fps=eval_env.metadata["render_fps"],
            #            step_starting_index=step-1)
            eval_env.close()
            break
        
def random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
    env = gym.make(env_name)
    if env_name.split('-')[0] == 'Hopper' or env_name.split('-')[0] == 'Walker2d':
        args.terminal_steps = 1000000
        args.reward_scale = 5
    elif env_name.split('-')[0] == 'HalfCheetah' or env_name.split('-')[0] == 'Ant':
        args.terminal_steps = 3000000
        args.reward_scale = 5
    elif env_name.split('-')[0] == 'Humanoid':
        args.terminal_steps = 10000000
        args.reward_scale = 20
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    random_seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    RETURNS = []
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,config=args)
    eval_agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,config=args)
    train(env,agent,args.n_epi,args.n_step,args.terminal_steps)
    env.close()