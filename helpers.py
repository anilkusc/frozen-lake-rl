import random
import numpy as np
import gym

def initialize_game(is_slippery=True):

    env = gym.make('FrozenLake-v1', render_mode="ansi",is_slippery=is_slippery)
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    return env,q_table

def find_action(epsilon,q_table,state,env):
     exp_tradeoff = random.uniform(0, 1)

     if exp_tradeoff > epsilon:
         # exploitation
         action = np.argmax(q_table[state, :])
     else:
         # exploration
         action = env.action_space.sample()
     return action

def play_game(env,q_table):
    wins = 0
    lost = 0
    max_steps = 1000
    for episode in range(1000):
        state = env.reset()[0]
        done = False
        print("****************************************************")
        print("EPISODE ", episode+1)
        for step in range(max_steps):
            #print(env.render())
            action = np.argmax(q_table[state,:])

            new_state, reward, done, truncated, info = env.step(action)

            if done:
                #print(env.render())
                if reward == 1:
                    print("****You reached the goal!****")
                    wins = wins + 1
                else:
                    print("****You fell through a hole!****")
                    lost = lost + 1
                break
            state = new_state
    print("Wins:",wins)
    print("Lost:",lost)

def dynamic_programming(env,q_table,
                        max_steps = 100,learning_rate = 0.2,discount_rate = 0.99,epsilon = 1,max_epsilon = 1,min_epsilon = 0.001,exploration_decay_rate = 0.001,num_episodes = 10000):
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()[0]
        done = False
        episode_rewards = 0

        for step in range(max_steps):
            action = find_action(epsilon,q_table,state,env)

            new_state, reward, done, truncated, info = env.step(action)
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
            episode_rewards += reward

            state = new_state
            if done == True:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)
    return q_table

def monte_carlo(env,q_table,
                        max_steps = 100,learning_rate = 0.7,discount_rate = 0.99,epsilon = 1,max_epsilon = 1,min_epsilon = 0.001,exploration_decay_rate = 0.001,num_episodes = 10000):
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()[0]
        done = False
        transitions = []
        global_reward = 0
        for step in range(max_steps):
            
            action = find_action(epsilon,q_table,state,env)

            new_state, reward, done, _, _ = env.step(action)
            transitions.append([state,action,reward])
            state = new_state
            global_reward = reward + discount_rate * global_reward
            if done:
                break
            
        average_reward = global_reward / len(transitions)
        for s,a,r in reversed(transitions):
            q_table[s, a] = q_table[s, a]* (1 - learning_rate) + learning_rate * average_reward
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)
        completion_percentage = (episode + 1) / num_episodes * 100
        print(f"Episode {episode + 1}/{num_episodes} - Completion: {completion_percentage:.2f}%")

    return q_table