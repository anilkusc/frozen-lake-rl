from helpers import play_game,initialize_game,dynamic_programming,monte_carlo

env,q_table = initialize_game(is_slippery=True)

q_table = monte_carlo(env,q_table,num_episodes=1000000)

play_game(env,q_table)

env.close()
