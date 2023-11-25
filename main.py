from helpers import play_game,initialize_game,dynamic_programming

env,q_table = initialize_game(is_slippery=False)

q_table = dynamic_programming(env,q_table)

play_game(env,q_table)

env.close()
