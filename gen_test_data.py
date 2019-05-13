# Generate test data - doubling board positions
# Run this file in GNU BG CLI
# > load python gen_test_data.py

from gnubg import *
import json
import csv

double_game_vec = []
double_take_drop = []

num_matches = 500
for match_i in range(num_matches):
    # Start game
    command("new game")
    # Auto play and end the game
    command("end game")
    
    # Get all the games info
    match_data = match()
    """
    with open('E:/adp/backgammon/files/sample/match_data.json', 'w') as outfile:
        json.dump(match_data, outfile)
    """
    #command("export match text E:/adp/backgammon/files/match_data.txt")
    len_games = len(match_data['games'][-1]['game'])
    #print "Match: {}, \nlength of game: {}".format(match_i+1,len_games) 

    for i in range(len_games):
        # Action = {'move','double','take','drop','resign'}
        action = match_data['games'][-1]['game'][i]['action']
        # Player = {'x','o'}
        player = match_data['games'][-1]['game'][i]['player']
        #print(player,action)
        if str(action) == 'double':
            if str(player)=='X':
                # Get ASCII board position info
                board_str = match_data['games'][-1]['game'][i]['board']
        # Accept double
        elif str(action) == 'take':
            # Decode to tuple
            board_tuple = positionfromid(board_str)
            # Opponent offers double
            # opponent point of view
            # Data format <label,0,1-24,25>
            if str(player)=='O':
                # sd - x, bg - o
                sd = list(board_tuple[0])
                bg = list(board_tuple[1])
                sd.reverse()
                in_vec = [y-x for x,y in zip(bg[:-1],sd[1:])]
                in_vec.reverse()
                in_vec = [1]+[-1*bg[-1]]+in_vec+[sd[0]]
                double_game_vec.append(in_vec)
        # Reject double
        elif str(action) == 'drop':
            # Decode to tuple
            board_tuple = positionfromid(board_str)
            # Opponent offers double
            # opponent point of view
            # Data format <label,0,1-24,25>
            if str(player)=='O':
                # sd - x, bg - o
                sd = list(board_tuple[0])
                bg = list(board_tuple[1])
                sd.reverse()
                in_vec = [y-x for x,y in zip(bg[:-1],sd[1:])]
                in_vec.reverse()
                in_vec = [0]+[-1*bg[-1]]+in_vec+[sd[0]]
                double_game_vec.append(in_vec)
            
# Save data to csv            
with open('E:/adp/backgammon/files/test_double.csv', 'a') as myfile:
    for game_i in double_game_vec:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(game_i)


