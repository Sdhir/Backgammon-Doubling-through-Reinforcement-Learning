# Generate train data - Board positions
# Run this file in GNU BG CLI
# > load python gen_train_data.py

from gnubg import *
import json
import csv

game_vec = []

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
    #command("export match text E:/adp/backgammon/files/sample/match_data.txt")
    len_games = len(match_data['games'][-1]['game'])
    #print "Match: {}, \nlength of game: {}".format(match_i+1,len_games) 

    for i in range(len_games):
        if i>=4:
            # Action = {'move','double','take','drop','resign'}
            action = match_data['games'][-1]['game'][i]['action']
            if str(action) == 'move':
                # Player = {'x','o'}
                player = match_data['games'][-1]['game'][i]['player']
                # Get ASCII board position info
                board_str = match_data['games'][-1]['game'][i]['board']
                # Decode to tuple
                board_tuple = positionfromid(board_str)
                """
                # my point of view
                if str(player)=='O':
                    bg = list(board_tuple[0])
                    sd = list(board_tuple[1])
                    sd.reverse()
                    #print sd-bg
                    in_vec = [y-x for x,y in zip(bg[:-1],sd[1:])]
                    in_vec = [sd[0]]+in_vec+[bg[-1]]
                    game_vec.append(in_vec)
                    #print in_vec
                """
                # Opponent offers double
                # opponent point of view
                # Data format <0,1-24,25>
                if str(player)=='X':
                    # sd - x, bg - o
                    sd = list(board_tuple[0])
                    bg = list(board_tuple[1])
                    sd.reverse()
                    in_vec = [y-x for x,y in zip(bg[:-1],sd[1:])]
                    in_vec.reverse()
                    in_vec = [-1*bg[-1]]+in_vec+[sd[0]]
                    game_vec.append(in_vec)
                    #print in_vec

# Save data to csv                
with open('E:/adp/backgammon/files/train_board_pos.csv', 'a') as myfile:
    for game_i in game_vec:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(game_i)

