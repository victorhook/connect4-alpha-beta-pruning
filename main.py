from time import time
import typing as t
import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

from student_code import student_move

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ['vi4672kr-s']

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = student_move(env, depth=3)
   #action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done


def play_game(vs_server = False, do_render: bool = False) -> float:
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """
   if vs_server:
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])

      # Botmove of -1 means we start.
      student_gets_move = botmove == -1

   else:
      state = env.reset(board=None)
      student_gets_move = random.choice([True, False])

   print('You start!') if student_gets_move else print('Bot starts!')

   done = False
   while not done:
      t0 = time()
      stmove = student_move(env, state)
      #print(f'Me     time:  {round(time() - t0, 3)} s')

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         t0 = time()
         res = call_server(stmove)
         #print(f'Server time:  {round(time() - t0, 3)} s')

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like
         if do_render:
            env.render(mode='human')

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            #print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
            pass
         return result
      else:
         pass
         #print("Current state (1 are student discs, -1 are servers, 0 is empty): ")


def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-g", "--gui", help = "Render gui", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   parser.add_argument("-r", "--rounds", help = "Number of runs to play", type=int, default=1)
   args = parser.parse_args()


   if args.online:
      print('Playing against the server')
   else:
      print('Playing local')

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   wins = 0
   ties = 0
   loss = 0
   errors = 0
   avg_time = 0
   for r in range(args.rounds):
      t0 = time()
      result = play_game(vs_server=args.online, do_render=args.gui)
      if result == 1:
         wins += 1
      elif result == 0.5:
         ties += 1
      elif result == -1:
         loss += 1
      elif result == -10:
         errors += 1

      dt = time() - t0
      avg_time += dt

   print(f'\n-- Games played: {r+1} --')
   print(f'wins: {wins}')
   print(f'ties: {ties}')
   print(f'loss: {loss}')
   print(f'errors: {errors}')
   print(f'average time per game: {round(avg_time/(r+1), 3)} s')
   print(f'--                   --')

   if args.stats:
      stats = check_stats()
      print(stats)


   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"



if __name__ == "__main__":
    main()
