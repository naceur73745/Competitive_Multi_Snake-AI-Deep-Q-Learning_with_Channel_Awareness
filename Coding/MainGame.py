import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy  as np
import torch 
import os 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time  
import numpy as np 
import pandas as pd 
from MultiAgent import Agent 
import math



#some attributes  

pygame.init()
font = pygame.font.SysFont("comicsans" , 50)
#font = pygame.font.SysFont('arial', 25)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BLOCK_SIZE = 30
SPEED = 100
FPS = 100
NUM_APPLES = 5



#the game  class 
class SnakeGame:

    def __init__(self, w=600, h=600, num_snakes=1):
        self.w = w
        self.h = h
        self.background_image = pygame.transform.scale(pygame.image.load("Brain.jpg"), (self.w, self.h))
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.num_snakes = num_snakes
        self.snakes = []
        self.frames_since_last_action = [0] * self.num_snakes 
        self.MAX_FRAMES_INACTIVITY = 1000  
        self.start_time = []
        self.snake_colors = [
            "white",
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
            "orange",
            "gray"
        ]

        self.reset()


    def reset_snake(self, snake_index):

        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

        self.heads[snake_index] = Point(x, y)
        self.snakes[snake_index] = [self.heads[snake_index], 
                      Point(self.heads[snake_index].x - BLOCK_SIZE, self.heads[snake_index].y), Point(self.heads[snake_index].x - (2 * BLOCK_SIZE), self.heads[snake_index].y)]
        self.directions[snake_index] =  Direction.RIGHT
        self.score[snake_index] = 0
        self.game_over[snake_index] = False
        self.start_time[snake_index] = time.time() 
        self.Apple_EatenSnakes[snake_index] = [0,0]

    def reset(self):
        self.frame_iteration = 0
        for  _ in range(self.num_snakes) : 
            self.start_time.append(time.time()) 
        #each snake will gonna take direction 
        self.game_over = [False] * (self.num_snakes)
        self.score = [0] * (self.num_snakes)
        self.Apple_EatenSnakes = [[0,0]]*(self.num_snakes)
        self.directions = [Direction.RIGHT for _ in range(self.num_snakes)]
        #each snake wil gonan have a body 
        self.heads = []  # List to store snake head positions
        self.snakes = []  # List to store snake body segments
        # Generate random starting positions for each snake
        for _ in range( self.num_snakes):

            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            head = Point(x, y)
            self.heads.append(head)
            snake = [head, Point(head.x - BLOCK_SIZE, head.y), Point(head.x - (2 * BLOCK_SIZE), head.y)]
            self.snakes.append(snake)

        self.food = []  # List to store apple positions
        self._place_food()  # Generate initial apples
 
    def get_state(self):
        states = []

        for snake_index in range(self.num_snakes):

            head = self.heads[snake_index]  # Access the head of the snake
            # All the pixels in the vicinity of the snake

            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)
            
            # Check which direction our snake has taken
            dir_l = self.directions[snake_index] == Direction.LEFT
            dir_r = self.directions[snake_index] == Direction.RIGHT
            dir_u = self.directions[snake_index] == Direction.UP
            dir_d = self.directions[snake_index] == Direction.DOWN


            # Collect positions, lengths, and actions of other snakes

            opponent_positions = []
            opponent_lengths = []
            opponent_actions = []

            for snake_idx in range(self.num_snakes):

                if snake_idx != snake_index:

                    opponent_positions.append(self.heads[snake_idx])
                    opponent_lengths.append(len(self.snakes[snake_idx]))  # Append the length of each opponent snake
                    opponent_action = [0, 0, 0, 0]  # Default action: no movement
                    if self.directions[snake_idx] == Direction.LEFT:
                        opponent_action = [1, 0, 0, 0]
                    elif self.directions[snake_idx] == Direction.RIGHT:
                        opponent_action = [0, 1, 0, 0]
                    elif self.directions[snake_idx] == Direction.UP:
                        opponent_action = [0, 0, 1, 0]
                    elif self.directions[snake_idx] == Direction.DOWN:
                        opponent_action = [0, 0, 0, 1]
                    opponent_actions.append(opponent_action)

            state = [
                #
                int((dir_r and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_d)))),
                # Danger right
                int((dir_u and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_d and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_d)))),
                # Danger left
                int((dir_d and (self.CheckForGetState(snake_index, point_r))) or
                    (dir_u and (self.CheckForGetState(snake_index, point_l))) or
                    (dir_r and (self.CheckForGetState(snake_index, point_u))) or
                    (dir_l and (self.CheckForGetState(snake_index, point_d)))),

                # Move direction
                int(dir_l),
                int(dir_r),
                int(dir_u),
                int(dir_d)
            ]
            
            # Compare snake lengths with other snakes
            my_length = len(self.snakes[snake_index])

            for opponent_length in opponent_lengths:

                if my_length > opponent_length:

                    state += [1, 0]  # Snake length is greater than opponent
                elif my_length < opponent_length:
                    state += [0, 1]  # Snake length is smaller than opponent
                else:
                    state += [0, 0]  # Snake length is equal to opponent

            for food_item in self.food:
                state += [
                    # Food location
                    int(food_item.x < head.x),   # food left
                    int(food_item.x > head.x),   # food right
                    int(food_item.y < head.y),   # food up
                    int(food_item.y > head.y)    # food down
                ]

            states.append(np.array(state, dtype=int))
        
        return states


    
    def CheckForGetState(self, snake_index, pt=None):
        if pt is None:
            pt = self.heads[snake_index]

        # Hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # Hits itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True

        # Hits other snakes
        for i, snake in enumerate(self.snakes):
            if i != snake_index:
                if pt in [body for body in snake]:
                    return True

        return False

    

    def _place_food(self):

        self.food = []
        for _ in range(NUM_APPLES):  # NUM_APPLES is the desired number of apples
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food.append(Point(x, y))
   
                            
    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
 
    def play_step(self, actions):

        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        snake_info = [[] for _ in range(self.num_snakes)]
        total_time_played = [[] for _ in range(self.num_snakes)]
        rewards = [0] * self.num_snakes
        self._move(actions)
        AppleDistanceList = []
        DistanceToSnakesList = []

        for snake_index in range(self.num_snakes):
            AppleDistance = []
            DistanceToSnakes = []

            for i, food in enumerate(self.food):
                apple_distance = self.calculate_distance(self.snakes[snake_index][0], self.food[i])
                AppleDistance.append(apple_distance)

            AppleDistanceList.append(AppleDistance)

            for i, snake in enumerate(self.snakes):
                if i != snake_index:
                    snake_distance = self.calculate_distance(self.snakes[snake_index][0], self.snakes[i][0])
                    DistanceToSnakes.append(snake_distance)

            DistanceToSnakesList.append(DistanceToSnakes)

            elapsed_time = time.time() - self.start_time[snake_index]
            snake_head = self.heads[snake_index]

            if self.collsion_wall(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -60
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index] = "I collided with the wall"
            elif self.collison_with_itself(snake_index):
                total_time_played[snake_index] = elapsed_time
                self.game_over[snake_index] = True
                rewards[snake_index] = -50
                self.frames_since_last_action[snake_index] = 0
                snake_info[snake_index] = "I collided with myself"
            else:
                self.snakes[snake_index].insert(0, snake_head)

                if snake_head in self.food:
                    self.score[snake_index] += 1
                    self.Apple_EatenSnakes[snake_index][1] += 1
                    for i, apple in enumerate(self.food):
                        if snake_head == apple:
                            rewards[snake_index] = 300
                            # Reward for eating an apple
                            self.food.pop(i)  # Remove the eaten apple
                            self._place_food()  # Generate a new apple to replace the eaten one
                            break
                    self.frames_since_last_action[snake_index] = 0
                    snake_info[snake_index] = "I ate an apple, yum!"
                else:
                    self.snakes[snake_index].pop()
                    self.frames_since_last_action[snake_index] += 1
                    snake_info[snake_index] = "I'm exploring the environment"

                    # Encourage collisions with other snakes
                    for other_snake_index in range(self.num_snakes):
                        if snake_index != other_snake_index:
                            if snake_head in self.snakes[other_snake_index]:
                                if len(self.snakes[snake_index]) <= len(self.snakes[other_snake_index]):
                                    rewards[snake_index] -= 70  # Encourage collision with longer snakes
                                    snake_info[snake_index] = "Another snake ate me!"
                                else:
                                    rewards[snake_index] += 70  # Encourage collision with shorter snakes
                                    snake_info[snake_index] = "I ate a snake!"
                                    self.score[index]+=1 

                    total_time_played[snake_index] = elapsed_time

                    # Check if nothing has happened for a long time
                    if self.frames_since_last_action[snake_index] >= self.MAX_FRAMES_INACTIVITY:
                        self.game_over[snake_index] = True
                        rewards[snake_index] = -10
                        self.reset_snake(snake_index)
                        self.frames_since_last_action[snake_index] = 0
                        snake_info[snake_index] = "I didn't do anything for n iterations"

        return rewards, self.game_over, self.score, snake_info, total_time_played, self.Apple_EatenSnakes , DistanceToSnakesList,AppleDistanceList





    def is_collision(self, snake_index, pt=None):

        if pt is None:

            pt = self.heads[snake_index]

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True  
        # hits itself
        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:
            return True
        return False
    
    def collsion_wall (self , snake_index , pt=None): 
        if pt is None:
            pt = self.heads[snake_index]
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True  
     
        return False
    
    def collison_with_itself (self, snake_index , pt=None): 
        if pt is None:
            pt = self.heads[snake_index]
        # hits itself

        if pt in [body for i, body in enumerate(self.snakes[snake_index][1:])]:

            return True



    def eate_other_snake(self, snake_index):

        head = self.heads[snake_index]
        truth = False 
        collided  = []
        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                #get the index of the hitted snake 
                collided.append(i)
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])
                if current_snake_length > collided_snake_length:
                    truth = True 
        return   truth  ,  collided
    
    def eaten_by_other_snake (self , snake_index ) : 
        head = self.heads[snake_index]
        for i, snake in enumerate(self.snakes):
            if i != snake_index and head in snake:
                collided_snake_length = len(snake)
                current_snake_length = len(self.snakes[snake_index])
                if current_snake_length <= collided_snake_length:
                    return True
        return False
    

    def grid (self) :
        for row in range(0  ,self.h , BLOCK_SIZE) :
            for col in range(0 , self.h , BLOCK_SIZE) :
                #draw rect
                rect = pygame.Rect(row, col, BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(self.display, "green", rect, 3 )
        pygame.display.update()



    def _update_ui(self):

        self.display.fill((0, 0, 0))
        self.display.blit(self.background_image, (0, 0))

        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, 'Red', (0, y), (self.w, y), 1)

        for snake_index in range(self.num_snakes):
            snake_color = self.snake_colors[snake_index]
            for point in self.snakes[snake_index]:
                pygame.draw.rect(
                    self.display,
                    snake_color,
                    pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE)
                )

            for i, food in enumerate(self.food):
                food_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                pygame.draw.rect(
                    self.display,
                    food_color,
                    pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE)
                )

            if self.game_over[snake_index]:
                font = pygame.font.Font(None, 50)
                text = font.render("Game Over", True, (255, 255, 255))
                self.display.blit(text, (self.w // 2 - text.get_width() // 2, self.h // 2 - text.get_height() // 2))

            score_font = pygame.font.Font(None, 36)
            score_text = score_font.render("Score: " + str(self.score[snake_index]), True, (255, 255, 255))
            self.display.blit(score_text, (10, 10 + 40 * snake_index))

        pygame.display.flip()
        self.clock.tick(FPS)


    def handle_user_input(self):

        #the action taken by the  human agent  

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return [0, 0, 1]  # Left turn action
        elif keys[pygame.K_DOWN]:
            return [0, 1, 0]  # Right turn action
        else:
            return [1, 0, 0]  # No change action


    def _move(self, actions):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        for snake_index in range(self.num_snakes):

            idx = clock_wise.index(self.directions[snake_index])
            if np.array_equal(actions[snake_index], [1, 0, 0]):
                new_dir = clock_wise[idx]  # no change
            elif np.array_equal(actions[snake_index], [0, 1, 0]):
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d
            self.directions[snake_index] = new_dir
            x = self.heads[snake_index].x
            y = self.heads[snake_index].y
            if self.directions[snake_index] == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.directions[snake_index] == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.directions[snake_index] == Direction.UP:
                y -= BLOCK_SIZE
            self.heads[snake_index] = Point(x, y)


def Create_agent(input_dim ,dim1 ,dim2,  n_actions , lr  ,butch_size , mem_size , gamma  ): 
  
  return Agent(input_dim ,dim1 , dim2,  n_actions , lr  ,butch_size , mem_size , gamma  )

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

if __name__ == '__main__':

    number_of_snakes   =  9  

    current_max = 0

    game = SnakeGame(num_snakes=number_of_snakes)

    agent = Agent(input_dimlsit=[43, 43 , 43,43, 43 , 43,43, 43 , 43], fc1_dimlsit=[256, 100,50,256, 100,50,256, 100,50], fc2_dimlist=[256, 256,256,256, 256,256,256, 256,256],
                fc3_dimlist=[256, 256,256,256, 256,256,256, 256,256], fc4_dimlist=[256, 256,256,256, 256,256,256, 256,256], n_actions=3, losslist=[nn.MSELoss() , nn.MSELoss(), nn.MSELoss(),nn.MSELoss() , nn.MSELoss(), nn.MSELoss(),nn.MSELoss() , nn.MSELoss(), nn.MSELoss()], lrlist=[0.001, 0.001,0.01,0.001, 0.001,0.01,0.001, 0.001,0.01],
                batch_size=10, mem_size=10000, gamma_list=[0.99, 0.75, 0.62,0.99, 0.75, 0.62 ,0.99, 0.75, 0.62  ] , num_agents=9)

    # Run the game loop
    running = True
    step  =  [0]*game.num_snakes 

    TotalPlayerScorePro = [0]*game.num_snakes 
    Total_PlayedTime = [0]*game.num_snakes 
    TotalTimeBeforeDeath = [0]*game.num_snakes 
    TotalSnakeEaten   = [0]*game.num_snakes 
    TotalAppleEaten = [0]*game.num_snakes 

    TotalEatenSnakes = []
    for   _ in range(game.num_snakes) : 
        TotalEatenSnakes.append([])
    TotalEatenApples = []
    for   _ in range(game.num_snakes) : 
        TotalEatenApples.append([])
    Total_score_list = []
    for   _ in range(game.num_snakes) : 
        Total_score_list.append([])
    Total_Time_List  = []
    for   _ in range(game.num_snakes) : 
        Total_Time_List.append([])
    DataFrames = []
    BestPerformance = [0]*game.num_snakes
    for agent_idx in range(number_of_snakes):
        data = {
            f'n_games{agent_idx}': [],
            f'playerScoreProRound{agent_idx}': [],
            f'playerTotalScore{agent_idx}': [],
            f'PlayedTimeBeforeDeath{agent_idx}': [],
            f'TotalPlayedTimeBeforeDeath{agent_idx}': [],
            f'MeanScore{agent_idx}': [],
            f'TimeOverScore{agent_idx}': [],
            f'TotalNumberofDeath{agent_idx}': [],
            f'TimeOverDeath{agent_idx}': [],
            f'Epsilon{agent_idx}': [],
            f'SnakeEatenProRound{agent_idx}': [],
            f'ApplesEatenProRound{agent_idx}': [],
            f'TotalSnakeEaten{agent_idx}': [],
            f'TotalApplesEaten{agent_idx}': [],
            f'SnakeEatenOverScore{agent_idx}': [],
            f'ApplesEatenOverScore{agent_idx}': [],
            f'CurrentState{agent_idx}': [],
        }
        DataFrames.append(data) 

    i = 0 

    while i <100 :
        
        old_states = game.get_state()
        actions = agent.choose_action(old_states)

        rewards, game_over, scores,  info  ,time_played ,apple_snake , DistanceToSnakesList ,DistanceToAppleList= game.play_step(actions)
        print(f" info : {DistanceToAppleList}")
        states_new = game.get_state()
        agent.short_mem(old_states, states_new, actions, rewards, game_over)

       
  
        if any(game_over) and not all(game_over):

            indices = [index for index, value in enumerate(game_over) if value == True]     
            for index in indices:

                step[index]+=1 
                TotalPlayerScorePro[index]   = TotalPlayerScorePro[index] + scores[index]
                Total_PlayedTime[index] = time_played[index]+ Total_PlayedTime[index]
  
                TotalSnakeEaten[index]  +=  apple_snake[index][1]
                TotalAppleEaten[index] +=  apple_snake[index][0]
                DataFrames[index][f'n_games{index}'].append(step[index])  
                DataFrames[index][f'CurrentState{index}'].append(info[index])  
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index]/ step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'PlayedTimeBeforeDeath{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalPlayedTimeBeforeDeath{index}'].append(Total_PlayedTime[index])
                if TotalPlayerScorePro[index]  > 0 :
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index]/TotalPlayerScorePro[index])
                else  : 
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])   
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TimeOverDeath{index}'].append(Total_PlayedTime[index]/step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'SnakeEatenProRound{index}'].append(apple_snake[index][0])
                DataFrames[index][f'ApplesEatenProRound{index}'].append(apple_snake[index][1])
                DataFrames[index][f'TotalSnakeEaten{index}'].append(TotalSnakeEaten[index])
                DataFrames[index][f'TotalApplesEaten{index}'].append(TotalAppleEaten[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'SnakeEatenOverScore{index}'].append(TotalSnakeEaten[index])
                else  : 
                    DataFrames[index][f'SnakeEatenOverScore{index}'].append(TotalSnakeEaten[index]/TotalPlayerScorePro[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'ApplesEatenOverScore{index}'].append(TotalAppleEaten[index])
                else  : 
                    DataFrames[index][f'ApplesEatenOverScore{index}'].append(TotalAppleEaten[index]/TotalPlayerScorePro[index])
                   
                game._update_ui()
                game.clock.tick(SPEED)
                game.reset_snake(index)

            '''   
            for  agent_index  in range(game.num_snakes) :
               if   BestPerformance[agent_index] < scores[agent_index] : 
                   BestPerformance[agent_index] = scores[agent_index] 
                   agent.save(agent_index)
            '''
            
        

        elif all(game_over)  : 
            indices = [index for index, value in enumerate(game_over) if value == True]   
            for  index  in indices  :
                step[index]+=1 
                TotalPlayerScorePro[index]+= scores[index]
    
                Total_PlayedTime[index] += time_played[index]
     
                TotalSnakeEaten[index]  +=  apple_snake[index][1]
                TotalAppleEaten[index] +=  apple_snake[index][0]
                DataFrames[index][f'n_games{index}'].append(step[index])  
                DataFrames[index][f'CurrentState{index}'].append(info[index])  
                DataFrames[index][f'MeanScore{index}'].append(TotalPlayerScorePro[index]/ step[index])
                DataFrames[index][f'playerTotalScore{index}'].append(TotalPlayerScorePro[index])
                DataFrames[index][f'PlayedTimeBeforeDeath{index}'].append(time_played[index])
                DataFrames[index][f'playerScoreProRound{index}'].append(scores[index])
                DataFrames[index][f'TotalPlayedTimeBeforeDeath{index}'].append(Total_PlayedTime[index])
                if TotalPlayerScorePro[index]  > 0 :
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index]/TotalPlayerScorePro[index])
                else  : 
                    DataFrames[index][f'TimeOverScore{index}'].append(Total_PlayedTime[index])   
                DataFrames[index][f'TotalNumberofDeath{index}'].append(step[index])
                DataFrames[index][f'TimeOverDeath{index}'].append(Total_PlayedTime[index]/step[index])
                DataFrames[index][f'Epsilon{index}'].append(agent.agents[index]['epsilon'])
                DataFrames[index][f'SnakeEatenProRound{index}'].append(apple_snake[index][0])
                DataFrames[index][f'ApplesEatenProRound{index}'].append(apple_snake[index][1])
                DataFrames[index][f'TotalSnakeEaten{index}'].append(TotalSnakeEaten[index])
                DataFrames[index][f'TotalApplesEaten{index}'].append(TotalAppleEaten[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'SnakeEatenOverScore{index}'].append(TotalSnakeEaten[index])
                else  : 
                    DataFrames[index][f'SnakeEatenOverScore{index}'].append(TotalSnakeEaten[index]/TotalPlayerScorePro[index])
                if TotalPlayerScorePro[index] == 0  : 
                    DataFrames[index][f'ApplesEatenOverScore{index}'].append(TotalAppleEaten[index])
                else  : 
                    DataFrames[index][f'ApplesEatenOverScore{index}'].append(TotalAppleEaten[index]/TotalPlayerScorePro[index])
            '''
            for  agent_index  in range(game.num_snakes) :
               if  BestPerformance[agent_index] < scores[agent_index] : 
                   BestPerformance[agent_index] = scores[agent_index] 
                   agent.save(agent_index)

                if  i %  500  : 
                    agent.save(agent_index ,i )
            '''

            game._update_ui()
            game.clock.tick(SPEED)
            game.reset()
            agent.long_mem() 
         
        #save the mdoell each 500 iterations  
        for  agent_index  in range(game.num_snakes) :
                if  i %  500  : 
                    agent.save(agent_index ,i )

        i+=1 

        game._update_ui()

        game.clock.tick(SPEED)

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
       
    pygame.quit()


save_path = '/home/naceur/Desktop/bachelor_project/Project/MultiAiSnake/Coding/ExcelFiles'


for dataFrame_index in range(game.num_snakes):
    df = pd.DataFrame(DataFrames[dataFrame_index])
    file_path = os.path.join(save_path, f'Test{dataFrame_index}.csv')  # Vollständiger Dateipfad
    df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))
