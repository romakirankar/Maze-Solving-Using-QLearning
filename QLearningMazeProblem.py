import numpy as np
import random
from tkinter import filedialog

class File():
    def __init__(self):
        self.action_read = "r"
        self.action_write = "w"
        self.output_file_name = "OutputMaze.txt"
        self.fileTypes = [("Text files", "*.txt")]
    
    def readFile(self):
        try:
            file_path_input = filedialog.askopenfilename(filetypes=self.fileTypes)
            if file_path_input:
                with open(file_path_input, self.action_read) as file_content:
                    self.maze_array = [list(map(int, line.strip())) for line in file_content]
                
                return self.maze_array
            
        except Exception as err:
            print(f"Some error occurred in uploading file: {err}")

    def writeFile(self, discovered_maze):
        try:
            with open(self.output_file_name, self.action_write) as file:
                for iRow in range(len(discovered_maze) - 1):
                    file.write(''.join(map(str, discovered_maze[iRow])) + '\n')
                file.write(''.join(map(str, discovered_maze[-1])))  # Remove the last newline
            
            print("Data was written successfully in file OutputMaze.txt")
        
        except Exception as e:
            print(f"Error occurred in saving: {e}")

    #converts the 2D grid represented by maze into a NumPy array
    def getNumpyArray(self):
        return np.array(self.maze_array)

class Qlearning():
    def __init__(self, maze, alpha, gamma, epsilon):
        self.maze = maze
        self.total_rows = maze.shape[0]
        self.total_cols =  maze.shape[1] 
        self.initial_state = (0,0)
        self.goal_state = (self.total_rows - 1 , self.total_cols - 1)
        self.actions = {'up': (-1, 0), 
                        'down': (1, 0), 
                        'left': (0, -1), 
                        'right': (0, 1)}
        self.q_table = {}
        self.episodes = 1000            #an episode is when an agent can no longer take a new action and ends up terminating
        self.alpha = alpha               #learning rate - influences the weight given to new information
        self.gamma = gamma                #discount factor - adjusts the importance of future rewards
        self.exp = epsilon                 #between exploration (choosing a random action) and exploitation (choosing the best-known action

        # Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        self.max_steps_per_episode = 100  # maximum steps allowed per episode to avoid infinite loop


    def initQTable(self):
        for row in range(0, self.total_rows):       #number of rows in a matrix/numpy array
            for col in range(0, self.total_cols):   #number of columns in a matrix/numpy array

                #for every open cell with value 0, set the q_table with its possible actions
                if maze[row, col] == 0:
                    self.q_table[(row, col)] = [0, 0, 0, 0]  #set actions in qtable= up, down, left and, right
    

    def setQTable(self):
        list_of_actions = ['up', 'down', 'left', 'right']

        for episode in range(0, self.episodes):
            #entrance cell is set to (0,0) for every episode
            current_state =  self.initial_state  
            current_step = 0

            while current_state != self.goal_state and current_step < self.max_steps_per_episode:

                #pick a random value from distribution - 0 to 1
                random_value = random.uniform(0, 1) 
                
                #explore if value is less than the threshold
                if random_value < self.exp:
                    action_key = random.choice(list_of_actions) #eg: 'up'
                
                #exploit if value is more than the threshold
                else:
                    #list of action values from qtable for the current state eg: [0.5, 0.6, 0.2, 0.8]
                    action_values_for_current_state = self.q_table[current_state]

                    #find index position of the action that has the maximum value eg: 3
                    index_action = np.argmax(action_values_for_current_state)

                    # Select an key action from dictionary with highest Q value eg: 'up'
                    action_key = list_of_actions[index_action]  
                
                #eg: (-1, 0 ) for action 'up'
                action_tuple = self.actions[action_key]  
                next_state = self.findNextState(current_state, action_tuple)
                reward = self.calculateReward(next_state)
                
                #invalid move, reassign the current state again to repeat the process to pick a new action at random
                if reward == -100: 
                    next_state = current_state 

                #initial value of the action reward of the current state in qtable
                old_value = self.q_table[current_state][list_of_actions.index(action_key)]

                #maximum value of the future reward for the next state 
                next_value = max(self.q_table[next_state])

                #re-calculate the reward for the current state using all the parameters (including next state's) and update
                self.q_table[current_state][list_of_actions.index(action_key)] = old_value + self.alpha * (reward + self.gamma * next_value- old_value)
                
                current_state = next_state
                current_step += 1

    def findNextState(self, current_state, action_tuple):
        #calculate the next state by adding up the action
        return (current_state[0] + action_tuple[0], current_state[1] + action_tuple[1])
    

    def calculateReward(self,  next_state):
        #current row is -1 or greater than total rows 
        # or current col is -1 or greater than total cols 
        # or the content of the state is 1 i.e a wall
        if  next_state[0] <  0 or next_state[0] >= self.total_rows or next_state[1] < 0 or next_state[1] >= self.total_cols or self.maze[next_state] == 1:
            return -100
        elif next_state == self.goal_state:
            return 100
        else: 
            #any other valid move to an available path
            return -1

    def getPath(self):
        list_of_actions = ['up', 'down', 'left', 'right']
        current_state = self.initial_state
        discovered_path = [current_state]
        current_step = 0

        while current_state != self.goal_state and current_step < self.max_steps_per_episode:
            #not a valid move to find the path
            if current_state not in self.q_table:
                break

            #list of action values from qtable for the current state eg: [0.5, 0.6, 0.2, 0.8]
            action_values_for_current_state = self.q_table[current_state]

            #find index position of the action that has the maximum value eg: 3
            index_action = np.argmax(action_values_for_current_state)

            # Select a key action from dictionary with highest Q value eg: 'up'
            action_key = list_of_actions[index_action]  

            #eg: (-1, 0 ) for action 'up'
            action_tuple = self.actions[action_key]  

            next_state = self.findNextState(current_state, action_tuple)

            # No valid moves
            if next_state == current_state: 
                return None
            
            current_state = next_state
            discovered_path.append(next_state)
            current_step += 1

        #replace 0 with 2 in the maze for the discovered path
        if current_state == self.goal_state:
            for row, col in discovered_path:
                self.maze[row, col] = 2
            return self.maze
        else:
            return None
        

File = File()
maze_array = File.readFile()

alpha = 0.1
gamma = 0.9
epsilon = 0.1

try:
    if maze_array:
        maze = File.getNumpyArray()
        QL = Qlearning(maze, alpha, gamma, epsilon)
        QL.initQTable()
        QL.setQTable()
        discovered_maze = QL.getPath()

        if discovered_maze is not None:
            print('Path Found!')
            File.writeFile(discovered_maze)
        else:
            print("No path found in the given maze")

    else:
        print("Some error occured")
except:
    print('No path found')