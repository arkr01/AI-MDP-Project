from time import time
from itertools import product
from numpy import zeros, full, array, linalg, identity

from game_state import GameState

"""
solution.py

Template file for you to implement your solution to Assignment 2.

You must implement the following method stubs, which will be invoked by the simulator during 
testing: __init__(game_env) plan_offline() select_action() 
    
To ensure compatibility with the autograder, please avoid using try-except blocks for Exception 
or OSError exception types. Try-except blocks with concrete exception types other than OSError
(e.g. try: ... except ValueError) are allowed. 

COMP3702 2021 Assignment 2 Support Code

Last updated by njc 02/09/21

REFERENCES:
- Tutorial 7 Solution Code (For Value/Policy/Linear Algebra Policy Iteration Implementations)
"""


def dict_argmax(d):
    """ FROM TUTORIAL 7 SOLUTION. GET ARGMAX FROM DICTIONARY."""
    max_value = max(d.values())
    for k, v in d.items():
        if v == max_value:
            return k


class Solver:

    def __init__(self, game_env):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor,
        and should be placed in the plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator
        if not completed within the limit.
        """
        self.game_env = game_env
        self.states = []
        self.state_indices = {}
        self.values = {}
        self.policy = {}
        self.IMPOSSIBLE_PENALTY = -1000
        self.discount = 0.9999
        self.USE_LIN_ALG = True  # From Tutorial 7 Solution Code
        self.epsilon = 0.001
        self.converged = False

    def plan_offline(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform value iteration and/or policy iteration and store the
        computed policy, or (optionally) to perform pre-processing for MCTS.

        This planning should not depend on the initial state, as during simulation this is not
        guaranteed to match the initial position listed in the input file (i.e. you may be given
        a different position to the initial position when select_action is called).

        The allowed run time for this method is given by 'game_env.offline_time'. The method will
        be terminated by the simulator if it does not complete within this limit - you should
        design your algorithm to ensure this method exits before the time limit is exceeded.
        """
        t0 = time()
        # Gets the cartesian product of all binary tuples, of length n, where n is the number of
        # gems (hence, gets all possible gem statuses)
        self.possible_gem_statuses = list(product((0, 1), repeat=self.game_env.n_gems))

        # Initialise list of states
        for r in range(1, self.game_env.n_rows - 1):
            for c in range(1, self.game_env.n_cols - 1):
                for gem_status in self.possible_gem_statuses:
                    if self.game_env.grid_data[r][c] not in self.game_env.COLLISION_TILES:
                        self.states.append(GameState(r, c, gem_status))
        self.values = {state: 0 for state in self.states}  # From Tutorial 7 Solution Code
        self.policy = {state: self.game_env.DROP_1 for state in self.states}

        for i, s in enumerate(self.states):
            self.state_indices[s] = i

        # Below initialisations are HEAVILY BASED OFF OF THE TUTORIAL 7 SOLUTION CODE
        # t model (lin alg)
        t_model = zeros([len(self.states), len(self.game_env.ACTIONS), len(self.states)])

        # r model (lin alg)
        r_model = zeros([len(self.states), len(self.game_env.ACTIONS)])
        for i, s in enumerate(self.states):
            for j, a in enumerate(self.game_env.ACTIONS):
                expected_reward = 0
                total_prob = 0
                current_transition = self.get_transition_info(s, a)
                for (next_state, reward, probability) in current_transition:
                    k = self.state_indices[next_state]
                    t_model[i][j][k] += probability
                    expected_reward += reward * probability
                r_model[i][j] = expected_reward
        self.t_model = t_model
        self.r_model = r_model

        # lin alg policy
        self.la_policy = full([len(self.states)], self.game_env.ACTION_INDEX[self.game_env.DROP_1])

        # optional: loop for ensuring your code exits before the time limit
        if self.game_env.offline_time > self.game_env.online_time:
            while not self.converged:
                self.policy_iteration()

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to
        perform (once for each step of the episode).

        You can use this to retrieve the optimal action for the current state from a stored
        offline policy (e.g. from value iteration or policy iteration), or to perform MCTS
        simulations from the current state.

        The allowed run time for this method is given by 'game_env.online_time'. The method will
        be terminated by the simulator if it does not complete within this limit - you should
        design your algorithm to ensure this method exits before the time limit is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """
        t0 = time()
        while time() - t0 < self.game_env.online_time:
            self.policy_iteration()
        return self.policy[state]

    def get_transition_info(self, state, action):
        """
        Transition Function for the agent. Returns a list of states, along with their
        corresponding rewards and probabilities.

        HEAVILY BASED OFF OF game_env.perform_action()
        """
        possible_outcomes = []
        is_solved = self.game_env.is_solved(state)

        # Check if game is solved/over
        if is_solved:
            reward = 0
            possible_outcomes.append((state, reward, 1))
            return possible_outcomes

        reward = -1 * self.game_env.ACTION_COST[action]

        # Handle invalid actions
        if action in {self.game_env.WALK_LEFT, self.game_env.WALK_RIGHT, self.game_env.JUMP}:
            # check walkable ground prerequisite if action is walk or jump
            if self.game_env.grid_data[state.row + 1][state.col] not in \
                    self.game_env.WALK_JUMP_ALLOWED_TILES:
                possible_outcomes.append((state, self.IMPOSSIBLE_PENALTY, 1))
                return possible_outcomes
        else:
            # check permeable ground prerequisite if action is glide or drop
            if self.game_env.grid_data[state.row + 1][state.col] not in \
                    self.game_env.GLIDE_DROP_ALLOWED_TILES:
                possible_outcomes.append((state, self.IMPOSSIBLE_PENALTY, 1))
                return possible_outcomes

        # handle each (valid) action type separately
        if action in self.game_env.WALK_ACTIONS:
            # Handle super-charged walk
            if self.game_env.grid_data[state.row + 1][state.col] == self.game_env.SUPER_CHARGE_TILE:
                move_dir = -1 if action == self.game_env.WALK_LEFT else 1
                next_row, next_col = state.row, state.col
                next_gem_status = state.gem_status

                # Move up to the last adjoining supercharge tile
                while self.game_env.grid_data[next_row + 1][next_col + move_dir] == \
                        self.game_env.SUPER_CHARGE_TILE:
                    next_col += move_dir
                    # check for collision or game over and update reward/bounce back position
                    # accordingly
                    next_row, next_col, reward, collision, is_game_over = \
                        self.game_env.check_collision_or_terminal(next_row, next_col, reward,
                                                                  row_move_dir=0,
                                                                  col_move_dir=move_dir)
                    if collision or is_game_over:
                        break

                current_row = next_row
                current_col = next_col
                current_gem_status = next_gem_status
                latest_reward = reward

                # Handle each potential distance moved beyond the last adjoining supercharge tile
                for dist in self.game_env.super_charge_probs.keys():
                    reward = latest_reward
                    next_row = current_row
                    next_col = current_col
                    next_gem_status = current_gem_status
                    # move current distance beyond the last adjoining supercharge tile
                    for d in range(dist):
                        next_col += move_dir
                        # check for collision or game over
                        next_row, next_col, reward, collision, is_game_over = \
                            self.game_env.check_collision_or_terminal(next_row, next_col, reward,
                                                                      row_move_dir=0,
                                                                      col_move_dir=move_dir)
                        if collision or is_game_over:
                            break

                    # check if a gem is collected or goal is reached (only do this for final
                    # position of charge)
                    next_gem_status, is_solved = \
                        self.game_env.check_gem_collected_or_goal_reached(next_row, next_col,
                                                                          next_gem_status)
                    possible_outcomes.append((GameState(next_row, next_col, next_gem_status),
                                              reward, self.game_env.super_charge_probs[dist]))
            else:
                # Handle conventional walk
                next_gem_status = state.gem_status
                prob = 1
                reward = -1 * self.game_env.ACTION_COST[action]
                # if player is above ladder
                if self.game_env.grid_data[state.row + 1][state.col] == self.game_env.LADDER_TILE \
                        and self.game_env.grid_data[state.row + 2][state.col] not in \
                        self.game_env.COLLISION_TILES:
                    # Fell down from ladder
                    next_row, next_col = state.row + 2, state.col

                    # check if a gem is collected or goal is reached
                    next_gem_status, is_solved = self.game_env.check_gem_collected_or_goal_reached(
                        next_row, next_col, next_gem_status)
                    possible_outcomes.append((GameState(next_row, next_col, next_gem_status),
                                              reward, self.game_env.ladder_fall_prob))
                    prob -= self.game_env.ladder_fall_prob
                # if not on ladder or there is a collision tile 2 tiles below the ladder
                col_move_dir = -1 if action == self.game_env.WALK_LEFT else 1
                row_move_dir = 0
                next_row, next_col = state.row, state.col + col_move_dir
                reward = -1 * self.game_env.ACTION_COST[action]
                next_gem_status = state.gem_status
                # check for collision or game over
                next_row, next_col, reward, collision, is_game_over = \
                    self.game_env.check_collision_or_terminal(next_row, next_col, reward,
                                                              row_move_dir=row_move_dir,
                                                              col_move_dir=col_move_dir)
                # check if a gem is collected or goal is reached
                next_gem_status, is_solved = self.game_env.check_gem_collected_or_goal_reached(
                    next_row, next_col, next_gem_status)
                possible_outcomes.append((GameState(next_row, next_col, next_gem_status),
                                          reward, prob))
        elif action == self.game_env.JUMP:
            # Handle super jump
            if self.game_env.grid_data[state.row + 1][state.col] == self.game_env.SUPER_JUMP_TILE:
                for dist in self.game_env.super_jump_probs.keys():
                    reward = -1 * self.game_env.ACTION_COST[action]
                    next_row, next_col = state.row, state.col
                    next_gem_status = state.gem_status
                    # move current distance upwards
                    for d in range(dist):
                        next_row -= 1
                        # check for collision or game over
                        next_row, next_col, reward, collision, is_game_over = \
                            self.game_env.check_collision_or_terminal(next_row, next_col,
                                                                      reward, row_move_dir=-1,
                                                                      col_move_dir=0)
                        if collision or is_game_over:
                            break

                    # check if a gem is collected or goal is reached (only do this for final
                    # position of charge)
                    next_gem_status, is_solved = \
                        self.game_env.check_gem_collected_or_goal_reached(next_row, next_col,
                                                                          next_gem_status)
                    possible_outcomes.append((GameState(next_row, next_col, next_gem_status),
                                              reward, self.game_env.super_jump_probs[dist]))
            # Handle conventional jump
            else:
                reward = -1 * self.game_env.ACTION_COST[action]
                next_row, next_col = state.row - 1, state.col
                next_gem_status = state.gem_status
                # check for collision or game over
                next_row, next_col, reward, collision, is_game_over = \
                    self.game_env.check_collision_or_terminal(next_row, next_col, reward,
                                                              row_move_dir=-1, col_move_dir=0)
                # check if a gem is collected or goal is reached
                next_gem_status, is_solved = self.game_env.check_gem_collected_or_goal_reached(
                    next_row, next_col, next_gem_status)
                possible_outcomes.append((GameState(next_row, next_col, next_gem_status), reward,
                                          1))
        elif action in self.game_env.GLIDE_ACTIONS:
            # select probabilities to sample move distance
            if action in {self.game_env.GLIDE_LEFT_1, self.game_env.GLIDE_RIGHT_1}:
                probs = self.game_env.glide1_probs
            elif action in {self.game_env.GLIDE_LEFT_2, self.game_env.GLIDE_RIGHT_2}:
                probs = self.game_env.glide2_probs
            else:
                probs = self.game_env.glide3_probs

            for dist in probs.keys():
                reward = -1 * self.game_env.ACTION_COST[action]
                # set movement direction
                move_dir = -1 if action in {self.game_env.GLIDE_LEFT_1,
                                            self.game_env.GLIDE_LEFT_2,
                                            self.game_env.GLIDE_LEFT_3} else 1
                # move current distance in chosen direction
                next_row, next_col = state.row + 1, state.col
                next_gem_status = state.gem_status
                for d in range(dist):
                    next_col += move_dir
                    # check for collision or game over
                    next_row, next_col, reward, collision, is_game_over = \
                        self.game_env.check_collision_or_terminal_glide(next_row, next_col,
                                                                        reward,
                                                                        row_move_dir=0,
                                                                        col_move_dir=move_dir)
                    if collision or is_game_over:
                        break

                # check if a gem is collected or goal is reached (only do this for final position
                # of charge)
                next_gem_status, is_solved = self.game_env.check_gem_collected_or_goal_reached(
                    next_row, next_col, next_gem_status)
                possible_outcomes.append((GameState(next_row, next_col, next_gem_status), reward,
                                          probs[dist]))
        elif action in self.game_env.DROP_ACTIONS:
            move_dist = {self.game_env.DROP_1: 1, self.game_env.DROP_2: 2,
                         self.game_env.DROP_3: 3}[action]

            # drop by chosen distance
            next_row, next_col = state.row, state.col
            next_gem_status = state.gem_status
            reward = -1 * self.game_env.ACTION_COST[action]
            for d in range(move_dist):
                next_row += 1

                # check for collision or game over
                next_row, next_col, reward, collision, is_game_over = \
                    self.game_env.check_collision_or_terminal_glide(next_row, next_col, reward,
                                                                    row_move_dir=1,
                                                                    col_move_dir=0)
                if collision or is_game_over:
                    break
            # check if a gem is collected or goal is reached (only do this for final position of
            # charge)
            next_gem_status, is_solved = self.game_env.check_gem_collected_or_goal_reached(
                next_row, next_col, next_gem_status)

            # GameState(next_row, next_col, next_gem_status)
            possible_outcomes.append((GameState(next_row, next_col, next_gem_status), reward, 1))
        return possible_outcomes

    def value_iteration(self):
        """
        Perform one iteration of value iteration. HEAVILY BASED OFF OF THE TUTORIAL 7 SOLUTION CODE
        """
        new_values = dict()
        new_policy = dict()
        for s in self.states:
            # Keep track of maximum value
            action_values = dict()
            for a in self.game_env.ACTIONS:
                total = 0
                current_transition = self.get_transition_info(s, a)
                for (next_state, reward, probability) in current_transition:
                    # Apply action
                    total += probability * (reward + (self.discount * self.values[next_state]))
                    action_values[a] = total
                    # Update state value with best action
                    new_values[s] = max(action_values.values())
                    new_policy[s] = dict_argmax(action_values)

        # Update values
        self.values = new_values
        self.policy = new_policy

    def policy_iteration(self):
        """
        Perform one iteration of policy iteration. HEAVILY BASED OFF OF THE TUTORIAL 7 SOLUTION CODE
        """
        new_policy = dict()

        # policy evaluation
        if not self.USE_LIN_ALG:
            # use 'naive'/iterative policy evaluation
            value_converged = False
            while not value_converged:
                new_values = dict()
                for s in self.states:
                    total = 0
                    current_transition = self.get_transition_info(s, self.policy[s])
                    for (next_state, reward, probability) in current_transition:
                        # Apply action
                        total += probability * (reward + (self.discount * self.values[next_state]))
                    # Update state value with best action
                    new_values[s] = total

                # Check convergence
                # differences = [abs(self.values[s] - new_values[s]) for s in self.states]
                # if max(differences) < self.epsilon:
                #     value_converged = True

                # Update values and policy
                self.values = new_values
        else:
            # use linear algebra for policy evaluation
            # V^pi = R + gamma T^pi V^pi
            # (I - gamma * T^pi) V^pi = R
            # Ax = b; A = (I - gamma * T^pi),  b = R
            state_numbers = array(range(len(self.states)))  # indices of every state
            t_pi = self.t_model[state_numbers, self.la_policy]
            r_pi = self.r_model[state_numbers, self.la_policy]
            values = linalg.solve(identity(len(self.states)) - (self.discount * t_pi), r_pi)
            self.values = {s: values[i] for i, s in enumerate(self.states)}

        # policy improvement
        for s in self.states:
            # Keep track of maximum value
            action_values = dict()
            for a in self.game_env.ACTIONS:
                total = 0
                current_transition = self.get_transition_info(s, a)
                for (next_state, reward, probability) in current_transition:
                    # Apply action
                    total += probability * (reward + (self.discount * self.values[next_state]))
                action_values[a] = total
            # Update policy
            new_policy[s] = dict_argmax(action_values)

        if new_policy == self.policy:
            self.converged = True

        self.policy = new_policy
        if self.USE_LIN_ALG:
            for i, s in enumerate(self.states):
                self.la_policy[i] = self.game_env.ACTION_INDEX[self.policy[s]]
