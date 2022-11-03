import numpy as np
import random
import time
import click

class Board:
    def __init__(self, n, heuristic_function=1) -> None:
        """
        heuristics_function: 1 or 2
        """
        self.n = n
        self.state = None
        self.initialize_board()
        self.seen_states = []
        self.heuristic_function = heuristic_function
        if heuristic_function == 1:
            self.heuristic = self.heuristic1
            self.goal = 0
        elif heuristic_function == 2:
            self.heuristic = self.heuristic2
            self.goal = self.n

        self.h_score = self.heuristic()

    def initialize_board(self):
        """
        Initialize the board with randm positions of queens.
        One queen per row.
        """
        self.state = np.zeros((self.n, self.n), dtype=np.int8)
        columns = list(range(self.n))
        random.shuffle(columns, random.random)  # Shuffle the columns in-place

        # Place the queens randomly
        for row, col in enumerate(columns):
            self.state[row, col] = 1

    def heuristic1(self, board=None):
        """
        Compute the heuristic value of a given state.
        The heuristic value is the number of pairs of queens that can attack each other.
        """
        if board is None:  # If no board is passed, use the current state
            board = self

        score = 0
        for row in range(board.n):
            for col in range(board.n):
                if board.state[row, col] == 1:  # If there is a queen in this position
                    # Check the column for other queens
                    score += np.sum(board.state[:, col]) - 1
                    # Check the diagonals for other queens
                    for i in range(1, 8):
                        if row + i < 8 and col + i < 8:
                            score += board.state[row + i, col + i]
                        if row - i >= 0 and col - i >= 0:
                            score += board.state[row - i, col - i]
                        if row + i < 8 and col - i >= 0:
                            score += board.state[row + i, col - i]
                        if row - i >= 0 and col + i < 8:
                            score += board.state[row - i, col + i]

        self.h_score = score
        return score

    def heuristic2(self, board=None):
        """
        Compute the heuristic value of a given state.
        The heuristic value is the number of queens that are not under attack.
        """
        if board is None:  # If no board is passed, use the current state
            board = self

        score = 0
        for row in range(board.n):
            for col in range(board.n):
                if board.state[row, col] == 1:  # If there is a queen in this position
                    temp_score = 0
                    # Check the column for other queens
                    temp_score += np.sum(board.state[:, col]) - 1
                    # Check the diagonals for other queens
                    for i in range(1, 8):
                        if row + i < 8 and col + i < 8:
                            temp_score += board.state[row + i, col + i]
                        if row - i >= 0 and col - i >= 0:
                            temp_score += board.state[row - i, col - i]
                        if row + i < 8 and col - i >= 0:
                            temp_score += board.state[row + i, col - i]
                        if row - i >= 0 and col + i < 8:
                            temp_score += board.state[row - i, col + i]

                    if temp_score == 0: # If the queen is not under attack at all
                        score += 1 # Increment the score
        self.h_score = score
        return score

    def get_children(self):
        """
        Get all the children of the current state.
        """
        children = []
        if self.heuristic_function == 1:
            best_child_score = np.Inf
        elif self.heuristic_function == 2:
            best_child_score = -1
            
        for row in range(self.n):
            for col in range(self.n):
                if self.state[row, col] == 1:  # If there is a queen in this position
                    for i in range(
                        self.n
                    ):  # Check all the possible positions for the queen
                        if i != col:
                            child = Board(self.n, self.heuristic_function)
                            child.state = self.state.copy()
                            child.state[
                                row, col
                            ] = 0  # Remove the queen from the current position
                            child.state[
                                row, i
                            ] = 1  # Move the queen to the new position
                            score = (
                                child.heuristic()
                            )  # Compute the heuristic score of the child

                            if (
                                self.heuristic_function == 1
                            ):  # If heuristic function 1 is used
                                if (
                                    score <= best_child_score
                                ):  # If the child is better than the best child so far:
                                    best_child_score = score
                                    children.insert(
                                        0, child
                                    )  # Insert the child at the beginning of the list
                                else:  # If the child is not better than the best child so far:
                                    children.append(
                                        child
                                    )  # Append the child to the end of the list

                            elif (
                                self.heuristic_function == 2
                            ):  # If heuristic function 2 is used
                                if (
                                    score >= best_child_score
                                ):  # If the child is better than the best child so far:
                                    best_child_score = score
                                    children.insert(
                                        0, child
                                    )  # Insert the child at the beginning of the list
                                else:  # If the child is not better than the best child so far:
                                    children.append(
                                        child
                                    )  # Append the child to the end of the list

        return children

    def search(self, time_out=-1, verbose=False):
        """
        Search for the solution using the hill climbing algorithm.
        """
        current_state = self
        previous_state = self

        children = current_state.get_children()  # Get the children of the initial state

        start = time.time()
        while (
            current_state.h_score != self.goal
        ):  # While the current state is not the goal state
            if verbose:
                print(current_state.state)
                print(f'Heuristic Score = {current_state.h_score}')
                print('\n------------------------------------------------------------\n')

            if current_state != previous_state:
                # Get the children only if the current state is
                # different from the previous state to avoid getting
                # the same children again. (For efficiency)
                children = current_state.get_children()
                previous_state = current_state

            current_state = children[0]  # Get the best child

            if time_out > 0:  # If a time out is specified
                if (
                    time.time() - start > time_out
                ):  # If the algorithm takes more than #time_out seconds
                    print("Time out, solution not found.")
                    return False  # Return False: solution not found

        # If the algorithm finds the solution
        print(f"Solution for {self.n}x{self.n} Queens Problem found!")
        print(current_state.state)  # Print the solution
        print(f"Heuristic Score = {current_state.h_score}")
        print("\n------------------------------------------------------------\n")
        return True

    def search_randomness(self, time_out=-1, verbose=False):
        """
        Search for the solution using the hill climbing algorithm with randomness added.
        The randomess helps the algorithm to scape from the local optimum when stuck.
        The randomness is added by using a random child from the children list instead of the first.
        The randomness is applied when we do not see any improvement in the heuristic score.
        """
        current_state = self
        previous_state = self

        children = current_state.get_children()  # Get the children of the initial state

        start = time.time()
        while (
            current_state.h_score != self.goal
        ):  # While the current state is not the goal state
            if verbose:
                print(current_state.state)
                print(f'Heuristic Score = {current_state.h_score}')
                print('\n------------------------------------------------------------\n')

            if current_state != previous_state:
                # Get the children only if the current state is
                # different from the previous state to avoid getting
                # the same children again. (For efficiency)
                children = current_state.get_children()
                previous_state = current_state

            if self.heuristic_function == 1:  # If heuristic function 1 is used
                if (
                    current_state.h_score <= children[0].h_score
                ):  # If the the best child is not better than the current state
                    current_state = children[
                        random.randint(0, len(children) - 1)
                    ]  # Use a random child
                else:  # If the best child is better than the current state
                    current_state = children[0]  # Use the best child

            elif self.heuristic_function == 2:  # If heuristic function 2 is used
                if (
                    current_state.h_score >= children[0].h_score
                ):  # If the the best child is not better than the current state
                    current_state = children[
                        random.randint(0, len(children) - 1)
                    ]  # Use a random child
                else:  # If the best child is better than the current state
                    current_state = children[0]  # Use the best child

            if time_out > 0:  # If a time out is specified
                if (
                    time.time() - start > time_out
                ):  # If the algorithm takes more than #time_out seconds
                    print("Time out, solution not found.")
                    return False  # Return False: solution not found

        # If the algorithm finds the solution
        print(f"Solution for {self.n}x{self.n} Queens Problem found!")
        print(current_state.state)  # Print the solution
        print(f"Heuristic Score = {current_state.h_score}")
        print("\n------------------------------------------------------------\n")
        return True



@click.command()
@click.option('--n', '-n', default=8, help='The size of the board.')
@click.option('--heuristic', default=1, help='The heuristic function to use.')
@click.option('--randomness', is_flag=True, help='Use randomness in the algorithm.')
@click.option('--time-out', default=-1, help='Time out in seconds.')
@click.option('--verbose', is_flag=True, help='Print the steps of the algorithm.')
def main(n, heuristic, randomness, time_out, verbose):
    """
    Example:
    - python n-queen.py --n=16 --heuristic=1 --time-out=5 --verbose
    - python n-queen.py --n=8 --heuristic=1 --randomness --verbose
    - python n-queen.py --n=12 --heuristic=2 --randomness --time-out=30 --verbose
    """
    if heuristic == 1:
        print("Heuristic function 1 is used.")
    elif heuristic == 2:
        print("Heuristic function 2 is used.")
    else:
        print("Invalid heuristic function.")
        return

    if randomness:
        print("Randomness is used.")
    else:
        print("Randomness is not used.")

    if time_out > 0:
        print(f"Time out is set to {time_out} seconds.")

    if verbose:
        print("Verbose mode is on.")

    print("\n------------------------------------------------------------\n")

    start = time.time()
    if randomness:
        Board(n, heuristic).search_randomness(time_out, verbose)
    else:
        Board(n, heuristic).search(time_out, verbose)

    print(f"Time taken: {time.time() - start} seconds")

if __name__ == "__main__":
    main()
