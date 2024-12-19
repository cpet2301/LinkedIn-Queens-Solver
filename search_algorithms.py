from collections import deque
import random

def is_valid_move(grid, row, col, queens_placed):
    """
    Check if placing a queen at (row, col) follows the constraints of:
    - No queens in the same row
    - No queens in the same column
    - No queens in the same color region
    - No queens neighboring each other

    Parameters:
        grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.
        row (int): The row in which the new queen will be placed.
        col (int): The column in which the new queen will be placed.
        queens_placed (list of tuples): A list of locations of current queens where each element contains:
            - row (int): The row in which the current queen is placed.
            - col (int): The column in which the current queen is placed.

    Returns:
        bool: Whether the placement of the new queen is valid
    """
    for qr, qc in queens_placed:
        if qr == row or qc == col: # Row and column constraints
            return False
        if grid[row][col][1] == grid[qr][qc][1]: # Color region constraint
            return False
        if abs(qr - row) <= 1 and abs(qc - col) <= 1: # Neighbor constraint
            return False
    return True

def calculate_conflicts(grid, queens_placed):
    """
    Calculate the number of conflicts for a given board configuration.

    A conflict occurs when two queens threaten each other (same row, column, region, or adjacent).

    Parameters:
        grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.
        queens_placed (list of tuples): A list of locations of current queens where each element contains:
            - row (int): The row in which the current queen is placed.
            - col (int): The column in which the current queen is placed.
    
    Returns:
        conflicts (int): The number of conflicts in the board configuration.
    """
    conflicts = 0
    N = len(grid)
    
    for i in range(len(queens_placed)):
        r1, c1 = queens_placed[i]
        for j in range(i + 1, len(queens_placed)):
            r2, c2 = queens_placed[j]
            
            if r1 == r2 or c1 == c2:  # Same row or column
                conflicts += 1
            if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:  # Adjacent queens
                conflicts += 1
            if grid[r1][c1][1] == grid[r2][c2][1]:  # Same region
                conflicts += 1
                
    return conflicts

def breadth_first_search(initial_grid):
    """
    This algorithm explores all possible configurations level by level. It checks all positions for each queen before moving to the next level.
    
    BFS ensures the shortest path to a solution is found, but it may require significant memory since it stores every state.

    Parameters:
        initial_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        tuple: A tuple that contains:
            - result (int): The success (1) or failure (0) of the search algorithm in finding a solution.
            - grid (2D list of tuples): A 2D grid where each cell contains:
                - is_queen (int): Indicates whether a queen is placed (1) or not (0).
                - region_id (int): The color identifier representing the region.
    """
    N = len(initial_grid)  # Size of the grid (NxN)
    
    # Step 1: Initialize a queue and visited set to store states
    queue = deque()
    visited = set()

    start_state = (initial_grid, [])  # Initial state with no queens placed
    queue.append(start_state)
    visited.add(tuple())

    # Step 2: Start exploring the grid
    while queue:
        grid, queens_placed = queue.popleft()  # Pop the first state from the queue

        # Step 2.1: If all queens are placed, return the solution
        if len(queens_placed) == N:
            print("Solution Found!")
            return 1, grid

        # Step 2.2: Explore all positions and add valid moves to the queue
        for r in range(N):
            for c in range(N):
                if (r, c) not in queens_placed and is_valid_move(grid, r, c, queens_placed):
                    # Step 2.2.1: Create a new state with the queen placed at (r, c)
                    new_grid = [row[:] for row in grid]  # Copy the grid
                    new_grid[r][c] = (1, new_grid[r][c][1])

                    new_queens = queens_placed + [(r, c)]
                    canonical_queens = tuple(sorted(new_queens))

                    # Step 2.2.2: If the new state has not been visited, add it to the queue
                    if canonical_queens not in visited:
                        queue.append((new_grid, new_queens))
                        visited.add(canonical_queens)

    # Step 3: If no solution is found after exploring all possibilities
    print("No solution found.")
    return 0, grid

def depth_first_search(initial_grid):
    """
    This algorithm explores each possible configuration deeply before backtracking to explore other configurations.
    
    It uses less memory than BFS but may not find the shortest path to a solution. It may explore unnecessary paths.

    Parameters:
        initial_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        tuple: A tuple that contains:
            - result (int): The success (1) or failure (0) of the search algorithm in finding a solution.
            - grid (2D list of tuples): A 2D grid where each cell contains:
                - is_queen (int): Indicates whether a queen is placed (1) or not (0).
                - region_id (int): The color identifier representing the region.
    """
    N = len(initial_grid)
    
    # Step 1: Initialize a stack and visited set to store states
    stack = []  # Stack for DFS, storing states (grid, queens_placed)
    visited = set()  # Set to track visited configurations

    start_state = (initial_grid, [])  # Initial state with no queens placed
    stack.append(start_state)
    visited.add(tuple())

    # Step 2: Start exploring the grid
    while stack:
        grid, queens_placed = stack.pop()  # Pop the last state from the stack

        # Step 2.1: If all queens are placed, return the solution
        if len(queens_placed) == N:
            print("Solution Found!")
            return 1, grid

        # Step 2.2: Explore all positions and add valid moves to the stack
        for r in range(N):
            for c in range(N):
                if (r, c) not in queens_placed and is_valid_move(grid, r, c, queens_placed):
                    # Step 2.2.1: Create a new state with the queen placed at (r, c)
                    new_grid = [row[:] for row in grid]
                    new_grid[r][c] = (1, new_grid[r][c][1])

                    new_queens = queens_placed + [(r, c)]
                    canonical_queens = tuple(sorted(new_queens))

                    # Step 2.2.2: If the new state has not been visited, add it to the stack
                    if canonical_queens not in visited:
                        stack.append((new_grid, new_queens))
                        visited.add(canonical_queens)

    # Step 3: If no solution is found after exploring all possibilities
    print("No solution found.")
    return 0, grid

def local_search(initial_grid):
    """
    This algorithm places queens randomly and tries to minimize conflicts (attacks between queens) by moving queens to less conflicted positions.
    
    It can get stuck in local optima (suboptimal solutions) and requires restarts if no solution is found.
    
    Parameters:
        initial_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        tuple: A tuple that contains:
            - result (int): The success (1) or failure (0) of the search algorithm in finding a solution.
            - grid (2D list of tuples): A 2D grid where each cell contains:
                - is_queen (int): Indicates whether a queen is placed (1) or not (0).
                - region_id (int): The color identifier representing the region.
    """
    N = len(initial_grid)  # Size of the grid (NxN)
    
    # Step 1: Generate a random initial state with no queens placed
    queens_placed = []  # List of positions (row, col) where queens are placed
    grid = [row[:] for row in initial_grid]  # Copy of the initial grid

    # Step 2: Randomly place queens initially
    for r in range(N):
        for c in range(N):
            if is_valid_move(grid, r, c, queens_placed):
                queens_placed.append((r, c))
                grid[r][c] = (1, grid[r][c][1])  # Place a queen at (r, c)
                break  # Stop after placing a queen in this row

    # Step 3: Iterate and improve the current state (local search)
    while True:
        # Step 3.1: Evaluate the current state by checking the conflicts
        conflicts = calculate_conflicts(grid, queens_placed)
        
        # If we've placed N queens and there are no conflicts, we found a solution
        if len(queens_placed) == N and conflicts == 0:
            print("Solution Found!")
            return 1, grid
        
        # Step 3.2: Generate neighboring states and pick the best one
        best_move = None
        best_conflicts = conflicts
        
        for i, (r, c) in enumerate(queens_placed):
            # Try moving the queen at position (r, c) to a new position
            for nr in range(N):
                for nc in range(N):
                    if (nr, nc) != (r, c) and is_valid_move(grid, nr, nc, queens_placed):
                        new_grid = [row[:] for row in grid]  # Copy the grid
                        new_queens_placed = queens_placed[:]
                        
                        # Move the queen and re-evaluate the number of conflicts
                        new_grid[r][c] = (0, new_grid[r][c][1])  # Remove the queen from (r, c)
                        new_grid[nr][nc] = (1, new_grid[nr][nc][1])  # Place the queen at (nr, nc)
                        new_queens_placed[i] = (nr, nc)
                        
                        new_conflicts = calculate_conflicts(new_grid, new_queens_placed)
                        
                        # If this new state has fewer conflicts, consider it
                        if new_conflicts < best_conflicts:
                            best_conflicts = new_conflicts
                            best_move = (new_grid, new_queens_placed)
        
        # If no improvement is possible (local minimum), return None
        if not best_move:
            print("No solution found.")
            return 0, grid
        
        # Apply the best move found
        grid, queens_placed = best_move

def backtracking_search(initial_grid):
    """
    This algorithm places queens one at a time and checks each possibility recursively. If a conflict is found, it backtracks to the previous step.
    
    It guarantees a solution if one exists but may be slower for large grids as it explores every possibility.
    
    Parameters:
        initial_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        tuple: A tuple that contains:
            - result (int): The success (1) or failure (0) of the search algorithm in finding a solution.
            - 2D list of tuples: A 2D grid where each cell contains:
                - is_queen (int): Indicates whether a queen is placed (1) or not (0).
                - region_id (int): The color identifier representing the region.
    """
    N = len(initial_grid)
    queens_placed = []
    grid = [row[:] for row in initial_grid]

    def backtrack(row):
        # Step 1: If we've placed queens in all rows, we've found a solution
        if row == N:
            print("Solution Found!")
            return 1, grid

        # Step 2: Try placing a queen in each column of the current row
        for col in range(N):
            # Step 2.1: If placing a queen at (row, col) is valid, place the queen
            if is_valid_move(grid, row, col, queens_placed):
                queens_placed.append((row, col))  # Add the queen to the list
                grid[row][col] = (1, grid[row][col][1])  # Place the queen on the grid
                
                # Step 2.2: Recursively attempt to place queens in the next row
                result = backtrack(row + 1)
                if result is not None:
                    return 1, result  # If a solution is found, return the grid

                # Step 2.3: If placing the queen leads to no solution, backtrack
                queens_placed.pop()  # Remove the queen from the list
                grid[row][col] = (0, grid[row][col][1])  # Remove the queen from the grid

        # Step 3: If no valid move is found in this row, return None (no solution in this branch)
        return 0, None

    # Step 4: Start the backtracking from the first row
    return backtrack(0)

def genetic_search(initial_grid):
    """
    This algorithm simulates natural selection by creating populations of solutions, selecting the best solutions, and combining them to form new solutions.

    This algorithm is not guaranteed to find a solution as it can stop prematurely or get stuck on local optima.
    
    Parameters:
        initial_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        tuple: A tuple that contains:
            - result (int): The success (1) or failure (0) of the search algorithm in finding a solution.
            - 2D list of tuples: A 2D grid where each cell contains:
                - is_queen (int): Indicates whether a queen is placed (1) or not (0).
                - region_id (int): The color identifier representing the region.
    """
    N = len(initial_grid)  # Size of the grid (NxN)
    population_size = 100
    generations = 1000
    mutation_rate = 0.1

    # Step 1: Initialize a random population
    population = []
    for _ in range(population_size):
        queens_placed = []
        grid = [row[:] for row in initial_grid]  # Copy of the initial grid
        for r in range(N):
            for c in range(N):
                if is_valid_move(grid, r, c, queens_placed):
                    queens_placed.append((r, c))
                    grid[r][c] = (1, grid[r][c][1])  # Place a queen
                    break  # Stop after placing one queen per row
        population.append((grid, queens_placed))

    def fitness(grid, queens_placed):
        """Fitness function: count the number of conflicts in the configuration."""
        return calculate_conflicts(grid, queens_placed)

    def selection(population):
        """Select two parents using tournament selection."""
        tournament_size = 5
        selected_parents = []
        for _ in range(2):
            tournament = random.sample(population, tournament_size)
            best_individual = min(tournament, key=lambda x: fitness(x[0], x[1]))
            selected_parents.append(best_individual)
        return selected_parents

    def crossover(parent1, parent2):
        """Crossover between two parents to create two offspring."""
        grid1, queens1 = parent1
        grid2, queens2 = parent2
        crossover_point = random.randint(1, N-1)
        
        # Combine the queens' positions up to the crossover point
        offspring1_queens = queens1[:crossover_point] + queens2[crossover_point:]
        offspring2_queens = queens2[:crossover_point] + queens1[crossover_point:]
        
        # Create new grids based on the offspring's queens' positions
        offspring1_grid = [row[:] for row in initial_grid]
        offspring2_grid = [row[:] for row in initial_grid]
        
        for r, c in offspring1_queens:
            offspring1_grid[r][c] = (1, offspring1_grid[r][c][1])
        for r, c in offspring2_queens:
            offspring2_grid[r][c] = (1, offspring2_grid[r][c][1])

        return (offspring1_grid, offspring1_queens), (offspring2_grid, offspring2_queens)

    def mutation(grid, queens_placed):
        """Apply mutation to a configuration (randomly move one queen)."""
        if random.random() < mutation_rate:
            r, c = random.choice(queens_placed)
            # Try moving this queen to a random valid position
            for nr in range(N):
                for nc in range(N):
                    if (nr, nc) != (r, c) and is_valid_move(grid, nr, nc, queens_placed):
                        new_grid = [row[:] for row in grid]
                        new_grid[r][c] = (0, new_grid[r][c][1])  # Remove the queen
                        new_grid[nr][nc] = (1, new_grid[nr][nc][1])  # Place the queen at new position
                        queens_placed.remove((r, c))
                        queens_placed.append((nr, nc))
                        return new_grid, queens_placed
        return grid, queens_placed  # No mutation occurred

    # Step 2: Evolve population for a number of generations
    for generation in range(generations):
        # Evaluate fitness of the population
        population.sort(key=lambda x: fitness(x[0], x[1]))
        best_individual = population[0]
        
        # Check if the best individual is the solution (no conflicts and exactly N queens placed)
        if len(best_individual[1]) == N and fitness(best_individual[0], best_individual[1]) == 0:
            print("Solution Found!")
            return 1, best_individual[0]

        # Selection
        parent1, parent2 = selection(population)

        # Crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Mutation
        offspring1 = mutation(offspring1[0], offspring1[1])
        offspring2 = mutation(offspring2[0], offspring2[1])

        # Replace the two worst individuals in the population with the offspring
        population[-2] = offspring1
        population[-1] = offspring2

    print("No solution found.")
    return 0, population[0][0]  # Return the best individual after all generations
