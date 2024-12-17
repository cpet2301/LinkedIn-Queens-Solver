import numpy as np
from collections import deque
import copy

def is_valid_move(grid, row, col, queens_placed):
    """
    Check if placing a queen at (row, col) follows the constraints of:
    - No queens in the same row
    - No queens in the same column
    - No queens in the same color region
    - No queens neighboring each other
    """
    for qr, qc in queens_placed:
        if qr == row or qc == col: # Row and column constraints
            return False
        if grid[row][col][1] == grid[qr][qc][1]: # Color region constraint
            return False
        if abs(qr - row) <= 1 and abs(qc - col) <= 1: # Neighbor constraint
            return False
    return True

def breadth_first_search(initial_grid):
    N = len(initial_grid)
    queue = deque()
    visited = set()

    start_state = (initial_grid, [])
    queue.append(start_state)
    visited.add(tuple())

    while queue:
        grid, queens_placed = queue.popleft()

        print(f"Checking state: {queens_placed}")

        if len(queens_placed) == N:
            print("Solution Found!")
            return grid

        for r in range(N):
            for c in range(N):
                if (r, c) not in queens_placed and is_valid_move(grid, r, c, queens_placed):
                    new_grid = [row[:] for row in grid]
                    new_grid[r][c] = (1, new_grid[r][c][1])

                    new_queens = queens_placed + [(r, c)]
                    canonical_queens = tuple(sorted(new_queens))

                    if canonical_queens not in visited:
                        queue.append((new_grid, new_queens))
                        visited.add(canonical_queens)

    print("No solution found.")
    return None

def depth_first_search(initial_board):
    pass

def local_search(initial_board):
    pass

def backtracking_search(initial_board):
    pass

def genetic_search(initial_board):
    pass