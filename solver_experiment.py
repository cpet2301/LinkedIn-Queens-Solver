import os
import time
import psutil
import tracemalloc
import cv2
from PIL import Image
from screen_capture import grid_count, extract_grid_colors, create_state_space, display_grid
from search_algorithms import breadth_first_search, depth_first_search, local_search, backtracking_search, genetic_search

def run_trial(folder_path, search_algorithm):
    """
    Runs a single trial over a folder of images for a specific search algorithm.

    Parameters:
        folder_path (string): The path to the folder of images being tested.
        search_algorithm (fun): The name of the function being tested.
    
    Returns:
        None: Prints out the results in terms of average time, memory usage, and accuracy.
    """
    num_images = 0
    total_time = 0
    total_space = 0
    successful_solutions = 0

    print(f"Testing {search_algorithm.__name__}...")
    print()

    process = psutil.Process(os.getpid())

    tracemalloc.start()

    for filename in os.listdir(folder_path):
        # Filter for PNG files
        if filename.endswith(".png"):
            num_images += 1

            image_path = os.path.join(folder_path, filename)

            cropped_image = cv2.imread(image_path)
            cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            state_grid = create_state_space(extract_grid_colors(cropped_image_pil))

            start_time = time.time()

            result, _ = search_algorithm(state_grid)

            end_time = time.time()

            time_taken = end_time - start_time
            total_time += time_taken

            current_memory = process.memory_info().rss
            total_space += current_memory

            successful_solutions += result
    
    # Average time complexity (time per image)
    average_time = total_time / num_images if num_images > 0 else 0

    # Average space complexity (memory usage per image)
    average_space = total_space / num_images if num_images > 0 else 0

    # Accuracy (percentage of solutions found)
    completeness = successful_solutions / num_images if num_images > 0 else 0
    
    print()
    print(f"{search_algorithm.__name__}:")
    print(f"Average time complexity: {average_time:.6f} seconds per image")
    print(f"Average space complexity: {average_space / (1024**2):.6f} MB per image")
    print(f"Completeness (accuracy): {completeness:.4f}")
    print()

if __name__ == "__main__":
    print("Trial 1: 7x7 Puzzles")
    print("--------------------")

    run_trial("7x7", breadth_first_search)
    run_trial("7x7", depth_first_search)
    run_trial("7x7", local_search)
    run_trial("7x7", backtracking_search)
    run_trial("7x7", genetic_search)

    print("Trial 2: 8x8 Puzzles")
    print("--------------------")

    run_trial("8x8", breadth_first_search)
    run_trial("8x8", depth_first_search)
    run_trial("8x8", local_search)
    run_trial("8x8", backtracking_search)
    run_trial("8x8", genetic_search)

    print("Trial 3: 9x9 Puzzles")
    print("--------------------")

    run_trial("9x9", breadth_first_search)
    run_trial("9x9", depth_first_search)
    run_trial("9x9", local_search)
    run_trial("9x9", backtracking_search)
    run_trial("9x9", genetic_search)

    print("Trial 4: 10x10 Puzzles")
    print("----------------------")

    run_trial("10x10", breadth_first_search)
    run_trial("10x10", depth_first_search)
    run_trial("10x10", local_search)
    run_trial("10x10", backtracking_search)
    run_trial("10x10", genetic_search)

    print("Trial 5 (bonus): 11x11 Puzzles")
    print("------------------------------")

    run_trial("11x11", breadth_first_search)
    run_trial("11x11", depth_first_search)
    run_trial("11x11", local_search)
    run_trial("11x11", backtracking_search)
    run_trial("11x11", genetic_search)