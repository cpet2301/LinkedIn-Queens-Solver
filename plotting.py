import matplotlib.pyplot as plt
import numpy as np

puzzle_sizes = ['7x7', '8x8', '9x9', '10x10', '11x11']

# Breadth-first search data
breadth_first_search_time = [0.845476, 10.534963, 61.283823, np.nan, np.nan]
breadth_first_search_space = [68.409375, 179.224219, 705.515625, np.nan, np.nan]
breadth_first_search_completeness = [1.0000, 1.0000, 1.0000, np.nan, np.nan]

# Depth-first search data
depth_first_search_time = [0.245405, 3.042529, 14.325647, np.nan, np.nan]
depth_first_search_space = [72.978125, 98.814844, 460.087500, np.nan, np.nan]
depth_first_search_completeness = [1.0000, 1.0000, 1.0000, np.nan, np.nan]

# Local search data
local_search_time = [0.001051, 0.000981, 0.001681, 0.001948, 0.001109]
local_search_space = [69.081250, 92.857812, 303.792969, 50.601562, 52.177734]
local_search_completeness = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

# Backtracking search data
backtracking_search_time = [0.001204, 0.000789, 0.001802, 0.001903, 0.001963]
backtracking_search_space = [65.230469, 88.960156, 305.300781, 51.511719, 52.552734]
backtracking_search_completeness = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

# Genetic search data
genetic_search_time = [0.938612, 0.984549, 1.341594, 1.559411, 2.146893]
genetic_search_space = [66.853906, 87.332031, 299.550000, 52.136719, 53.144531]
genetic_search_completeness = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]

def plot_metric(puzzle_sizes, *args, metric_name, ylabel, title):
    """Function to plot each metric for all algorithms."""
    plt.figure(figsize=(10, 6))
    
    labels = ['breadth_first_search', 'depth_first_search', 'local_search', 
              'backtracking_search', 'genetic_search']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, data in enumerate(args):
        # Handle NaN values by not plotting them for the failed cases
        plt.plot(puzzle_sizes[:4], data[:4], label=f'{labels[i]} (non-11x11)', marker='o', linestyle='-', color=colors[i])
        plt.plot(puzzle_sizes[4:], data[4:], label=f'{labels[i]} (11x11)', marker='x', linestyle='--', color=colors[i])
    
    # Adjusting the y-axis limits to ensure small values are visible (especially for local_search)
    plt.ylim(bottom=0)  # Set the bottom of the y-axis to 0
    
    # Set labels and title
    plt.xlabel("Puzzle Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Display the plot
    plt.show()

def plot_completion(puzzle_sizes, *args, metric_name, ylabel, title):
    """Function to plot completion as a histogram for all algorithms."""
    # Convert puzzle sizes to index values for plotting
    x = np.arange(len(puzzle_sizes))
    
    # Define width of bars for each algorithm (adjust if necessary)
    width = 0.15
    
    # Create subplots for each algorithm
    plt.figure(figsize=(10, 6))
    
    labels = ['breadth_first_search', 'depth_first_search', 'local_search', 
              'backtracking_search', 'genetic_search']
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Plot bars for each algorithm
    for i, data in enumerate(args):
        # Bar chart for completeness values
        plt.bar(x + i * width, data, width, label=labels[i], color=colors[i], edgecolor='black')
    
    # Set labels and title
    plt.xlabel("Puzzle Size")
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Set x-axis tick labels
    plt.xticks(x + width * 2, puzzle_sizes)
    
    # Display legend
    plt.legend()
    
    # Display the plot
    plt.show()

# Plot for time complexity
plot_metric(puzzle_sizes, 
            breadth_first_search_time, depth_first_search_time, local_search_time, 
            backtracking_search_time, genetic_search_time,
            metric_name='Time', ylabel='Average Time Complexity (seconds)', 
            title='Algorithm Time Complexity vs Puzzle Size')

# Plot for space complexity
plot_metric(puzzle_sizes, 
            breadth_first_search_space, depth_first_search_space, local_search_space, 
            backtracking_search_space, genetic_search_space,
            metric_name='Space', ylabel='Average Space Complexity (MB)', 
            title='Algorithm Space Complexity vs Puzzle Size')

# Plot for completion (accuracy) using histogram
plot_completion(puzzle_sizes, 
                  breadth_first_search_completeness, depth_first_search_completeness, local_search_completeness, 
                  backtracking_search_completeness, genetic_search_completeness,
                  metric_name='Completeness', ylabel='Completeness (accuracy)', 
                  title='Algorithm Completeness (Accuracy) vs Puzzle Size')