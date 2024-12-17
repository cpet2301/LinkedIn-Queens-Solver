import cv2
from PIL import Image
import pyautogui
import numpy as np
import search_algorithms

def detect_puzzle():
    """
    Detect a LinkedIn Queens puzzle from a screenshot.
    
    Captures the screen, processes the image to detect edges and lines, and identifies square shapes
    representing the puzzle grid. It then crops and saves the detected puzzle region for further analysis.

    Parameters:
        None
     
    Returns:
        None
    """
    # Step 1: Capture the screen
    screen = pyautogui.screenshot()
    screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

    # Step 2: Convert to grayscale and apply Gaussian blur
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_gray = cv2.GaussianBlur(screen_gray, (5, 5), 0)

    # Step 3: Use Canny edge detection to detect edges
    edges = cv2.Canny(screen_gray, 100, 200)

    # Step 4: Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(screen, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Draw detected lines in black

    # Step 5: Find contours to detect square shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Filter contours to only include approximate squares
    max_area = 0
    best_bbox = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = abs(w - h) / float(max(w, h))
        area = w * h

        if aspect_ratio < 0.1 and area > max_area:
            max_area = area
            best_bbox = (x, y, w, h)

    # Step 7: Save the best bounding box
    if best_bbox:
        x, y, w, h = best_bbox
        cropped_puzzle = screen[y:y + h, x:x + w]

        cv2.imwrite("detected_puzzle.png", cropped_puzzle)
        print(f"Puzzle cropped and saved as detected_puzzle.png at ({x}, {y}, {w}, {h})")
    else:
        print("No squares detected.")

def grid_count(img):
    """
    Count the number of rows and columns in a grid-like image.
    
    Analyzes a grid image by converting it to grayscale, applying thresholding, and detecting horizontal and vertical lines. The function attempts to find the number of rows and columns by searching for continuous pixel runs.
    
    Parameters:
        img (numpy.ndarray or PIL.Image): The input grid image. If it's a NumPy array (OpenCV image), it will be converted to grayscale. If it's a PIL image, it will be converted to a NumPy array.
    
    Returns:
        int: The highest count of rows and columns detected, serving as a failsafe measure in case one of them is inaccurate.
    """
    # Step 1: Convert to grayscale NumPy array if needed
    if isinstance(img, np.ndarray):  # If OpenCV image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:  # If PIL Image
        img = np.array(img.convert("L"))

    # Step 2: Define image thresholds and get resolution
    img[img > 100] = 255
    img[img <= 100] = 0

    height, width = img.shape
    row_count, col_count = 0, 0
    threshold_run = 20
    run_length = 0

    # Step 4: Detect horizontal grid lines (rows)
    while row_count == 0:
        random_col = np.random.randint(10, width - 20)
        for x in range(height):
            if img[x, random_col]:
                run_length += 1
            else:
                if run_length >= threshold_run:
                    row_count += 1
                run_length = 0

    run_length = 0

    # Step 4: Detect vertical grid lines (columns)
    while col_count == 0:
        random_row = np.random.randint(10, height - 20)
        for y in range(width):
            if img[random_row, y]:
                run_length += 1
            else:
                if run_length >= threshold_run:
                    col_count += 1
                run_length = 0

    # Step 5: Return the highest grid count (failsafe in case one of them is inaccurate)
    return max(row_count, col_count)

def extract_grid_colors(img):
    """
    Extract a grid-based representation of colors from an image.
    
    Extracts the grid by sampling the color at the center of each grid cell and mapping each cell to a unique color index.
    
    Parameters:
        img (PIL.Image): The input grid image from which the colors need to be extracted.
    
    Returns:
        numpy.ndarray: A 2D grid array where each cell contains an integer representing the color index. Dominant colors are mapped to unique indices, while black (background) is reserved as a special index.
    """
    # Step 1: Determine grid size (number of rows and columns)
    grid_size = grid_count(img)  # Use the earlier function to determine grid dimensions
    image_width, image_height = img.size  # Image dimensions

    # Step 2: Identify dominant solid colors in the image
    max_colors = 10000  # Limit to prevent overflows for large images
    color_frequencies = img.getcolors(maxcolors=max_colors)  # Count pixels for each color
    
    # Step 3: Sort colors by frequency, prioritize black (RGB: (0, 0, 0))
    color_frequencies.sort(key=lambda item: 1e10 if sum(item[1]) == 0 else item[0], reverse=True)
    dominant_colors = color_frequencies[:grid_size + 1]  # Keep most frequent colors, enough for the grid
    dominant_colors = dominant_colors[1:]  # Exclude black (assumed as the background color)

    # Step 4: Create a color-to-index mapping
    color_to_index = {}
    for index, (_, rgb_color) in enumerate(dominant_colors):
        color_to_index[rgb_color] = index  # Assign an index to each RGB color
    color_to_index[(0, 0, 0)] = grid_size  # Reserve the last index for black (background)

    # Step 5: Estimate square size (width and height of each grid cell)
    square_size = image_width // grid_size

    # Step 6: Initialize the grid color board
    grid_color_board = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Step 7: Sample colors at approximate midpoints of each grid cell
    for row in range(grid_size):  # Iterate over rows
        sample_y = square_size // 2 + row * square_size  # Approximate midpoint Y-coordinate for the row
        for col in range(grid_size):  # Iterate over columns
            sample_x = square_size // 2 + col * square_size  # Approximate midpoint X-coordinate for the column

            shift_direction = 1  # Pixel shift direction for artifact handling
            color_index = -1  # Initialize color index as invalid
            attempts = 0  # Retry counter
            max_attempts = 10  # Maximum retries to avoid infinite loops

            # Handle compression artifacts by adjusting sample point
            while color_index == -1 and attempts < max_attempts:  # Retry loop with limit
                # Ensure sampling point is within image bounds
                if 0 <= sample_x < image_width and 0 <= sample_y < image_height:
                    sampled_color = img.getpixel((sample_x, sample_y))  # Sample color at current point
                    color_index = color_to_index.get(sampled_color, -1)  # Check if it's a known color
                else:
                    break  # Exit if sampling point goes out of bounds

                # Adjust for artifacts or black background
                if color_index == grid_size:  # If color is black (edge case)
                    sample_x -= shift_direction  # Revert shift
                    shift_direction = -shift_direction  # Change direction
                    color_index = -1  # Reset to continue searching
                else:
                    sample_x += shift_direction  # Shift sample point horizontally

                attempts += 1  # Increment retry counter

            # Fallback if no valid color is found after retries
            if color_index == -1:
                color_index = grid_size  # Default to black (background)

            # Assign the color index to the grid
            grid_color_board[row, col] = color_index

    return grid_color_board

def create_state_space(tile_grid):
    """
    Transform the input color grid into a state space grid where each cell contains a tuple (is_queen, region_id).
    
    Each cell in the new grid contains:
        - is_queen (int): Initially 0, as no queens are placed in the grid.
        - region_id (int): The color identifier representing the region from the input grid.

    Parameters:
        tile_grid (list of list of int): The input 2D grid containing region IDs representing colors.
        
    Returns:
        state_grid (2D list of tuples): A 2D tuple array where each cell contains:
            - is_queen (int): Initially set to 0, indicating no queen placement.
            - region_id (int): The color identifier representing the region.
    """
    state_grid = []
    is_queen = 0

    for row in tile_grid:
        new_row = []
        for region_id in row:
            new_row.append((is_queen, region_id))  # No queen placed, represented by 0
        state_grid.append(new_row)

    return state_grid

def display_grid(state_grid):
    """
    Display the state grid with consistent spacing and borders for neatness.
    
    Each cell in the grid contains a color ID and an optional queen marker (Q).
    The grid is displayed with uniform cell spacing and horizontal/vertical borders to maintain visual clarity.
    
    Parameters:
        state_grid (2D list of tuples): A 2D grid where each cell contains:
            - is_queen (int): Indicates whether a queen is placed (1) or not (0).
            - region_id (int): The color identifier representing the region.

    Returns:
        None: The function prints the grid directly to the console.
    """
    cell_width = 6  # Width for each cell (e.g., '10 (Q)' fits well)

    horizontal_border = '+' + ('-' * cell_width + '+') * len(state_grid[0])
    print(horizontal_border)

    for row in state_grid:
        row_str = '|'
        for cell in row:
            is_queen, region_id = cell
            if is_queen == 1:
                cell_content = f"{region_id}(Q)"
            else:
                cell_content = f"{region_id}"
            
            # Pad content to align each cell to the fixed width
            cell_str = f"{cell_content:<{cell_width}}"
            row_str += cell_str + '|'
        print(row_str)
        print(horizontal_border)

# Test
if __name__ == "__main__":
    # detect_puzzle()

    image_path = "sample_puzzle.png"
    cropped_image = cv2.imread(image_path)
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    state_grid = create_state_space(extract_grid_colors(cropped_image_pil))

    display_grid(state_grid)

    display_grid(search_algorithms.breadth_first_search(state_grid))
