
def max_fill(grid, capacity):
    """Calculate the maximum number of times a bucket can be lowered to empty wells.

    Args:
        grid: A 2D list representing the layout of wells and their water levels.
        capacity: An integer representing the capacity of each bucket.

    Returns:
        The maximum number of times a bucket can be lowered to empty wells.

    Raises:
        ValueError: If grid is not rectangular or if any element in grid is not 0 or 1.
        TypeError: If grid is not a list of lists, or if capacity is not an integer.
    """
    
    # Checking the type and value constraints
    if not all(isinstance(row, list) for row in grid):
        raise ValueError("Grid must be a 2D list")
        
    if not all(len(row) == len(grid[0]) for row in grid):
        raise ValueError("All rows in the grid must have the same length")
    
    if not isinstance(capacity, int):
        raise TypeError("Capacity must be an integer")
        
    if capacity < 1 or capacity > 10:
        raise ValueError("Capacity must be between 1 and 10 inclusive")
        
    if not all(cell == 0 or cell == 1 for row in grid for cell in row):
        raise ValueError("All cells in the grid must be either 0 or 1")
    
    # Calculating the maximum number of times a bucket can be lowered
    max_fill = 0
    for row in grid:
        water = sum(row)
        if water % capacity == 0:
            max_fill += water // capacity
        else:
            max_fill += water // capacity + 1
    
    return max_fill
