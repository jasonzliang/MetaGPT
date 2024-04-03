
def max_fill(grid, capacity):
    """
    Calculate the number of times to lower the buckets.

    Args:
        grid (list[list[int]]): A rectangular grid representing wells and their water levels.
        capacity (int): The capacity of each bucket.

    Returns:
        int: The number of times to lower the buckets.
    """
    # Initialize variables
    total_water = 0
    max_fill_count = 0
    
    # Calculate total water in all wells
    for row in grid:
        total_water += sum(row)
        
    # If there is no water, return 0
    if total_water == 0:
        return 0
    
    # Calculate the number of times to lower the buckets
    while total_water > 0:
        max_fill_count += 1
        total_water -= capacity
        
    return max_fill_count
