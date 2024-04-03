
def max_fill(grid, capacity):
    # Initialize the total water and the number of times to lower the buckets.
    total_water = sum([sum(row) for row in grid])
    num_lowers = 0

    # While there is still water left, lower the bucket and fill it up.
    while total_water > 0:
        total_water -= capacity
        num_lowers += 1
    
    return num_lowers if total_water == 0 else num_lowers - 1
