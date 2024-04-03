
def max_fill(grid, capacity):
    # Calculate the total water in each row and sort them in descending order.
    water = sorted([sum(row) for row in grid], reverse=True)
    # Initialize the number of times to lower the buckets.
    count = 0
    # Iterate over the sorted list of water amounts.
    for w in water:
        # If there is still water left and we can fill the bucket,
        while w > 0 and capacity > 0:
            # Fill the bucket with as much water as possible.
            if w >= capacity:
                w -= capacity
            else:
                w = 0
            # Increase the number of times to lower the buckets.
            count += 1
    return count
