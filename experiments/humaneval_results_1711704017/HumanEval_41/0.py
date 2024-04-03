
def car_race_collision(n: int) -> int:
    """
    Calculate the number of collisions between two sets of cars moving in opposite directions.
    Each set contains n cars, and they start out very far from each other.
    All cars move at the same speed. Two cars collide when a car moving left to right hits a car moving right to left.
    However, the cars are infinitely sturdy and strong; as a result, they continue moving in their trajectory as if they did not collide.
    """
    # Check for invalid inputs
    if n < 0:
        raise ValueError("Number of cars cannot be negative.")
    
    # Calculate the number of collisions
    # Since each car from one set will collide with exactly one car from the other set, 
    # the total number of collisions is simply the product of the two sets.
    return n * n
