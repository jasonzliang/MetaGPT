
def car_race_collision(n: int) -> int:
    """
    Calculate the number of collisions between two sets of cars moving in opposite directions on a straight line.

    Args:
        n: The number of cars in each set.

    Returns:
        The total number of collisions that would occur if the cars kept moving at the same speed and did not stop due to collisions.

    Raises:
        ValueError: If the input is less than 1 or not an integer.
    """
    # Check for valid input
    if not isinstance(n, int) or n < 1:
        raise ValueError("Input must be a positive integer.")
    
    # Since each car from one set collides with every car from the other set, 
    # the total number of collisions would be the product of the two sets.
    return n * n
