
def eat(number: int, need: int, remaining: int) -> list[int]:
    """Calculate the total number of eaten carrots after meals and the remaining carrots.

    Args:
        number: The number of carrots that have been eaten.
        need: The number of carrots needed to be eaten.
        remaining: The number of remaining carrots in stock.

    Returns:
        A list containing the total number of eaten carrots after meals and the remaining carrots.

    Raises:
        ValueError: If any of the parameters is not an integer or if they are out of range.
    """
    
    # Checking if all inputs are integers
    if not all(isinstance(i, int) for i in [number, need, remaining]):
        raise ValueError("All input values must be integers.")
        
    # Checking if all inputs are within the specified range
    if not all(0 <= i <= 1000 for i in [number, need, remaining]):
        raise ValueError("All input values must be between 0 and 1000 inclusive.")
    
    total_eaten = number + min(need, remaining)
    remaining -= min(need, remaining)
    
    if remaining < 0:
        remaining = 0
        
    return [total_eaten, remaining]
