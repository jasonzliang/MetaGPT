
def eat(number, need, remaining):
    """
    Calculate the total number of eaten carrots and remaining carrots after meals.
    """
    # Check if inputs are valid
    if not all(0 <= i <= 1000 for i in [number, need, remaining]):
        raise ValueError("All input values must be between 0 and 1000.")
    
    # Calculate total eaten carrots
    total_eaten = number + min(need, remaining)
    
    # Calculate remaining carrots
    if need > remaining:
        remaining = 0
    else:
        remaining -= need
        
    return [total_eaten, remaining]
