
def choose_num(x, y):
    """This function takes two positive numbers x and y and returns the biggest even integer number that is in the range [x, y] inclusive. If there's no such number, then the function should return -1."""
    
    # Check if inputs are valid
    if not (isinstance(x, int) and isinstance(y, int)):
        raise ValueError("Both x and y must be integers.")
    if x < 0 or y < 0:
        raise ValueError("Both x and y must be positive numbers.")
    
    # Check if there's a number in the range [x, y] that is even
    for i in range(y, x - 1, -1):
        if i % 2 == 0:
            return i
            
    # If no such number exists, return -1
    return -1
