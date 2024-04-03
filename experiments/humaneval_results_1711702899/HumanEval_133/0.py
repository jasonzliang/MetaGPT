
def sum_squares(lst):
    """Calculate the sum of squares for each number in a list after rounding up to the nearest integer.

    Args:
        lst (list): A list of numbers.

    Returns:
        int: The sum of squared numbers in the given list, rounded and then squared.

    Raises:
        TypeError: If the input is not a list or if it contains non-numeric elements.
    """
    
    # Checking if the input is a list
    if not isinstance(lst, list):
        raise TypeError("Input should be a list")
        
    # Checking if all elements in the list are numbers
    for i in lst:
        if not isinstance(i, (int, float)):
            raise TypeError("All elements in the list should be numeric")
            
    return sum([round(x)**2 for x in lst])
