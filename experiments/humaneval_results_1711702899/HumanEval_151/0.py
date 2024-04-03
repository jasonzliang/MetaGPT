
def double_the_difference(lst):
    """Given a list of numbers, return the sum of squares of the odd integers in the list. 
       Ignore negative numbers and non-integer numbers.

    Args:
        lst (list): A list of numbers.

    Returns:
        int: The sum of squares of the odd integers in the list.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list contains non-integer or negative numbers.
    """
    
    # Check if lst is a list
    if not isinstance(lst, list):
        raise TypeError("Input should be a list")
        
    # Initialize sum to 0
    total = 0
    
    for num in lst:
        # Check if the number is an integer and odd
        if isinstance(num, int) and num % 2 != 0:
            total += num ** 2
        elif not isinstance(num, int):
            raise ValueError("List should only contain integers")
        else:
            continue
            
    return total
