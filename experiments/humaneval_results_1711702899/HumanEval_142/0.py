
def sum_squares(lst):
    """This function takes a list of integers as input. It squares the integer entry if its index is a multiple of 3 and cubes the integer entry if its index is a multiple of 4 but not a multiple of 3. The function then returns the sum of all entries in the list.

    Args:
        lst (list): A list of integers.

    Returns:
        int: The sum of all entries in the list after applying the specified operations.

    Raises:
        TypeError: If the input is not a list or if any element in the list is not an integer.
    """
    
    # Checking if the input is a list and if each element in the list is an integer
    if not isinstance(lst, list):
        raise TypeError("Input should be a list")
    for i in lst:
        if not isinstance(i, int):
            raise TypeError("All elements in the list should be integers")
    
    # Initializing sum to 0
    total_sum = 0
    
    # Iterating over each element in the list with its index
    for i, val in enumerate(lst):
        if i % 3 == 0:
            # If index is a multiple of 3, square the value and add to sum
            total_sum += val ** 2
        elif i % 4 == 0:
            # If index is a multiple of 4 but not 3, cube the value and add to sum
            total_sum += val ** 3
    
    return total_sum
