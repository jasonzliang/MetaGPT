
def add_even_elements_at_odd_indices(lst):
    """Given a non-empty list of integers lst, returns the sum of even elements that are at odd indices.

    Args:
        lst: A list of integers.

    Returns:
        The sum of even elements at odd indices in the list.

    Raises:
        ValueError: If the input is not a list or if it contains non-integer values.
    """
    
    # Checking if lst is a list and if all its elements are integers
    if not isinstance(lst, list) or not all(isinstance(i, int) for i in lst):
        raise ValueError("Input should be a non-empty list of integers")
    
    # Checking if the list is empty
    if len(lst) == 0:
        return 0
    
    # Adding up even elements at odd indices
    result = 0
    for i in range(len(lst)):
        if i % 2 != 0 and lst[i] % 2 == 0:
            result += lst[i]
            
    return result
