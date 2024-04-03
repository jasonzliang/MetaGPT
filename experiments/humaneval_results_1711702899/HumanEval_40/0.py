
def triples_sum_to_zero(lst):
    """
    This function takes a list of integers as an input. 
    It returns True if there are three distinct elements in the list that sum to zero, and False otherwise.

    Args:
        lst (list): A list of integers.

    Returns:
        bool: True if there are three distinct elements in the list that sum to zero, False otherwise.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list contains less than 3 elements.
    """
    
    # Checking if the input is a list
    if not isinstance(lst, list):
        raise TypeError("The input should be a list.")
        
    # Checking if the list has at least three elements
    if len(lst) < 3:
        raise ValueError("The list should have at least three elements.")
    
    # Sorting the list in ascending order
    lst.sort()
    
    for i in range(len(lst)-2):
        left = i + 1
        right = len(lst) - 1
        
        while left < right:
            if lst[i] + lst[left] + lst[right] == 0:
                return True
            elif lst[i] + lst[left] + lst[right] < 0:
                left += 1
            else:
                right -= 1
                
    return False
