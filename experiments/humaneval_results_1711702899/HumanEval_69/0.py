
def search(lst):
    """Return the greatest integer greater than zero that has a frequency 
       greater than or equal to its value in the given list. If no such number exists, return -1.

    Args:
        lst (list of int): A non-empty list of positive integers.

    Returns:
        int: The greatest integer that meets the criteria described above. 
             If no such number exists, returns -1.

    Raises:
        ValueError: If the input is not a list or if it contains non-positive integers.
    """
    
    # Checking if lst is a list and if all elements are positive integers
    if not isinstance(lst, list) or not all(isinstance(i, int) and i > 0 for i in lst):
        raise ValueError("Input should be a non-empty list of positive integers")
    
    # Counting the frequency of each number in the list
    freq = {num: lst.count(num) for num in set(lst)}
    
    # Finding numbers that meet the criteria and returning the maximum one
    result = max([num for num, count in freq.items() if num <= count], default=-1)
    
    return result
