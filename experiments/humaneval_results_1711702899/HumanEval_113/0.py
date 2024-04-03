
def odd_count(lst):
    """Given a list of strings, where each string consists of only digits, 
    returns a list where each element i is the number of odd elements in the
    string i of the input.

    Args:
        lst: A list of strings, where each string contains only digits.

    Returns:
        A list of strings where each string explains how many odd numbers are 
        there in the corresponding indexed string of the input list.

    Raises:
        TypeError: If the input is not a list or if any element of the list is not a string.
        ValueError: If any string contains non-digit characters.
    """
    
    # Checking if lst is a list and all elements are strings
    if not isinstance(lst, list):
        raise TypeError("Input should be a list.")
    for i in lst:
        if not isinstance(i, str):
            raise TypeError("All elements of the input list should be strings.")
    
    # Checking if all characters in each string are digits
    for i in lst:
        if not i.isdigit():
            raise ValueError("Each string should contain only digits.")
            
    result = []
    for i, s in enumerate(lst):
        odd_count = sum([1 for c in s if int(c) % 2 != 0])
        result.append(f"the number of odd elements {odd_count}n the str{odd_count}ng {odd_count} of the {odd_count}nput.")
    return result
