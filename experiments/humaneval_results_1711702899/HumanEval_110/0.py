
def exchange(lst1, lst2):
    """This function takes two lists of numbers as input and determines if it is possible to 
    perform an exchange operation between them to make all elements in the first list even.

    Args:
        lst1 (list): The first list of integers.
        lst2 (list): The second list of integers.

    Returns:
        str: "YES" if it is possible to exchange elements between the lists to make all 
             elements in the first list even, otherwise "NO".

    Raises:
        TypeError: If either lst1 or lst2 is not a list.
        ValueError: If either lst1 or lst2 contains non-integer values.
    """
    
    # Check if inputs are lists
    if not (isinstance(lst1, list) and isinstance(lst2, list)):
        raise TypeError("Both arguments must be of type 'list'")
        
    # Check if all elements in both lists are integers
    for i in lst1 + lst2:
        if not isinstance(i, int):
            raise ValueError("All elements in the list must be of type 'int'")
    
    # If there exists an odd number in lst1 and no even number in lst2, return "NO"
    for i in lst1:
        if i % 2 != 0:
            if all(j % 2 == 0 for j in lst2):
                return "NO"
    
    # If none of the above conditions are met, return "YES"
    return "YES"
