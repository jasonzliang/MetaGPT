
def is_sorted(lst):
    """
    Given a list of numbers, return whether or not they are sorted in ascending order. 
    If list has more than 1 duplicate of the same number, return False. 
    Assume no negative numbers and only integers.
    """
    
    # Check if list is empty or contains only one element
    if len(lst) < 2:
        return True
    
    # Check for duplicates in list
    if len(set(lst)) != len(lst):
        return False
    
    # Check if list is sorted in ascending order
    for i in range(len(lst) - 1):
        if lst[i] > lst[i + 1]:
            return False
            
    return True
