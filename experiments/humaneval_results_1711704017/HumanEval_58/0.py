
def common(l1: list, l2: list) -> list:
    """Return sorted unique common elements for two lists.
    >>> common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121])
    [1, 5, 653]
    >>> common([5, 3, 2, 8], [3, 2])
    [2, 3]
    """
    # Check if both inputs are lists
    if not isinstance(l1, list) or not isinstance(l2, list):
        raise ValueError("Both inputs must be of type 'list'")
    
    # Convert the lists to sets for efficient intersection operation
    set_l1 = set(l1)
    set_l2 = set(l2)
    
    # Find common elements and convert back to list
    common_elements = list(set_l1 & set_l2)
    
    # Sort the list of common elements
    common_elements.sort()
    
    return common_elements
