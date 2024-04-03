
def largest_smallest_integers(lst):
    """
    Create a function that returns a tuple (a, b), where 'a' is the largest of negative integers, 
    and 'b' is the smallest of positive integers in a list. If there is no negative or positive integers, 
    return them as None.

    Args:
        lst: A list of integers.

    Returns:
        A tuple (a, b) where 'a' is the largest negative integer and 'b' is the smallest positive integer in the list.
        If there are no negative or positive integers, return them as None.

    Raises:
        ValueError: If the input is not a list of integers.
    """
    
    if not all(isinstance(i, int) for i in lst):
        raise ValueError("All elements in the list must be integers.")
        
    negatives = [num for num in lst if num < 0]
    positives = [num for num in lst if num > 0]
    
    a = max(negatives) if negatives else None
    b = min(positives) if positives else None
    
    return (a, b)
