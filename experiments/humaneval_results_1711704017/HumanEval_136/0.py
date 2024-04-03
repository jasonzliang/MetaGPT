
def largest_smallest_integers(lst):
    """
    Create a function that returns a tuple (a, b), where 'a' is
    the largest of negative integers, and 'b' is the smallest
    of positive integers in a list.
    If there is no negative or positive integers, return them as None.
    """
    negatives = [i for i in lst if i < 0]
    positives = [i for i in lst if i > 0]
    
    # Check if the lists are empty and assign None if they are
    a = max(negatives) if negatives else None
    b = min(positives) if positives else None
    
    return (a, b)
