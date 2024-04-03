
def pairs_sum_to_zero(l):
    """
    pairs_sum_to_zero takes a list of integers as an input.
    It returns True if there are two distinct elements in the list that sum to zero, and False otherwise.
    """
    # Check for invalid inputs
    if not isinstance(l, list):
        raise ValueError("Input must be a list")
    if not all(isinstance(i, int) for i in l):
        raise ValueError("All elements of the input list must be integers")
    
    # Create a set to store seen numbers and check each number against it
    seen = set()
    for num in l:
        if -num in seen:
            return True
        seen.add(num)
        
    return False
