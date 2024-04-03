
def make_a_pile(n):
    """
    Given a positive integer n, this function creates a pile of stones where the first level has n stones. 
    The number of stones in each subsequent level is either the next odd or even number depending on whether n is odd or even respectively.
    
    Args:
        n (int): A positive integer representing the number of stones in the first level.
        
    Returns:
        list: A list where each element represents the number of stones in a level, starting from the second level.
        
    Raises:
        ValueError: If the input is not a positive integer.
    """
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    pile = [n]
    next_num = n + 1 if n % 2 == 0 else n + 2
    
    while len(pile) < n:
        pile.append(next_num)
        next_num += 2
        
    return pile
