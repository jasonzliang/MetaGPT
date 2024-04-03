
def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    # The number of n-digit positive integers starting or ending with 1 is equal to the count of (n-1)-digit numbers plus 2.
    # This is because we can add either '1' at both ends or '1' at one end and any other digit at the other end.
    
    return starts_one_ends(n - 1) + 2 if n > 1 else 2
