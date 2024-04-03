
def rounded_avg(n: int, m: int) -> str:
    """Computes the average of integers from n through m (inclusive), 
    rounds to nearest integer, and converts that to binary. If n > m, returns -1.

    Args:
        n: The first positive integer.
        m: The second positive integer.

    Returns:
        A string representing the binary representation of the rounded average 
        if n <= m, otherwise -1.

    Raises:
        ValueError: If either n or m is not a positive integer.
    """
    # Check that both inputs are positive integers
    if not (isinstance(n, int) and isinstance(m, int) and n > 0 and m > 0):
        raise ValueError("Both inputs must be positive integers.")
    
    # If n > m, return -1
    if n > m:
        return -1
    
    # Compute the average of the integers from n through m (inclusive)
    avg = sum(range(n, m + 1)) / ((m - n) + 1)
    
    # Round to nearest integer and convert to binary
    rounded_avg_bin = bin(round(avg))
    
    return rounded_avg_bin
