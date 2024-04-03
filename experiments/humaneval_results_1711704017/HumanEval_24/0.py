
def largest_divisor(n: int) -> int:
    """ For a given number n, find the largest number that divides n evenly, smaller than n.
    >>> largest_divisor(15)
    5
    """
    if not isinstance(n, int):
        raise ValueError("Input must be an integer.")
    elif n <= 0:
        raise ValueError("Input must be a positive integer.")
    else:
        for i in range(n-1, 0, -1): # Start from n-1 and go down to 1
            if n % i == 0: # If n is divisible by i (i.e., remainder of division is 0)
                return i # Return the largest number that divides n evenly
