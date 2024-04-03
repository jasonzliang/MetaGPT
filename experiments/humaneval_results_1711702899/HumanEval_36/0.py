
def fizz_buzz(n: int) -> int:
    """Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.

    Args:
        n: The upper limit for the range of numbers to check (non-inclusive).

    Returns:
        The total count of the digit 7 that appear in numbers divisible by 11 or 13 and less than n.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    count = 0
    for i in range(1, n):
        if (i % 11 == 0 or i % 13 == 0) and '7' in str(i):
            count += str(i).count('7')
            
    return count
