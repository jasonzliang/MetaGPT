
def get_odd_collatz(n):
    """
    Given a positive integer n, returns a sorted list of the odd numbers in the Collatz sequence.

    Args:
        n: A positive integer.

    Returns:
        A sorted list of the odd numbers in the Collatz sequence for n.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if type(n) != int or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    collatz_sequence = []
    while n != 1:
        collatz_sequence.append(n)
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
    collatz_sequence.append(1)
    
    odd_collatz_numbers = sorted([i for i in collatz_sequence if i % 2 != 0])
    
    return odd_collatz_numbers
