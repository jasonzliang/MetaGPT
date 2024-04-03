
def fib4(n: int) -> int:
    """Computes the n-th element of the fib4 number sequence.

    The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).

    Args:
        n: The index of the element in the sequence.

    Returns:
        The n-th element of the fib4 number sequence.

    Raises:
        ValueError: If n is less than 0.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    
    # Initialize base cases
    fib_sequence = [0, 0, 2, 0]
    
    # Compute the n-th element of the sequence
    for i in range(4, n + 1):
        fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2] + fib_sequence[i - 3] + fib_sequence[i - 4])
    
    return fib_sequence[n]
