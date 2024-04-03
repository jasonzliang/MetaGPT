
def fibfib(n: int) -> int:
    """Computes the n-th element of the FibFib number sequence.

    The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).

    Args:
        n: The index of the element in the FibFib sequence.

    Returns:
        The n-th element of the FibFib sequence.

    Raises:
        ValueError: If n is less than 0.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    
    # Initialize base cases
    fibfib_sequence = [0, 0, 1] + [0] * (n - 2)
    
    for i in range(3, n + 1):
        fibfib_sequence[i] = fibfib_sequence[i-1] + fibfib_sequence[i-2] + fibfib_sequence[i-3]
        
    return fibfib_sequence[n]
