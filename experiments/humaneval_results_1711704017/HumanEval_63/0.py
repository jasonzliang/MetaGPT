
def fibfib(n: int) -> int:
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    This function efficiently computes the n-th element of the FibFib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    elif n == 0 or n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        # Initialize the first three FibFib numbers.
        fibfib_numbers = [0, 0, 1]
        
        # Compute each subsequent FibFib number from the previous three.
        for i in range(3, n + 1):
            next_number = sum(fibfib_numbers)
            
            # Keep only the last three numbers to avoid unnecessary memory usage.
            fibfib_numbers[i % 3] = next_number
        
        return fibfib_numbers[n % 3]
