
def fib(n: int) -> int:
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
    # Error handling for invalid inputs
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    elif n < 0:
        raise ValueError("Input must be a non-negative integer.")

    # Base cases
    if n == 0:
        return 0
    elif n == 1:
        return 1
    
    # Initialize variables for the last two Fibonacci numbers
    fib_n_2, fib_n_1 = 0, 1

    # Calculate the n-th Fibonacci number iteratively
    for _ in range(2, n + 1):
        fib_n = fib_n_2 + fib_n_1
        fib_n_2, fib_n_1 = fib_n_1, fib_n
    
    return fib_n_1
