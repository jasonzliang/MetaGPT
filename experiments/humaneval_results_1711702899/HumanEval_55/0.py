
def fib(n: int) -> int:
    """Return n-th Fibonacci number.

    Args:
        n (int): The position of the desired Fibonacci number in the sequence.

    Returns:
        int: The n-th Fibonacci number.

    Raises:
        ValueError: If input is less than 1.
    """
    if n < 1:
        raise ValueError("Input should be greater than or equal to 1.")
    
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
        
    return b
