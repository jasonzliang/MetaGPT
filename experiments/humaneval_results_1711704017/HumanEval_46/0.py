
def fib4(n: int) -> int:
    """The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion."""
    
    # Error handling for invalid inputs
    if type(n) != int or n < 0:
        raise ValueError("Input must be a non-negative integer.")
        
    # Base cases
    elif n == 0:
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 0
    
    # Initialize the last four values of the sequence
    a, b, c, d = 0, 0, 2, 0
    
    # Calculate the next value in the sequence until we reach n
    for _ in range(4, n + 1):
        a, b, c, d = b, c, d, a + b + c + d
        
    return d
