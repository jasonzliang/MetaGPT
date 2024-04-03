
def modp(n: int, p: int) -> int:
    """Return 2^n modulo p (be aware of numerics)."""
    # Check if n or p is negative
    if n < 0 or p <= 0:
        raise ValueError("Both n and p must be non-negative and p should be greater than 0.")
    
    # Calculate 2^n modulo p using the built-in pow function with three arguments
    return pow(2, n, p)
