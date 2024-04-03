
def tribonacci(n):
    """Computes the first n + 1 numbers of the Tribonacci sequence.

    Args:
        n: A non-negative integer representing the number of elements to compute in the sequence.

    Returns:
        A list containing the first n + 1 numbers of the Tribonacci sequence.

    Raises:
        ValueError: If the input is not a non-negative integer.
    """
    if type(n) != int or n < 0:
        raise ValueError("Input must be a non-negative integer.")
    
    # Initialize base cases
    tribonacci_sequence = [1, 3] + [0] * (n - 1)
    
    for i in range(2, n + 1):
        if i % 2 == 0:
            tribonacci_sequence[i] = 1 + i // 2
        else:
            tribonacci_sequence[i] = tribonacci_sequence[i - 1] + tribonacci_sequence[i - 2] + tribonacci_sequence[i + 1]
    
    return tribonacci_sequence[:n+1]
