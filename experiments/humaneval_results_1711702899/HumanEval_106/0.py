
def f(n):
    """Compute a list of size n where the value at index i is either the factorial of i if i is even or the sum of numbers from 1 to i if i is odd.

    Args:
        n (int): The size of the list to be computed.

    Returns:
        list: A list of size n where each element at index i is either the factorial of i if i is even or the sum of numbers from 1 to i if i is odd.

    Raises:
        ValueError: If input parameter n is not a positive integer.
    """
    
    # Check that n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
        
    result = []
    for i in range(1, n+1):
        if i % 2 == 0: # If index is even
            factorial = 1
            for j in range(1, i+1):
                factorial *= j
            result.append(factorial)
        else: # If index is odd
            sum_of_numbers = sum(range(i+1))
            result.append(sum_of_numbers)
            
    return result
