
def get_max_triples(n):
    """
    This function generates an array of length n where each element is i^2 - i + 1 for 1 <= i <= n.
    It then counts the number of triples (a[i], a[j], a[k]) such that i < j < k and a[i] + a[j] + a[k] is a multiple of 3.

    Args:
        n: A positive integer representing the length of the array to be generated.

    Returns:
        The number of valid triples in the form of (a[i], a[j], a[k]) where i < j < k and a[i] + a[j] + a[k] is a multiple of 3.

    Raises:
        ValueError: If n is not a positive integer.
    """
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
        
    # Generate the array
    arr = [i**2 - i + 1 for i in range(1, n+1)]
    
    # Initialize counter
    count = 0
    
    # Iterate over all possible triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if (arr[i] + arr[j] + arr[k]) % 3 == 0:
                    count += 1
    
    return count
