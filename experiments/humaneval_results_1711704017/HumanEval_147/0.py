
def get_max_triples(n):
    """
    Calculate the number of triples (a[i], a[j], a[k]) in an array where i < j < k and 
    a[i] + a[j] + a[k] is a multiple of 3.
    
    Parameters:
        n (int): The length of the array to be created.
        
    Returns:
        int: The number of valid triples in the array.
    """
    # Create an integer array 'a' with values i*i - i + 1 for 1 <= i <= n
    a = [i * i - i + 1 for i in range(1, n+1)]
    
    count = 0
    # Iterate over all possible triples (i, j, k) where i < j < k
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Check if the sum of a[i], a[j] and a[k] is a multiple of 3
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    return count
```

# Example usage:
print(get_max_triples(5))  # Outputs: 1
