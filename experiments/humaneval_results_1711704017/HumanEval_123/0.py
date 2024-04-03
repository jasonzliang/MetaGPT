
def get_odd_collatz(n):
    """
    Given a positive integer n, return a sorted list that has the odd numbers in collatz sequence.
    """
    # Error handling for invalid inputs
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    # Initialize collatz sequence with starting number
    collatz_sequence = [n]
    
    # Generate the Collatz sequence
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        collatz_sequence.append(n)
        
    # Filter out even numbers and sort the list in increasing order
    odd_collatz_sequence = sorted([num for num in collatz_sequence if num % 2 != 0])
    
    return odd_collatz_sequence
```

# Example usage:
# print(get_odd_collatz(5))  # Outputs: [1, 5]
