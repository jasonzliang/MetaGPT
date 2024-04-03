
def tribonacci_sequence(n):
    """Generate the first n+1 numbers of the Tribonacci sequence."""
    # Initialize the list with the first three elements
    trib = [3, 2, 8]

    # Generate the rest of the sequence up to n
    for i in range(3, n + 1):
        if i % 2 == 0:
            trib.append(int(1 + i / 2))
        else:
            trib.append(trib[i - 1] + trib[i - 2] + trib[i])
    return trib[:n+1]

# Example usage:
print(tribonacci_sequence(3))
