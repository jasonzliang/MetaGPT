
def intersection(interval1, interval2):
    """Determine if the length of the intersection of two intervals is a prime number.

    Args:
        interval1: A tuple representing the first interval.
        interval2: A tuple representing the second interval.

    Returns:
        "YES" if the length of the intersection is a prime number, otherwise "NO".

    Raises:
        ValueError: If either interval is not a tuple or does not contain exactly two integers.
    """
    
    # Check that both intervals are tuples and contain exactly two integers
    for interval in (interval1, interval2):
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("Each interval must be a tuple of two integers.")
    
    # Calculate the intersection of the intervals
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    # If there is no intersection, return "NO"
    if start > end:
        return "NO"
    
    # Calculate the length of the intersection
    length = end - start + 1
    
    # Check if the length is a prime number
    for i in range(2, int(length ** 0.5) + 1):
        if length % i == 0:
            return "NO"
            
    return "YES"
