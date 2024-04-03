
def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    elif n == 2:
        return True
    else:
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

def intersection(interval1, interval2):
    """Determine whether the length of intersection of two intervals is a prime number."""
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    if start > end:  # If there's no intersection
        return "NO"
    
    length_intersection = end - start + 1
    return "YES" if is_prime(length_intersection) else "NO"
