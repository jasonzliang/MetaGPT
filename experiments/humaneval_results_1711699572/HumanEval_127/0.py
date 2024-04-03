
def is_prime(n):
    if n <= 1 or (n % 2 == 0 and n > 2): 
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def intersection(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    if start > end:  # If the two intervals don't intersect
        return "NO"
    
    length_intersection = end - start + 1
    return "YES" if is_prime(length_intersection) else "NO"
