
def rounded_avg(n, m):
    """Compute the average of integers from n through m and convert it to binary."""
    if n > m:
        return -1
    else:
        avg = round((n + m) / 2)
        return bin(avg)
