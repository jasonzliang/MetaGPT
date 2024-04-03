def fruit_distribution(s,n):
    s = s.split()
    total_fruits = int(s[0]) + int(s[2]) + int(s[4])
    return n - total_fruits