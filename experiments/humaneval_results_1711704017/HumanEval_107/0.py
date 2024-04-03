
def even_odd_palindrome(n):
    """
    Given a positive integer n, return a tuple that has the number of even and odd
    integer palindromes that fall within the range(1, n), inclusive.
    """
    if not isinstance(n, int) or n < 1 or n > 10**3:
        raise ValueError("Input must be an integer between 1 and 10^3")
    
    even_count = 0
    odd_count = 0

    for i in range(1, n + 1):
        if str(i) == str(i)[::-1]: # check if palindrome
            if i % 2 == 0:
                even_count += 1
            else:
                odd_count += 1
    
    return (even_count, odd_count)
