
def even_odd_palindrome(n):
    """
    Given a positive integer n, return a tuple that has the number of even and odd
    integer palindromes that fall within the range(1, n), inclusive.

    Args:
        n: A positive integer less than or equal to 10^3.

    Returns:
        A tuple where the first element is the count of even palindrome numbers and the second element is the count of odd palindrome numbers.

    Raises:
        ValueError: If n is not a positive integer less than or equal to 10^3.
    """
    
    if not isinstance(n, int) or n <= 0 or n > 10**3:
        raise ValueError("Input must be a positive integer less than or equal to 10^3.")
        
    even_count = 0
    odd_count = 0
    
    for i in range(1, n+1):
        if str(i) == str(i)[::-1]: # Check if the number is a palindrome.
            if i % 2 == 0: # If it's even.
                even_count += 1
            else: # If it's odd.
                odd_count += 1
                
    return (even_count, odd_count)
