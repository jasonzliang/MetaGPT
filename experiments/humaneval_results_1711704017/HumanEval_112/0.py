
def reverse_delete(s, c):
    """
    We are given two strings s and c, you have to deleted all the characters in s that are equal to any character in c
    then check if the result string is palindrome.
    A string is called palindrome if it reads the same backward as forward.
    You should return a tuple containing the result string and True/False for the check.
    """
    
    # Remove characters from s that are in c
    new_s = ''.join([char for char in s if char not in c])
    
    # Check if the resulting string is a palindrome
    is_palindrome = new_s == new_s[::-1]
    
    return (new_s, is_palindrome)
