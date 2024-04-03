
def is_palindrome(q):
    """Check if a list q is palindromic."""
    return q == q[::-1]

def will_it_fly(q, w):
    """
    Check if an object q will fly. 
    An object will fly if it's balanced (it is a palindromic list) and the sum of its elements is less than or equal to the maximum possible weight w.
    """
    try:
        return is_palindrome(q) and sum(q) <= w
    except TypeError as e:
        print("Invalid input: ", str(e))
