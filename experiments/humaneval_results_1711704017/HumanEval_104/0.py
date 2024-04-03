
def unique_digits(x):
    """Given a list of positive integers x. Return a sorted list of all 
    elements that has no even digit.

    Note: Returned list should be sorted in increasing order.
    
    For example:
    >>> unique_digits([15, 33, 1422, 1])
    [1, 15, 33]
    >>> unique_digits([152, 323, 1422, 10])
    []
    """
    def has_even_digit(n):
        """Check if a number contains an even digit."""
        return any(int(d) % 2 == 0 for d in str(n))
    
    # Filter out numbers with even digits and sort the result.
    return sorted(n for n in x if not has_even_digit(n))
