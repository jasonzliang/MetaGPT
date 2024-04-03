
def filter_integers(values):
    """ Filter given list of any python values only for integers.
    >>> filter_integers(['a', 3.14, 5])
    [5]
    >>> filter_integers([1, 2, 3, 'abc', {}, []])
    [1, 2, 3]
    """
    # Initialize an empty list to store the integers
    ints = []

    # Iterate over each value in the input list
    for val in values:
        # Check if the current value is of integer type
        if isinstance(val, int):
            # If it is, append it to our list of integers
            ints.append(val)
    
    # Return the final list of integers
    return ints
