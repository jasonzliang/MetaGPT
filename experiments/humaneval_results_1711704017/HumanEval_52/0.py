
def below_threshold(l: list, t: int) -> bool:
    """Return True if all numbers in the list l are below threshold t.
    >>> below_threshold([1, 2, 4, 10], 100)
    True
    >>> below_threshold([1, 20, 4, 10], 5)
    False
    """
    try:
        # Check if all elements in the list are less than threshold
        return all(x < t for x in l)
    except TypeError as e:
        print("Invalid input. Please provide a list of integers and an integer as arguments.")
        raise e
