
def monotonic(lst):
    """Return True if list elements are monotonically increasing or decreasing."""
    # Check for empty list
    if not lst:
        return True

    direction = None  # Initialize the direction to None

    for i in range(len(lst) - 1):
        # If direction is still undetermined, set it based on first two elements
        if direction is None:
            if lst[i] < lst[i + 1]:
                direction = 'increasing'
            elif lst[i] > lst[i + 1]:
                direction = 'decreasing'

        # If direction has been determined, check for violation of the rule
        else:
            if (direction == 'increasing' and lst[i] > lst[i + 1]) or \
               (direction == 'decreasing' and lst[i] < lst[i + 1]):
                return False

    # If we reached here, the list is monotonic
    return True
