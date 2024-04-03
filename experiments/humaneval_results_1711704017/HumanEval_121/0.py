
def solution(lst):
    """Given a non-empty list of integers, return the sum of all of the odd elements that are in even positions.

    Examples
    solution([5, 8, 7, 1]) ==> 12
    solution([3, 3, 3, 3, 3]) ==> 9
    solution([30, 13, 24, 321]) ==>0
    """
    
    # Initialize the sum to zero.
    odd_sum = 0
    
    # Iterate over the list with step size of 2 (even positions).
    for i in range(0, len(lst), 2):
        # Check if the element at even position is odd.
        if lst[i] % 2 != 0:
            # If it's odd, add it to the sum.
            odd_sum += lst[i]
    
    return odd_sum
