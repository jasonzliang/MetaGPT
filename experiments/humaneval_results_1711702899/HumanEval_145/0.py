
def order_by_points(nums):
    """
    This function sorts the given list of integers in ascending order according to 
    the sum of their digits. If there are several items with similar sum of their 
    digits, they are ordered based on their index in original list.

    Args:
        nums (list): A list of integers.

    Returns:
        list: The sorted list according to the problem's specifications.

    Raises:
        TypeError: If the input is not a list.
        ValueError: If the list contains non-integer values.
    """
    
    # Check if nums is a list
    if not isinstance(nums, list):
        raise TypeError("Input should be a list.")
        
    # Check if all elements in nums are integers
    for i in nums:
        if not isinstance(i, int):
            raise ValueError("All elements in the input list should be integers.")
    
    # Sort the list based on the sum of digits and their original index
    return sorted(nums, key=lambda x: (sum(int(digit) for digit in str(abs(x))), nums.index(x)))
