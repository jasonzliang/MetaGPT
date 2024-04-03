
def minSubArraySum(nums):
    """
    Given an array of integers nums, find the minimum sum of any non-empty sub-array
    of nums.

    Args:
        nums: A list of integers.

    Returns:
        The smallest sum of a sub-array in nums.

    Raises:
        ValueError: If the input is not a list or if it contains non-integer values.
    """
    
    # Checking if the input is valid
    if not isinstance(nums, list):
        raise ValueError("The input should be a list.")
    for num in nums:
        if not isinstance(num, int):
            raise ValueError("All elements of the list should be integers.")
    
    # Initializing minimum sum to infinity and current sum to 0
    min_sum = float('inf')
    curr_sum = 0
    
    for num in nums:
        # If adding the current number makes the current sum less than 0, then we start a new sub-array from this number
        if curr_sum > 0:
            curr_sum = num
        else:
            curr_sum += num
        
        # Updating minimum sum if necessary
        min_sum = min(min_sum, curr_sum)
    
    return min_sum
