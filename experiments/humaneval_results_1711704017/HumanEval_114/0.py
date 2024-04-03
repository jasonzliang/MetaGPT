
def minSubArraySum(nums):
    """
    Given an array of integers nums, find the minimum sum of any non-empty sub-array
    of nums.
    Example
    minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
    minSubArraySum([-1, -2, -3]) == -6
    """
    
    # Initialize current_sum and min_sum to the first element of nums.
    current_sum = min_sum = nums[0]

    # Iterate over the rest of nums.
    for num in nums[1:]:
        # Update current_sum to be the maximum between the current number and the sum of current_sum and the current number.
        current_sum = max(num, current_sum + num)
        
        # Update min_sum to be the minimum between min_sum and current_sum.
        min_sum = min(min_sum, current_sum)
    
    return min_sum
