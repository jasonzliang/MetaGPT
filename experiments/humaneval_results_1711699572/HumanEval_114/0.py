
def minSubArraySum(nums):
    current_sum = nums[0]
    min_sum = nums[0]

    for i in range(1, len(nums)):
        current_sum = min(current_sum + nums[i], nums[i])
        min_sum = min(min_sum, current_sum)

    return min_sum
