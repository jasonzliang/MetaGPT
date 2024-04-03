
def order_by_points(nums):
    """
    Sort the given list of integers in ascending order according to the sum of their digits. 
    If there are several items with similar sum of their digits, order them based on their index in original list.
    """
    
    # Define a helper function to calculate the sum of digits
    def digit_sum(n):
        return sum(int(digit) for digit in str(abs(n)))

    # Sort the numbers by their digit sums and their indices
    nums.sort(key=lambda x: (digit_sum(x), nums.index(x)))
    
    return nums
