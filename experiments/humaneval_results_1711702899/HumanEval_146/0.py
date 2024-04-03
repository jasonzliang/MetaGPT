
def specialFilter(nums):
    """This function takes an array of numbers as input and returns 
    the number of elements in the array that are greater than 10 and both 
    first and last digits of a number are odd (1, 3, 5, 7, 9).

    Args:
        nums: An array of numbers.

    Returns:
        The count of elements in the array that meet the specified conditions.

    Raises:
        TypeError: If input is not a list or if any element of the list is not an integer.
    """
    
    # Checking if nums is a list and all elements are integers
    if not isinstance(nums, list) or not all(isinstance(i, int) for i in nums):
        raise TypeError("Input should be a list of integers")
        
    count = 0
    
    # Iterating over each number in the array
    for num in nums:
        str_num = str(abs(num))  # Converting to string to get first and last digits
        
        # Checking if number is greater than 10, first digit is odd and last digit is odd
        if num > 10 and int(str_num[0]) % 2 != 0 and int(str_num[-1]) % 2 != 0:
            count += 1
            
    return count
