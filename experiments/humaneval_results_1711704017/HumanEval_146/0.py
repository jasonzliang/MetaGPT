
def specialFilter(nums):
    """Write a function that takes an array of numbers as input and returns 
    the number of elements in the array that are greater than 10 and both 
    first and last digits of a number are odd (1, 3, 5, 7, 9)."""
    
    def is_odd(n):
        """Check if a number's first or last digit is odd."""
        return int(str(n)[0]) in [1, 3, 5, 7, 9] and int(str(n)[-1]) in [1, 3, 5, 7, 9]
    
    # Filter the list based on conditions
    filtered_nums = [num for num in nums if num > 10 and is_odd(abs(num))]
    
    return len(filtered_nums)
