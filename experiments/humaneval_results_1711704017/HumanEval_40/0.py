
def triples_sum_to_zero(lst):
    """
    This function takes a list of integers as an input. 
    It returns True if there are three distinct elements in the list that sum to zero, and False otherwise.
    """
    
    # Check for invalid inputs
    if not isinstance(lst, list) or not all(isinstance(i, int) for i in lst):
        raise ValueError("Input must be a list of integers")
        
    # Sort the list to make it easier to find triples that sum to zero
    lst.sort()
    
    # Iterate through each element in the list (except the last two)
    for i in range(len(lst)-2):
        # Initialize two pointers, one at the start and one at the end of the remaining elements
        left = i + 1
        right = len(lst) - 1
        
        while left < right:
            current_sum = lst[i] + lst[left] + lst[right]
            
            # If the sum is zero, we've found a triple that sums to zero
            if current_sum == 0:
                return True
                
            # If the sum is less than zero, move the left pointer to the right
            elif current_sum < 0:
                left += 1
            
            # If the sum is more than zero, move the right pointer to the left
            else:
                right -= 1
    
    # If we've gone through all possible triples and found no triple that sums to zero, return False
    return False
