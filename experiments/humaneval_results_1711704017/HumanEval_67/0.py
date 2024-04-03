
def fruit_distribution(s,n):
    """
    This function takes a string s that represents the number of apples and oranges in the basket 
    and an integer n that represents the total number of fruits in the basket. It returns the 
    number of mango fruits in the basket.
    """
    
    # Split the string into words to get the numbers of apples and oranges
    fruit_numbers = s.split()
    apple_count = int(fruit_numbers[0])
    orange_count = int(fruit_numbers[2])
    
    # Calculate the number of mango fruits by subtracting the total count of apples and oranges from the total fruit count
    mango_count = n - apple_count - orange_count
    
    return mango_count
