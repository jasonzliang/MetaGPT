
def fruit_distribution(fruit_string, total_fruits):
    """
    This function calculates the number of mango fruits in a basket given 
    the string that represents the total number of apples and oranges 
    and an integer that represent the total number of the fruits in the basket.

    Args:
        fruit_string: A string representing the number of apples and oranges in the basket.
        total_fruits: An integer representing the total number of fruits in the basket.

    Returns:
        The number of mango fruits in the basket.

    Raises:
        ValueError: If the fruit_string does not contain a valid representation 
                    of the number of apples and oranges.
    
    """
    # Split the string into words to get the numbers of apples and oranges
    try:
        fruits = fruit_string.split()
        num_apples, num_oranges = int(fruits[0]), int(fruits[2])
    except ValueError as e:
        raise ValueError("Invalid input string") from e
    
    # Calculate the number of mango fruits by subtracting the total number 
    # of apples and oranges from the total number of fruits in the basket.
    num_mango = total_fruits - num_apples - num_oranges
    
    return num_mango
