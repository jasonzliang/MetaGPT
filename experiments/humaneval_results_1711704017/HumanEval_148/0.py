
def bf(planet1, planet2):
    """
    This function takes two planet names as strings and returns a tuple containing all planets whose orbits are 
    located between the orbit of planet1 and the orbit of planet2, sorted by the proximity to the sun.
    If planet1 or planet2 are not correct planet names, it will return an empty tuple.
    """
    solar_system = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    # Check if both planets are in the solar system
    if planet1 not in solar_system or planet2 not in solar_system:
        return ()
    
    # Get the indices of the two planets in the list
    index1 = solar_system.index(planet1)
    index2 = solar_system.index(planet2)
    
    # If the first planet is to the left of the second, return the planets between them
    if index1 < index2:
        return tuple(solar_system[index1+1 : index2])
    else:  # If the first planet is to the right of the second, return the planets between them in reverse order
        return tuple(solar_system[index2+1 : index1][::-1])
```

# Example usage:
print(bf("Jupiter", "Neptune"))  # Output: ('Saturn', 'Uranus')
print(bf("Earth", "Mercury"))  # Output: ('Venus',)
print(bf("Mercury", "Uranus"))  # Output: ('Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn')
