
def planets_between(planet1, planet2):
    """Returns a tuple of all planets whose orbits are located between the orbit 
    of planet1 and the orbit of planet2, sorted by the proximity to the sun.

    Args:
        planet1: The name of the first planet as a string.
        planet2: The name of the second planet as a string.

    Returns:
        A tuple containing all planets whose orbits are located between the orbit 
        of planet1 and the orbit of planet2, sorted by the proximity to the sun.
        
    Raises:
        ValueError: If planet1 or planet2 are not correct planet names.
    """
    
    solar_system = ["Mercury", "Venus", "Earth", "Mars", 
                   "Jupiter", "Saturn", "Uranus", "Neptune"]
    
    if planet1 not in solar_system or planet2 not in solar_system:
        raise ValueError("Invalid planet name(s).")
        
    index1 = solar_system.index(planet1)
    index2 = solar_system.index(planet2)
    
    if index1 < index2:
        return tuple(solar_system[index1+1 : index2])
    else:
        return tuple(solar_system[index2+1 : index1])
