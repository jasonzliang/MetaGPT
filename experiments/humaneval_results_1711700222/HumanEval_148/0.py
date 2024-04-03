
def bf(planet1, planet2):
    planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
    if planet1 in planets and planet2 in planets:
        index1 = planets.index(planet1)
        index2 = planets.index(planet2)
        if index1 < index2:
            return tuple(planets[index1+1 : index2])
        else:
            return tuple(planets[index2+1 : index1])
    else:
        return ()
