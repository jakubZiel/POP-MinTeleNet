import math


def edgeCapacity(modulity : int, o : int) -> int:
    if modulity <= 0:
        raise ValueError("modulity must be positive")

    return math.ceil(o / modulity)