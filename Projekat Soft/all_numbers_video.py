import vector as v
import numpy as np


def get_distances(brojevi, centar):
    distances = []
    for __broj in brojevi:
        x, y, w, h = __broj.granice
        __centar = (int((x + x + w) / 2.0), int((y + y + h) / 2.0))
        distances.append(v.distance(centar, __centar))
    return distances


def check_distances(distances):
    if distances.__len__() != 0 and distances[np.argmin(distances)] < 20:
        return True
    else:
        return False


class All_numbers:

    def __init__(self):
        self.__brojevi = []

    def update(self, brojevi):
        for broj in brojevi:
            x, y, w, h = broj.granice
            centar = (int((x + x + w) / 2.0), int((y + y + h) / 2.0))
            distances = get_distances(self.__brojevi, centar)
            if not check_distances(distances):
                self.__brojevi.append(broj)
            else:
                i = np.argmin(distances)
                self.__brojevi[i].granice = broj.granice


        return self.__brojevi
    