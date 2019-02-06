import numpy as np
from numpy.linalg import norm
import cv2


def get_mask(img, lh, ls, lv, hh, hs, hv):
    mask = cv2.inRange(img, np.array([lh, ls, lv]), np.array([hh, hs, hv]))
    return mask

def prosao_broj(linija, broj):
    x, y, w, h = broj.granice
    centar = (int((x + x + w) / 2.0), int((y + y + h) / 2.0))

    t1, t2 = linija
    flag = False
    if t1[0] < centar[0] < t2[0] and t2[1] - 10 < centar[1] < t1[1] + 10:
        distance = norm(np.cross(np.array(t2) - np.array(t1), np.array(t1) - np.array(centar))) / norm(np.array(t2) - np.array(t1))

        if distance < 11:
            flag = True

    return flag


def pronadji_liniju(img):
    mask = get_mask(img, 90, 0, 0, 255, 70, 70)
    img = cv2.bitwise_and(img, img, mask=mask)

    low_threshold = 50
    high_threshold = 150

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments

    t1_x, t1_y, t2_x, t2_y = 5000, -5000, -5000, 5000

    linije = cv2.HoughLinesP(edges, rho, theta, threshold, min_line_length, max_line_gap)


    arrayX1 = []
    arrayX2 = []
    arrayY1 = []
    arrayY2 = []

    for line in linije:
        arrayX1.append(line[0, 0])
        arrayY1.append(line[0, 1])
        arrayX2.append(line[0, 2])
        arrayY2.append(line[0, 3])

    t1_x = min(arrayX1)
    t1_y = max(arrayY1)
    t2_x = max(arrayX2)
    t2_y = min(arrayY2)

    return (t1_x, t1_y), (t2_x, t2_y)
