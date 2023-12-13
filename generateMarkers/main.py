import cv2 as cv
from cv2 import aruco

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

MARKER_SIZE = 400 # pixels

for id in range(20):

    marker_image = aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
    cv.imshow("img", marker_image)
    cv.imwrite(f"markers/marker_{id}.png",marker_image)
    # cv.waitKey(0)
    # break 