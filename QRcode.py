import cv2
import cv2.aruco as aruco
import os


# Load dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Generate markers
for i in range(4):
    marker = aruco.generateImageMarker(aruco_dict, i, 200)

    filename = f"marker_{i}.png"
    cv2.imwrite(filename, marker)

    print(f"Saved {filename}")
    print("Saving to:", os.getcwd())