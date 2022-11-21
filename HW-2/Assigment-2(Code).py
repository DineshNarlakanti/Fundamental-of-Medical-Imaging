import math
import cv2
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

def MyPhantom(N1, N2, PosB1, PosB2, LengthB1, LengthB2, PosC1, LengthC1, LengthC2, WaterB, T1B, T2starB, WaterC, T1C, T2starC, tr_parameter, te_parameter):
    object_matrix = np.zeros(shape=[N1, N2, 3], dtype=np.uint8)
    object_T1_map = np.zeros(shape=[N1, N2, 3], dtype=np.uint8)
    object_T2_map = np.zeros(shape=[N1, N2, 3], dtype=np.uint8)
    output_image = np.zeros(shape=[N1, N2, 3], dtype=np.uint8)

    for i in range(1 ,N1):
        for j in range(1 ,N2):
            object_matrix[i, j] = 50
            object_T1_map[i, j] = 1
            object_T2_map[i, j] = 1
            if ((( j -(LengthB1 + ( 2* PosB2) + (0.5 * LengthC2)) ) /LengthC2 )**2 + ((i - PosC1) / LengthC1) ** 2 <= 1):
                object_matrix[i, j] = WaterC
                object_T1_map[i, j] = T1C
                object_T2_map[i, j] = T2starC
        if i >= PosB1 and i < PosB1 + LengthB1:
            for j in range(PosB2, PosB2 + LengthB2):
                object_matrix[i, j] = WaterB
                object_T1_map[i, j] = T1B
                object_T2_map[i, j] = T2starB

    for i in range(1, 100):
        for j in range(1, 100):
            output_image[i, j] = object_matrix[i, j] * (1 - np.exp(-tr_parameter / object_T1_map[i, j])) * np.exp(-te_parameter / object_T2_map[i,j])


    plt.imshow(object_T1_map)
    plt.title("T1 Map")
    plt.savefig("T1 Map")

    plt.imshow(object_T2_map)
    plt.title("T2 Map")
    plt.savefig("T2 Map")

    plt.imshow(output_image)
    plt.title("Final Image_" + str(te_parameter))
    plt.savefig("Final Image_" + str(te_parameter))

cntrast = []
for i in it.chain(range(0,50,10) , range(50,500,50) , range(500,5000,500) ):#, range(2000, 10000, 1000)
    MyPhantom(100, 100, 30, 20, 20, 20, 45, 12, 10, 220, 1500, 100, 255, 200, 150, 500, i)
    img_name = "Final Image_" + str(i) + ".png"
    img = cv2.imread(img_name)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = img_grey.std()
    cntrast.append(contrast)
    print(i, contrast)

print(cntrast)




