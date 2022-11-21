import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def BigSquare(N1, N2, PosB1, PosB2, LengthB1, LengthB2, PosC1, LengthC1, LengthC2, SIA, SIB, SIC):
    whiteBlankImage = np.zeros(shape=[N1, N2, 3], dtype=np.uint8)
    for i in range(1,N1):
        for j in range(1,N2):
            whiteBlankImage[i,j] = SIA
            if (((j-(LengthB1 + (2* PosB2) + (0.5 * LengthC2)))/LengthC2)**2 + ((i-PosC1)/LengthC1)**2 <= 1):
                whiteBlankImage[i, j] = SIC
        if i >= PosB1 and i < PosB1+LengthB1:
            for j in range(PosB2,PosB2+LengthB2):
                whiteBlankImage[i,j] = SIB

    plt.imshow(whiteBlankImage)
    plt.show()
    
BigSquare(100,100,30,20,20,20,45,12,10,255,25,125)