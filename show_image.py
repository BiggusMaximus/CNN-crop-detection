import sys  # to access the system
import cv2
img = cv2.imread("NLB.JPG", cv2.IMREAD_ANYCOLOR)

if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Image not found or has invalid dimensions')
