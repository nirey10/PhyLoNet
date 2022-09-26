import cv2

img1 = cv2.imread('081090.jpg')
img2 = cv2.imread('081094.jpg')
size = 64
resized1 = cv2.resize(img1, (size, size))
resized2 = cv2.resize(img2, (size, size))

cv2.imwrite("img1"+str(size)+".jpg", resized1)
cv2.imwrite("img2"+str(size)+".jpg", resized2)