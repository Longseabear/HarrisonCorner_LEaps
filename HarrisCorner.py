import matplotlib.pyplot as plt
import scipy.misc as sc
from scipy.signal import convolve2d
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

kernel_y = [[-1,-1,-1],[0,0,0],[1,1,1]]
kernel_x = np.transpose(kernel_y)
kernel_gaussian = np.array([ [1, 4, 6, 4, 1 ],[ 4, 16, 24, 16, 4 ],
    [ 6, 24, 36, 24, 6 ],[ 4, 16, 24, 16, 4 ],[ 1, 4, 6, 4, 1 ] ])/255

dx = [1,1,1,0,0,-1,-1,-1]
dy = [1,0,-1,1,-1,1,0,-1]
def check_maximum(corner):
    for i in range(8):
        next_x = dx[i] + 1
        next_y = dy[i] + 1
        if corner[1,1] < corner[next_y,next_x]:
            return False
    return True

def harris_corners(img, threshold=0.02):
    img = rgb2gray(img)

    tx = convolve2d(img, kernel_x, 'same')
    ty = convolve2d(img, kernel_y, 'same')

    dx2 = tx*tx
    dy2 = ty*ty
    dxy = tx*ty

    gdx2 = convolve2d(dx2, kernel_gaussian, 'same')
    gdy2 = convolve2d(dy2, kernel_gaussian, 'same')
    gdxy = convolve2d(dxy, kernel_gaussian, 'same')

    corners_craft = (gdx2*gdy2-gdxy*gdxy)-0.04*(gdx2+gdy2)*(gdx2+gdy2)
    corners = []

    [h,w] = np.shape(img)
    #non maximum-suppression
    for i in range(2,h-2):
        for j in range(2,w-2):
            if corners_craft[i,j] > threshold:
                if check_maximum(corners_craft[i-1:i+2, j-1:j+2]):
                    corners.append(np.array([i,j]))
    return corners

img = plt.imread("C:\\Users\\LEaps\\Pictures\\lenna.png")
corner = harris_corners(img,0.0001)
corner_y = [val[0] for val in corner]
corner_x = [val[1] for val in corner]

plt.figure()
plt.imshow(img)

plt.scatter(corner_x, corner_y, s=10, c='blue', marker='x')
plt.show()