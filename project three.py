# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:11:20 2019

@author: Mohammadmahdi
"""

import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as la
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

m = np.zeros((80,120,12),dtype="float64")

m[:,:,0] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb1.bmp",0)
m[:,:,1] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb2.bmp",0)
m[:,:,2] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb3.bmp",0)
m[:,:,3] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb4.bmp",0)
m[:,:,4] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb5.bmp",0)
m[:,:,5] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb6.bmp",0)
m[:,:,6] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb7.bmp",0)
m[:,:,7] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb8.bmp",0)
m[:,:,8] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb9.bmp",0)
m[:,:,9] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb10.bmp",0)
m[:,:,10] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb11.bmp",0)
m[:,:,11] = cv.imread("C:/Users/Mohammadmahdi/Desktop/test work/Remote Sensing computer practice/project three/data/testb12.bmp",0)


# n = reshape(m,[1,9600,12]);

n = np.reshape(m,(1,9600,12))


plt.subplot(341),plt.hist(n[:,:,0],256,[0,256]),plt.title('histogram of 1th image')
plt.subplot(342),plt.hist(n[:,:,1],256,[0,256]),plt.title('histogram of 2th image')
plt.subplot(343),plt.hist(n[:,:,2],256,[0,256]),plt.title('histogram of 3th image')
plt.subplot(344),plt.hist(n[:,:,3],256,[0,256]),plt.title('histogram of 4th image')
plt.subplot(345),plt.hist(n[:,:,4],256,[0,256]),plt.title('histogram of 5th image')
plt.subplot(346),plt.hist(n[:,:,5],256,[0,256]),plt.title('histogram of 6th image')
plt.subplot(347),plt.hist(n[:,:,6],256,[0,256]),plt.title('histogram of 7th image')
plt.subplot(348),plt.hist(n[:,:,7],256,[0,256]),plt.title('histogram of 8th image')
plt.subplot(349),plt.hist(n[:,:,8],256,[0,256]),plt.title('histogram of 9th image')
plt.subplot(3410),plt.hist(n[:,:9],256,[0,256]),plt.title('histogram of 10th image')
plt.subplot(3411),plt.hist(n[:,:,10],256,[0,256]),plt.title('histogram of 11th image')
plt.subplot(3412),plt.hist(n[:,:,11],256,[0,256]),plt.title('histogram of 12th image')

N = np.zeros((12,9600),dtype="float64")

for i in range(12):
    N[i,:]=n[:,:,i]
   
M = np.mean(N,axis=1)

N = (N.transpose() - M).transpose()

C = np.cov(N)

A = np.zeros((12,3),dtype="float64")
w,v = la.eig(C)

A[:,0] = v[:,0]
A[:,1] = v[:,1]
A[:,2] = v[:,2]

A = np.transpose(A)

y = np.matmul(A,N)

n = np.zeros((80,120,3))

for i in range(3):
    n[:,:,i] = np.reshape(y[i,:],(80,120))

n = n.astype("uint8")
# rgb_img = cv2.merge([r,g,b])
r = n[:,:,0]
g = n[:,:,1]
b = n[:,:,2]

rgb_img = cv.merge([r,g,b])
rgb_img1 = rgb_img[:,:,::-1]
plt.figure()
plt.imshow(rgb_img1),plt.title('PC1 PC2 PC3'),plt.xticks([]),plt.yticks([]),plt.colorbar()


hsv_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2YCR_CB)
plt.figure()
plt.imshow(hsv_img),plt.title('rgb2hsv'),plt.xticks([]),plt.yticks([])





# bands = 0,3,10 and PAN 6



B1 = m[:,:,0]
B2 = m[:,:,3]
B3 = m[:,:,10]
PAN = m[:,:,6]

BB = B1 + B2 + B3
R = ((B3)/(BB))*PAN
G = ((B2)/(BB))*PAN
B = ((B1)/(BB))*PAN


brovey_img = cv.merge([R,G,B]).astype("uint8")

plt.imshow(brovey_img),plt.title('transformed image using brovey ')
plt.xticks([]),plt.yticks([])




gamma = 0.3
rr = np.reshape(r,(9600,))
gg = np.reshape(g,(9600,))
bb = np.reshape(b,(9600,))


fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0, 0].set_title('2D histogram of PC1 and PC2')
axes[0, 0].hist2d(rr, gg, bins=200,norm=mcolors.PowerNorm(gamma))
axes[0,1].set_title('2D histogram of PC1 and PC3')
axes[0,1].hist2d(rr,bb,bins=200,norm=mcolors.PowerNorm(gamma))
axes[1,0].set_title('2D histogram of PC2 and PC3')
axes[1,0].hist2d(gg,bb,bins=200,norm=mcolors.PowerNorm(gamma))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#x, y = np.random.rand(2, 100) * 4
x = rr
y = gg

hist, xedges, yedges = np.histogram2d(x, y, bins=255, range=[[0, 256], [0, 256]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()


