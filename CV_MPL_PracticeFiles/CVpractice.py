#Code example via Rob Mulla YouTube channel "Image Processing with OpenCV and Python"
import pandas as pd
import numpy as np

from glob import glob

import cv2
import matplotlib.pylab as plt

#Reading in images
#Get a list of images with .jpg file type
image_paths = glob(r"C:\Users\sarah\VSCode Projects\AudreyII\*.jpg")

#Read the images and print their sizes
img_mpl = plt.imread(image_paths[0])
img_cv2 = cv2.imread(image_paths[0])
#print(img_mpl.shape), print(img_cv2.shape)

#Flatten array and plot pixel intensities into a histogram 
pd.Series(img_mpl.flatten()).plot(kind='hist',bins = 50, title= "Distribution of Pixel Values")
#plt.show()

#Display images using matplotlib
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img_mpl)
ax.axis('off')
#plt.show()


#Display RGB Channels of our image
fig, axs = plt.subplots(1,3, figsize=(15,5))
axs[0].imshow(img_mpl[:,:,0], cmap='Reds')
axs[1].imshow(img_mpl[:,:,1], cmap='Greens')
axs[2].imshow(img_mpl[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red Channel')
axs[1].set_title('Green Channel')
axs[2].set_title('Blue Channel')
#plt.show()

#CV2 READS CHANNELS AS BGR
#Reading images with cv2
fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV Image')
axs[1].set_title('MatPlot Image')
#plt.show()

plt.close('all')
#Convert from BGR to RGB
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig,ax = plt.subplots()
ax.imshow(img_cv2_rgb)
ax.axis('off')
#plt.show()


#Image manipulation
img = plt.imread(image_paths[0])
fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(img)
ax.axis('off')
#plt.show()

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray.shape #now we only have 2 channels
fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(img_gray, cmap="Greys")
ax.axis('off')
ax.set_title('Grey Image')
#plt.show()

#Resizing and Scaling
img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(img_resized)
ax.axis('off')
ax.set_title('Resized Image')
#plt.show()

img_resized = cv2.resize(img, (100,200))
fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(img_resized)
ax.axis('off')
ax.set_title('Resized Image')
#plt.show()

#Adding interpolation
img_resize = cv2.resize(img, (5000,5000), interpolation=cv2.INTER_CUBIC)
fig, ax = plt.subplots(figsize = (8,8))
ax.imshow(img_resize)
ax.axis('off')
ax.set_title('Resized & Interpolated Image')
#plt.show()

plt.close('all')
#Using Kernels (look up oneline)
kernel_sharpening = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
img_sharp = cv2.filter2D(img,-1, kernel_sharpening )
fig, axs = plt.subplots(2,1,figsize = (10,10))
axs[0].imshow(img_sharp)
axs[0].axis('off')
axs[0].set_title('Sharpened Image')

kernel_blurring = np.ones((3,3), np.float32)/9 #make blurrier by increasing the divisor
img_blurred = cv2.filter2D(img, -1, kernel_blurring)
axs[1].imshow(img_blurred)
axs[1].axis('off')
axs[1].set_title('Blurred Image')

plt.show()

#Saving the image
plt.imsave('mpl_ethan.png', img_blurred)

img_bgr = cv2.imread('cv2_ethan.png')

# Convert the image from BGR to RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Save the image in RGB format
cv2.imwrite('rgb_image.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))