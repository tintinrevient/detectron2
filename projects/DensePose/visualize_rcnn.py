import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np

im  = cv2.imread('./pix/demo_im.jpg')
im_IUV = cv2.imread('./pix/demo_im_IUV.png')
im_INDS = cv2.imread('./pix/demo_im_INDS.png',  0)

fig = plt.figure(figsize=[15,15])
plt.imshow(np.hstack((im_IUV[:,:,0]/24., im_IUV[:,:,1]/256., im_IUV[:,:,2]/256.)))
plt.title('I, U and V images.'); plt.axis('off'); plt.show()

fig = plt.figure(figsize=[12,12])
plt.imshow(im[:,:,::-1])
plt.contour(im_IUV[:,:,1]/256., 10, linewidths = 1)
plt.contour(im_IUV[:,:,2]/256., 10, linewidths = 1)
plt.axis('off'); plt.show()

fig = plt.figure(figsize=[12,12])
plt.imshow(im[:,:,::-1])
plt.contour(im_INDS, linewidths = 4)
plt.axis('off'); plt.show()