#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import PIL as pil
import matplotlib.pyplot as plt


# In[3]:


# converting the Mammogram image into a numpy array
input_img = pil.Image.open("./Mammogram.png")
input_arr = np.array(input_img)

# plugging into the equation s = (L-1) - r
# L = 256 since Mammogram.png is an 8-bit grayscale image
# r = input_image is a numpy array where each value corresponds to the pixel intensity
# of the input image, Mammogram.png, where I applied the formula to each pixel --> s
negative_img = (256 - 1) - input_arr

# displaying the negative image with grayscale applied
# negative image itself is neon in color, so added gray cmap to get image to be darker

plt.title("Negative Image")
plt.imshow(negative_img, cmap="gray")
plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)
plt.show()


# In[4]:


# applying the log transformation onto the negative image
# using the formula s = log(1+r)
log_img = np.log(1+negative_img)

# displaying the log transformed image of the negative image

plt.title("Log Transformed Image")
plt.imshow(log_img, cmap="gray")
plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)
plt.show()


# In[5]:


# applying power law transformation onto the negative image
# using the formula s=r^gamma, for example, gamma=0.1

power_img = np.power(negative_img, 0.1)

#displaying the power-law transformed image of the negative image

plt.title("Power-law Transformed Image, Gamma=0.1")
plt.imshow(power_img, cmap="gray")
plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)
plt.show()


# In[6]:


# applying piecewise linear transformation for the contrast stretch on the negative image
# where r_min is the minimum intensity value in the input image and
# r_max is the maximum intensity value in the input image

r_min = np.min(input_arr)
r_max = np.max(input_arr)
slope = (256-1)/(r_max-r_min)
piecewise_img = negative_img * slope

# displaying the piecewise transformed image to the user

plt.title("Piecewise Linear Transformed Image")
plt.imshow(piecewise_img, cmap="gray")
plt.tick_params(left=False,right=False,labelleft=False,labelbottom=False,bottom=False)
plt.show()


# In[ ]:




