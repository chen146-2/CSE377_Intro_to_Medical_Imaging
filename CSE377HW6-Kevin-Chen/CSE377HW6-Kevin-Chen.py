#!/usr/bin/env python
# coding: utf-8

# ## CSE 377 - Homework 06
# 
# ### Kevin Chen - CHEN146 - 113448049

# ### Task 01
# 
# #### Load two input polyp images 'src_medical.jpg' and 'trg_medical.jpg' as f_src (x,y) and f_trg (x,y), respectively. Show the input images

# In[1]:


# Importing the required libraries to complete this assignment

import numpy as np
import matplotlib.pyplot as plt
import PIL as pil


# In[2]:


# loading the input images to their respective variable names

src_img = pil.Image.open("src_medical.jpg")
trg_img = pil.Image.open("trg_medical.jpg")

print('shape of the src img original:', src_img.size)
print('shape of the trg img original:', trg_img.size)
# the dimensions of the two images
# src -> (444, 522, 3)
# trg -> (555, 653, 3)
# Need to resize the images so there is no complications to the addition
# of arrays in question 5. I will scale down the trg_img since increasing
# the src image would decrease pixel density.

trg_img = trg_img.resize(src_img.size)

print('shape of the src img after:', src_img.size)
print('shape of the trg img after:', trg_img.size)

# As we can see from the print statements, both the src and trg images
# have the same size / dimensions, making it easier to deal with later.
f_src = np.array(src_img)
f_trg = np.array(trg_img)


# In[3]:


# showing the src_medical image to the user

fig, ax = plt.subplots(figsize=(6,6))
plt.title("The Source Domain Image", size=18)
plt.axis('off')
ax.imshow(f_src)
plt.show()


# In[4]:


# Showing the trg_medical image to the user

fig, ax = plt.subplots(figsize=(6,6))
plt.title("The Target Domain Image", size=18)
plt.axis('off')
ax.imshow(f_trg)
plt.show()


# ### Task 02
# 
# #### Calculate the Fourier transformation of these two images:
# #### F_src (u,v) = F (f_src (x,y) ) 
# #### F_trg (u,v) = F (f_trg (x,y) )
# 
# #### Calculate and show the spectrums of the two images in the frequency domain, i.e., |F_src (u,v)| and |F_trg (u,v)|.
# 
# #### Calculate and show the phases of the two images in the frequency domain, i.e., |Phi_src (u,v)| and |Phi_trg (u,v)|.
# 
# #### Note, the polyp images are color images with red, green, and blue channels.

# In[5]:


# defining a function to calculate the Fourier transformation of an image given as input

def fourier_transform(arr, name):
    
    # obtaining the rgb values of the input image as separate into individual arrays
    
    blue = arr[:,:,0]
    green = arr[:,:,1]
    red = arr[:,:,2]
    
    # converting each rgb array into a numpy array, converting to grayscale to get values
    # between [0,255]. this would ensure no negative values when displaying image.
    
    r = np.array(red)
    g = np.array(green)
    b = np.array(blue)
    
    # using the Fourier transformation function for 2d arrays from the NumPy library, fft.
    
    ft_r = np.fft.fft2(r)
    ft_g = np.fft.fft2(g)
    ft_b = np.fft.fft2(b)
    
    # shift the arrays to the center of the spectrum
    
    ft_r = np.fft.fftshift(ft_r)
    ft_g = np.fft.fftshift(ft_g)
    ft_b = np.fft.fftshift(ft_b)
    
    # Apply log and absolute value functions to get spectrums of the image in the frequency domain
    
    spect_r = np.log10(abs(ft_r))
    spect_g = np.log10(abs(ft_g))
    spect_b = np.log10(abs(ft_b))
    
    # In order to get the phases of the image, going to need to use numpy's angle()
    
    phase_r = np.angle(ft_r)
    phase_g = np.angle(ft_g)
    phase_b = np.angle(ft_b)
    
    #phase_r = np.
    #phase_r = np.
    
    # Plotting the graphs
    if (name != 'Source-adapt-to-target,'):
        fig, axs = plt.subplots(2, 3, figsize=(20,8))
    
        axs[0,0].imshow(spect_r, cmap='gray')
        axs[0,0].set_title((name + ' Spectrum of Red'))
        axs[0,0].axis('off')

        axs[0,1].imshow(spect_g, cmap='gray')
        axs[0,1].set_title((name + ' Spectrum of Green'))
        axs[0,1].axis('off')

        axs[0,2].imshow(spect_b, cmap='gray')
        axs[0,2].set_title((name + ' Spectrum of Blue'))
        axs[0,2].axis('off')
        
        axs[1,0].imshow(phase_r, cmap='gray')
        axs[1,0].set_title((name + ' Phase of Red'))
        axs[1,0].axis('off')

        axs[1,1].imshow(phase_g, cmap='gray')
        axs[1,1].set_title((name + ' Phase of Green'))
        axs[1,1].axis('off')

        axs[1,2].imshow(phase_b, cmap='gray')
        axs[1,2].set_title((name + ' Phase of Blue'))
        axs[1,2].axis('off')
    
        plt.show()
        
    else:
        plt.figure(figsize=(20,8))
        plt.subplot(1,3,1)
        plt.imshow(spect_r, cmap='gray')
        plt.title((name + 'Spectrum of Red'))
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(spect_g, cmap='gray')
        plt.title((name + 'Spectrum of Green'))
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(spect_b, cmap='gray')
        plt.title((name + 'Spectrum of Blue'))
        plt.axis('off')
        plt.show()
        
    #rgb_arr = np.dstack((spect_r, spect_g, spect_b))
    return ft_r, ft_g, ft_b


# In[6]:


F_src_r, F_src_g, F_src_b = fourier_transform(f_src, 'Source')


# In[7]:


F_trg_r, F_trg_g, F_trg_b = fourier_transform(f_trg, 'Target')


# ### Task 03
# 
# #### Create and show a Gaussian Low-pass filter H_Lp (u,v) with standard derivation D_0.

# In[8]:


trg_r = f_trg.shape[0]
trg_c = f_trg.shape[1]
H_Lp = np.zeros((trg_r, trg_c))
center_x, center_y = trg_r // 2, trg_c // 2
D_0 = 2
for i in range(trg_r):
    for j in range(trg_c):
        dist = np.sqrt((i - center_x)**2 + (j-center_y)**2)
        H_Lp[i,j] = np.exp(-dist**2 / (2*D_0**2))
        #H_Lp[i,j] = np.exp(-(i**2+j**2)/(2*D_0**2))
plt.imshow(H_Lp, cmap='gray')
plt.title('Gaussian Low-Pass Filter')
plt.axis('off')
plt.show()


# ### Task 04
# 
# #### Create and show a Gaussian High-pass filter H_Hp (u,v) with standard deviation D_0.

# In[9]:


H_Hp = np.zeros((trg_r, trg_c))
H_Hp = 1 - H_Lp
            
plt.imshow(H_Hp, cmap='gray')
plt.title('Gaussian Low-Pass Filter')
plt.axis('off')
plt.show()


# ### Task 05 
# 
# #### Apply the high-pass filter to the source image and the low-pass filter to target image to blend the two images' spectrums:
# 
# #### |G(u,v)| = |F_src (u,v)| H_Hp (u,v) + |F_trg (u,v)| H_Lp (u,v)
# 
# #### Show |G(u,v)|.

# In[10]:


#src_shift = F_src * H_Hp
#H_Hp_src = np.zeros_like(F_src)
G_r = (np.log(abs(F_src_r)) * H_Hp) + (np.log(abs(F_trg_r)) * H_Lp)
G_g = (np.log(abs(F_src_g)) * H_Hp) + (np.log(abs(F_trg_g)) * H_Lp)
G_b = (np.log(abs(F_src_b)) * H_Hp) + (np.log(abs(F_trg_b)) * H_Lp)
plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.imshow(G_r, cmap='gray')
plt.title('Source-adapt-to-target, Spectrum of Red')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(G_g, cmap='gray')
plt.title('Source-adapt-to-target, Spectrum of Green')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(G_b, cmap='gray')
plt.title('Source-adapt-to-target, Spectrum of Blue')
plt.axis('off')
plt.show()


# ### Task 06
# 
# #### Convert the domain adapted source image to the spatial domain and visualize it.
# 
# #### f_src->trg = F^-1 ( |G(u,v)| e^(phi_src (u,v)) )

# In[11]:


spect = np.dstack((G_r, G_g, G_b))
phase_r = np.angle(F_src_r)
phase_g = np.angle(F_src_g)
phase_b = np.angle(F_src_b)
inv_r = np.fft.ifft2(np.fft.ifftshift(np.exp(G_r) * np.exp(1j * np.angle(phase_r))))
inv_g = np.fft.ifft2(np.fft.ifftshift(np.exp(G_g) * np.exp(1j * np.angle(phase_g))))
inv_b = np.fft.ifft2(np.fft.ifftshift(np.exp(G_b) * np.exp(1j * np.angle(phase_b))))
img = pil.Image.merge('RGB', (pil.Image.fromarray(inv_r.real.astype('uint8')), pil.Image.fromarray(inv_g.real.astype('uint8')), pil.Image.fromarray(inv_b.real.astype('uint8'))))
plt.imshow(img)
plt.show()

# unable to get the domain adapted source image to the spatial domain ://


# In[ ]:




