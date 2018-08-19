# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:10:28 2018

@author: 3D Printer
"""
#%%
plt.subplot(1,4,4)
plt.imshow(sat_img_list[2],vmin=0,vmax=1,cmap=cm.RdYlBu), plt.colorbar()


plt.subplot(1,4,3)
plt.imshow(sat_img_list[1],vmin=0,vmax=1,cmap=cm.RdYlBu), plt.colorbar()

plt.subplot(1,4,2)
plt.imshow(sat_img_list[0],vmin=0,vmax=1,cmap=cm.RdYlBu), plt.colorbar()

plt.subplot(1,4,1)
plt.imshow(ratio_list[0],cmap=cm.RdYlBu)

#%%
import matplotlib as mpl
import cv2
import matplotlib.cm as cm

red = find_band(lamb,640)
green = find_band(lamb,550)
blue = find_band(lamb,460)

rgb_img_list = []
for i in range(len(sat_img_list)):
    spectral.save_rgb(files[i][:-5]+'.png', img_list[i], [red, green, blue]) # This line must be commented if the RGB images are already saved
    rgb_img_list.append(cv2.imread(files[i][:-5]+'.png'))

fig = plt.figure()

#Plot a representative RGB Image
plt.subplot(2,len(sat_img_list),2)
plt.imshow(rgb_img_list[i])
plt.xticks([])
plt.yticks([])
plt.axis('off')

cmap=cm.RdYlBu
cmap.set_under(color='black',alpha=0)
timepoints = ['Baseline','Hypoxia (1m)','Resucitation (4m)']
for i in range(len(sat_img_list)):
    #rgb_img = np.zeros([img_list[i].shape[0],img_list[i].shape[1],3])
    # rgb_img[:,:,0] = img_list[i][:,:,red].reshape([img_list[i].shape[0],img_list[i].shape[1]])*255
    # rgb_img[:,:,1] = img_list[i][:,:,green].reshape([img_list[i].shape[0],img_list[i].shape[1]])*255
    # rgb_img[:,:,2] = img_list[i][:,:,blue].reshape([img_list[i].shape[0],img_list[i].shape[1]])*255
    rgb_img = rgb_img_list[i][17:-17,17:-17]
    rgb_img = rgb_img.astype('uint8')
    plt.subplot(2,len(sat_img_list),i+1+len(sat_img_list))
    norm = mpl.colors.Normalize(vmin=0,vmax=1)
    plt.imshow(rgb_img)
    sat_img = sat_img_list[i]
    #sat_img_blur = cv2.blur(sat_img,(5,5))
    plt.imshow(sat_img,vmin=0.01,vmax=1,cmap=cmap)
    plt.title(timepoints[i])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
cax = fig.add_axes([0.35, .1, 0.35, 0.05])
cbar = plt.colorbar(orientation='horizontal',cax=cax,label='Saturation')

for i in range(len(img_list)):
    plt.subplot(2,len(img_list),i+1+len(img_list))
    plt.imshow(1/ratio_list[i],cmap=cm.RdYlBu)
#%%
img_spec = t_list[0]/v_list[0]
img_spec = img_spec[find_band(lamb,450):find_band(lamb,600)]
lamb_a = lamb[find_band(lamb,450):find_band(lamb,600)]
fit_sat = fitter(spec_fit_1, lamb_a, img_spec,maxiter=200, acc=0.0001)

#%%
from sklearn.metrics import r2_score
x = img_spec
y = fit_sat(lamb_a)
r2_score(x,y)

#%%
red = find_band(lamb,620)
green = find_band(lamb,495)
blue = find_band(lamb,450)
spectral.imshow(img_list[i],(red,green,blue),stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))
spectral.imshow(img2_c,(red,green,blue),stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))
spectral.imshow(img3_c,(red,green,blue),stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))