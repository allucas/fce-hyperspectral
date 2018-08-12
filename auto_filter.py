# Load the libraries of interest
# This code performs an average filtering in all the files in a given folder

import os
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle

#%%
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_hyper_image(filename):
    hdr_name = filename+'.hdr'
    lib = spectral.envi.read_envi_header(hdr_name)
    img = envi.open(hdr_name, filename)
    return lib, img

def mean_spectrum(roi):
    mean_spec = np.mean(roi,axis=1)
    mean_spec = np.mean(mean_spec,axis=0)
    return mean_spec

def get_range(x,y,xlims):
    y = y[x>=xlims[0]]
    x = x[x>=xlims[0]]
    y = y[x<=xlims[1]]
    x = x[x<=xlims[1]]
    return x, y

def get_norm_spec(filename,roi1,roi2):
    lib, img = load_hyper_image(filename)
    roi = img[roi1[0]:roi1[1],roi1[2]:roi1[3],:]
    norm = img[roi2[0]:roi2[1],roi2[2]:roi2[3],:]
    lamb = np.array(lib['wavelength']).astype(float)
    norm_spec = mean_spectrum(roi)/mean_spectrum(norm)
    return lamb,norm_spec

def find_refl(refl,lamb,val):
    diff = np.abs(lamb-val)
    return refl[np.argmin(diff)]

def find_band(lamb,wav):
    diff = np.abs(lamb-wav)
    return np.argmin(diff)

def avg_filter(img,lamb,win):
    def get_kernel(idxx,idxy,win):
        idx_range = np.arange(win)
        shiftx = int(idxx-int(win/2))
        shifty = int(idxy-int(win/2))
        idxx_new = idx_range + shiftx
        idxy_new = idx_range + shifty
        return idxx_new, idxy_new
    mean_kernel = np.zeros([win*win,len(lamb)])
    img2 = np.zeros(img.shape)
    for i in range(img.shape[0]-int(win/2)*2):
        for j in range(img.shape[1]-int(win/2)*2):
            idx1 = i + int(win/2)
            idx2 = j + int(win/2)
            rx, ry = get_kernel(idx1,idx2, win)
            count = 0
            for ii in range(len(rx)):
                for jj in range(len(ry)):
                    mean_kernel[count,:] = img[rx[ii],ry[jj],:]
                    count = count + 1
            img2[idx1,idx2,:] = np.mean(mean_kernel,axis=0)
        print(i)
        #if i == 2600: #Change this if the filtering must stop at a certain
        #    break
    return img2

folder_path = ''#CHANGE THIS TO THE FOLDER PATH
folder = ''# CHANGE THIS TO THE NAME OF THE FOLDER

# Go to the folder and get the names of all the .bil files in the folder
os.chdir(folder_path+'//'+folder)
dir_list = os.listdir()
dir_list2 = []

for i in range(len(dir_list)):
    if dir_list[i][-3:]=='bil':
        dir_list2.append(dir_list[i])

files = dir_list2

# Perform an average filtering in all the .bil files in the folder
# save the filtered image into a python pickle object

for i in range(0,len(files)):
    os.chdir(folder_path+'//'+folder)
    lib, img = load_hyper_image(files[i])
    lamb = np.array(lib['wavelength']).astype(float)
    img_all = img.load()
    img2 = avg_filter(img_all,lamb,7)
    fname = files[i][:-4]
    os.chdir(folder_path+'//'+folder+'//'+'filtered')
    spectral.envi.save_image(files[i]+'.hdr',img2,ext='')
    save_obj(lamb,'lamb')
    del img
    del img2
    del img_all