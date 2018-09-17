# Initialize Funcitons
import os
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
import cv2
#%%
def save_obj(obj, name ):
    input('ARE YOU SURE YOU WANT TO SAVE THIS OBJECT...')
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
    img2 = img
    for i in range(img.shape[0]-int(win/2)*2):
        for j in range(img.shape[1]-int(win/2)*2):
            idx1 = i + int(win/2)
            idx2 = j + int(win/2)
            rx, ry = get_kernel(idx1,idx2, win)
            count = 0
            for ii in range(len(rx)):
                for jj in range(len(ry)):
                    mean_kernel[count,:] = img_all[rx[ii],ry[jj],:]
                    count = count + 1
            img2[idx1,idx2,:] = np.mean(mean_kernel,axis=0)
        print(i)
        if i == 2000:
            break
    return img2

#%% Load the images from a given folder

#img_folder = 'F:\\Research Data\\hyperspectral\\hypoxia\\H005\\trial1\\filtered'
img_folder = 'F:\\Research Data\\hyperspectral\\dilution\\H006'
files = ['E3.bil']
#files = ['BL_c.bil','H30_c.bil','H2_c.bil','H4_c.bil','R30_c.bil','R2_c.bil','R4_c.bil']
sat_fname = 'sat_list2_2'
#files = ['cube0.bil','cube1.bil','cube2.bil','cube3.bil','cube4.bil','cube5.bil','cube6.bil','cube7.bil','cube8.bil','cube9.bil','cube10.bil','cube11.bil']

sat_img_list = []

ratio_list = []
for k in range(len(files)):
    img_list = []
    os.chdir(img_folder)
    lamb,img = load_hyper_image(files[k])
    #img_all = img.load()
    img_all = img
    img_list.append(img_all)
    lamb = load_obj('lamb')
    
    # Calculate the wavelength ratio for segmentation
    
    import matplotlib.cm as cm
    num = 576
    denom = 486
    for i in range(len(img_list)):
        #plt.subplot(1,len(img_list),1+i)
        img = img_list[i][10:-10,10:-10,:]
        ratio1 = img[:,:,find_band(lamb,num)]/img[:,:,find_band(lamb,denom)]
        #ratio2 = cv2.medianBlur(ratio2,15)
        ratio1 = ratio1.reshape([ratio1.shape[0],ratio1.shape[1]])
        # plt.imshow(1-ratio1[:,:], cmap=cm.hot, vmin=0, vmax=1)
        # cbar = plt.colorbar()
        # cbar.set_label('Wavelength Ratio', rotation=270)
        ratio_list.append(ratio1)
#%%    
    # Mask the image to get the vessel background for analysis
    
    def get_avg_bkg(img, mask):
    # mask corresponds to the vessel mask. (i.e. vessels=1, tissue=0)
        img1_t = np.zeros(img.shape)
        img1_v = np.zeros(img.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if ~(mask[i,j]==True):
                    img1_t[i,j,:] = img[i,j,:]
                else:
                    img1_v[i,j,:] = img[i,j,:]
        t_mean_1 = img1_t.mean(axis=1).mean(axis=0)
        v_mean_1 = img1_v.mean(axis=1).mean(axis=0)
        return t_mean_1, v_mean_1, img1_v
    
    def applyGabor(img):    
        def build_filters():
            filters_scales = []
            for lambd in np.arange(5,10,1):
                filters = []
                ksize = 16
                for theta in np.arange(0, np.pi, np.pi / 32):
                    params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':lambd,
                              'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
                    kern = cv2.getGaborKernel(**params)
                    kern /= 1.5*kern.sum()
                    filters.append((kern,params))
                filters_scales.append(filters)
                return filters_scales
    
        def process(img, filters):
            results = []
            for kern,params in filters:
                fimg = cv2.filter2D(img, cv2.CV_32FC3, kern)
                results.append(fimg)
            return results
    
    
        # main
        g = img
        filters = build_filters()
        multip_img = np.ones(g.shape)
        for i in range(len(filters)):
            filtered_images = process(g, filters[i])
            summed_img = np.zeros(filtered_images[0].shape)
            for i in range(len(filtered_images)):
                summed_img += np.abs(filtered_images[i]-1)
            multip_img *= summed_img
        bin_img = (multip_img[:,:]>17).astype(float)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        erosion = cv2.erode(bin_img,kernel,iterations = 2)
        return (erosion).astype('uint8')
    
    # def process(img, filters):
    #     results = []
    #     for kern,params in filters:
    #         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    #         results.append(fimg)
    #     return results
    
    # def applyGabor(img):
    #     filters = []
    #     ksize = 16
    #     for theta in np.arange(0, np.pi, np.pi / 32):
    #         params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
    #                   'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
    #         kern = cv2.getGaborKernel(**params)
    #         kern /= 1.5*kern.sum()
    #         filters.append((kern,params))
    #     filtered_images = process(img, filters)
    #     summed_img = np.zeros(filtered_images[0].shape)
    #     for i in range(len(filtered_images)):
    #         summed_img += np.abs(filtered_images[i]-1)
    #     summed_img[summed_img>0]=1
    #     return summed_img
    
    t_list = []
    v_list = []
    img_v_list = []
    thresh_list = []
    for i in range(len(ratio_list)):
        #thresh_1 = (1-ratio_list[i][:,:]>0.3)
        thresh_1 = applyGabor(ratio_list[i])
        thresh_list.append(thresh_1)
        plt.subplot(1,len(img_list),i+1)
        plt.imshow(thresh_1)
        t1,v1, img_v = get_avg_bkg(img_list[i],thresh_1)
        img_v_list.append(img_v)
        t_list.append(t1)
        v_list.append(v1)
    
    import scipy.io as sio
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    def get_range(x,y,xlims):
        y = y[x>=xlims[0]]
        x = x[x>=xlims[0]]
        y = y[x<=xlims[1]]
        x = x[x<=xlims[1]]
        return x, y
    
    # Load the absorbance data and perform a spectral fit
    
    import pandas as pd
    oxy_fname = 'D:\\Documents\\Alfredo_Projects\\Hyperspectral\\git_folder\\extinction_coeffs\\oxy_5nm.csv'
    #oxy_fname = 'C:\\Users\\Alfredo\\Documents\\University\\FCE\\hyperspectral\\git_folder\\extinction_coeffs\\oxy.csv'
    deoxy_fname = 'D:\\Documents\\Alfredo_Projects\\Hyperspectral\\git_folder\\extinction_coeffs\\deoxy_5nm.csv'
    #deoxy_fname = 'C:\\Users\\Alfredo\\Documents\\University\\FCE\\hyperspectral\\git_folder\\extinction_coeffs\\deoxy.csv'
    moxy = pd.read_csv(oxy_fname)
    mdeoxy = pd.read_csv(deoxy_fname)
    
    moxy_lamb = moxy['Wavelength'].values
    mdeoxy_lamb = mdeoxy['Wavelength'].values
    
    moxy_val = moxy['Absorbance'].values
    mdeoxy_val = mdeoxy['Absorbance'].values
    
    moxy_lamb_abs, moxy_abs = get_range(moxy_lamb,moxy_val,[450,600])
    mdeoxy_lamb_abs, mdeoxy_abs = get_range(mdeoxy_lamb,mdeoxy_val,[450,600])
    
    moxy_lamb_refl, moxy_refl = get_range(moxy_lamb,np.log(1/moxy_val),[450,600])
    mdeoxy_lamb_refl, mdeoxy_refl = get_range(mdeoxy_lamb,np.log(1/mdeoxy_val),[450,600])
    
    # Define the absorbance and scattering of skin
    
    def mua(wav):
        #mua.skinbaseline
        return 0.244 + 85.3*np.exp(-(wav - 154)/66.2)
    
    def musp(wav):
        mie = (2*(10**5))*(wav**(-1.5))
        ray = (2*(10**12))*(wav**(-4))
        return mie+ray
    
    def mu_eff(wav):
        return np.sqrt(3*mua(wav)*(mua(wav) + musp(wav)))
    
    def eps(val_abs,lamb,wav):
        eps_val =[]
        for i in range(len(wav)):
            eps_val.append(val_abs[find_band(lamb,wav[i])])
        return eps_val
    
    #%% Create a custom model for fitting the saturation
    from astropy.modeling import models, fitting
    from astropy.modeling.models import custom_model
    import time
    from slacker import Slacker
    slackClient = Slacker('xoxb-419910545015-419018721605-mdJSoOh18yD0lSzXLAxIHC5b')
    @custom_model
    def spec_fit(lamb, b0=1, b1=0.005, chb=1, chbo=1):
        epshb = eps(mdeoxy_abs,lamb=mdeoxy_lamb_abs,wav=lamb)
        epshbo = eps(moxy_abs,lamb=moxy_lamb_abs,wav=lamb)
        return b0 + mu_eff(lamb)*b1 + chbo*epshbo + chb*epshb
    
    
    spec_fit_1 = spec_fit()
    fitter = fitting.LevMarLSQFitter()
    
    try:
        for ii in range(len(img_list)):
            img_test = np.log(t_list[ii]/img_v_list[ii])
            img_test = img_test[7:-7,7:-7]
            sat_img = np.zeros([img_test.shape[0], img_test.shape[1]])
            for i in range(img_test.shape[0]):
                print(i)
                start_time = time.time()
                for j in range(img_test.shape[1]):
                    img_spec = img_test[i,j,:]
                    if ~(np.isinf(img_spec).all()):
                        img_spec = img_spec[find_band(lamb,450):find_band(lamb,600)]
                        lamb_a = lamb[find_band(lamb,450):find_band(lamb,600)]
                        fit_sat = fitter(spec_fit_1, lamb_a, img_spec,maxiter=200, acc=0.0001)
                        sat_val = fit_sat.chbo.value/(fit_sat.chb.value+fit_sat.chbo.value)
                        if sat_val>1:
                            sat_val=1
                        elif sat_val<0:
                            sat_val=0.0001
                        sat_img[i,j] = sat_val
                print("--- %s seconds ---" % (time.time() - start_time))
            sat_img_list.append(sat_img)
        
        #% Send text to slack when code finishes running.
        message = 'Analysis for ' + files[k]+ ' done!'
        slackClient.chat.post_message('#hyperspectral',message)
        del img_list, ratio_list, img, img_all, img_test, img_v
    except MemoryError:
        message = 'Memory error on image iteration:'+str(ii)+ '!'
        slackClient.chat.post_message('#hyperspectral',message)

message = 'Full analysis for done!'
slackClient.chat.post_message('#hyperspectral',message)
save_obj(sat_img_list,sat_fname)