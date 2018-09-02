
import os
import spectral
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pickle
#%% Define the custom functions to be used
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

def get_pomega(mat1,k):
    mat = mat1.copy()
    vals = np.reshape(mat,[mat.shape[0]*mat.shape[1]])
    vals.sort()
    vals_k = np.zeros(mat.shape)
    count_k = 0
    while count_k < k:
        i,j = np.where(mat1==vals[-1-count_k])
        vals_k[i,j] = vals[-1-count_k]
        count_k += 1
    return vals_k

def GoDec(X,rank,card,power, error_bound):
    iter_max=1e+2
    count=1
    RMSE=[]

    #matrix size
    m,n =np.shape(X)
    if m<n:
        X=X.T

    #initialization of L and S
    L=X
    S=np.zeros(X.shape)

    while True:

        #Update of L
        Y2=np.random.rand(n,rank)
        for i in range(power+1):
            Y1=np.matmul(L,Y2)
            Y2=np.matmul(L.T,Y1)
        Q,R=np.linalg.qr(Y2)
        LQ = np.matmul(L,Q)
        L_new=np.matmul(LQ,Q.T)

        #Update of S
        T=L-L_new+S
        L=L_new
        S = get_pomega(T,card)

        #Error, stopping citeria
        T=T-S
        RMSE.append(np.linalg.norm(T,ord=2))
        if (RMSE[-1]<error_bound) or (count>iter_max):
            break
        else:
            L=L+T

        count += 1


    LS=L+S
    error=np.linalg.norm(LS-X,ord=2)/np.linalg.norm(X,ord=2)
    if m<n:
        LS=LS.T
        L=L.T
        S=S.T
    return L,S



os.chdir('F:\\Research Data\\hyperspectral\\dilution\\H006\\filtered')
lib, img = load_hyper_image('E3_2_c.bil')
lamb = np.array(lib['wavelength']).astype(float)
img_all = img.load()
img_all = img_all[0:250,0:500,:]
## Apply the GoDec algorithm to the image
# Initialize the first patch
step=5
q = 25
img_filt = np.zeros(img_all.shape)
patch = img_all[0:q,0:q,:]
patch_mat = np.zeros([q*q,img_all.shape[2]])
for k in range(patch.shape[2]):
    patch_mat[:,k] = np.reshape(patch[:,:,k],q*q)
L,S = GoDec(patch_mat,7,4000,10,error_bound=10)
L_mat = np.reshape(L,[q,q,300])
img_filt[0:q,0:q,:] = L_mat
patch_old = L_mat

# Initialize first column
for i in range(step,img_all.shape[0]-q,step):
    j=0
    patch = img_all[0+i:q+i,0+j:q+j,:]
    for k in range(patch.shape[2]):
        patch_mat[:,k] = np.reshape(patch[:,:,k],q*q)
    L,S = GoDec(patch_mat,7,4000,10,error_bound=10)
    print(i)
    L_mat = np.reshape(L,[q,q,300])
    patch_new = L_mat.copy()
    patch_new[0:-step,:] = (patch_new[0:-step,:]+patch_old[step:,:])/2
    img_filt[0+i:q+i,0+j:q+j,:] = patch_new
    patch_old = patch_new

# Initialize first row
patch_old = img_filt[0+i:q+i,0+j:q+j,:]

for j in range(step,img_all.shape[1]-q,step):
    i=0
    patch = img_all[0+i:q+i,0+j:q+j,:]
    for k in range(patch.shape[2]):
        patch_mat[:,k] = np.reshape(patch[:,:,k],q*q)
    L,S = GoDec(patch_mat,7,4000,10,error_bound=10)
    print(j)
    L_mat = np.reshape(L,[q,q,300])
    patch_new = L_mat.copy()
    patch_new[:,0:-step] = (patch_new[:,0:-step]+patch_old[:,step:])/2
    img_filt[0+i:q+i,0+j:q+j,:] = patch_new
    patch_old = patch_new

# Run the algorithm in the remaining patches
for i in range(step,img_all.shape[0]-q,step):
    for j in range(step*2,img_all.shape[1]-q,step):
        patch = img_all[0+i:q+i,0+j:q+j,:]
        for k in range(patch.shape[2]):
            patch_mat[:,k] = np.reshape(patch[:,:,k],q*q)
        L,S = GoDec(patch_mat,7,4000,10,error_bound=10)
        print('Column: ',j, '. Row: ', i)
        L_mat = np.reshape(L,[q,q,300])
        patch_new = L_mat.copy()
        patch_new[0:-step,0:-step] = (patch_new[0:-step,0:-step]+patch_old[step:,step:])/2
        img_filt[0+i:q+i,0+j:q+j,:] = patch_new
        patch_old = patch_new