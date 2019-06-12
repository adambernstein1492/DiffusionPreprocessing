import numpy as np
import nibabel as nib
import dipy.io
import scipy.special
import scipy.ndimage
import time
from .util import progress_update


def lpca_denoising(data_path, bvals_path, bvecs_path, out_path, mask_path="None", kernel_size=5, noise_map_kernel_size=3, threshold_factor=2.3):
    # Load in the Data
    data_nii,bvals,_ = load_diffusion_data(data_path, bvals_path, bvecs_path)
    data = data_nii.get_data().astype('float64')

    # Load in Mask (or make fake mask)
    if mask_path != "None":
        mask = nib.load(mask_path)
        mask = mask.get_data()

    else:
        mask = np.ones((data.shape[0], data.shape[1], data.shape[2]))


    # Estimate the Noise Map
    threshold_value = noise_map(data,bvals)

    # Allocate Space for Output
    data_padded = np.zeros((data.shape[0]+kernel_size-1, data.shape[1]+kernel_size-1, data.shape[2]+kernel_size-1, data.shape[3]))
    data_padded[kernel_size//2:data.shape[0]+kernel_size//2,kernel_size//2:data.shape[1]+kernel_size//2,kernel_size//2:data.shape[2]+kernel_size//2,:] = data
    data_denoised_padded = np.zeros((data.shape[0]+kernel_size-1, data.shape[1]+kernel_size-1, data.shape[2]+kernel_size-1, data.shape[3]))
    weight_img = np.zeros(data_denoised_padded.shape)

    # Perform PCA Denoising
    count = 0.0
    percent_prev = 0.0


    t0 = time.time()

    num_vox = data.shape[0] * data.shape[1] * data.shape[2]
    for i in range (data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):

                # Check if in mask
                in_mask = np.sum(mask[i:i+kernel_size,j:j+kernel_size,k:k+kernel_size])

                if in_mask != 0:
                    # Isolate local data cube
                    cube = data_padded[i:i+kernel_size,j:j+kernel_size,k:k+kernel_size,:]
                    pca_data = np.reshape(cube, (kernel_size**3, data.shape[3]))

                    # Local PCA
                    eigvals, eigvecs, princomps, mean_value = pca(pca_data)

                    # Reduce Data
                    filt_pca_data, num_eig_vals = filter_prin_comps(eigvals, eigvecs, princomps, threshold_value[i,j,k])
                    filt_pca_data += mean_value
                    filt_pca_data = np.reshape(filt_pca_data, (kernel_size, kernel_size, kernel_size, data.shape[3]))

                    # Weight Denoised Values
                    data_denoised_padded[i:i+kernel_size,j:j+kernel_size,k:k+kernel_size,:] += (filt_pca_data * (1.0/num_eig_vals))
                    weight_img[i:i+kernel_size,j:j+kernel_size,k:k+kernel_size,:] += (1.0/num_eig_vals)

                # Update Progress
                count += 1.0
                percent = np.around((count / num_vox * 100), decimals = 1)
                if(percent != percent_prev):
                    progress_update("Denoising: ", percent)
                    percent_prev = percent

    t1 = time.time()

    total = t1-t0
    print(total)

    # Set final image value
    data_denoised_padded /= weight_img
    data_denoised = data_denoised_padded[kernel_size//2:data.shape[0]+kernel_size//2, kernel_size//2:data.shape[1]+kernel_size//2, kernel_size//2:data.shape[2]+kernel_size//2, :]

    # Save Denoised Image to NIFTI
    denoised_img = nib.Nifti1Image(data_denoised, data_nii.affine, data_nii.header)
    nib.save(denoised_img, out_path)

    print("Done Denoising")



def load_diffusion_data(dwi_file, bvals_file, bvecs_file):
    dwi = nib.load(dwi_file)
    bvals,bvecs = dipy.io.read_bvals_bvecs(bvals_file, bvecs_file)

    return dwi,bvals,bvecs


def filter_prin_comps(eigvals,eigvecs,prin_comps,threshold):
    # Count number of Eigvals above threshold
    count = 0
    for i in range(eigvals.shape[0]):
        if(eigvals[i] >= threshold):
            count += 1
        else:
            pass

    # Allocate Space for new Eigen Vector and Principal Component matrices
    eigvecs_filt = np.zeros((eigvecs.shape[0],count))
    prin_comps_filt = np.zeros((prin_comps.shape[0],count))

    # Reshape the Matrics
    index = 0
    for i in range(eigvals.shape[0]):
        if(eigvals[i] >= threshold):
            eigvecs_filt[:,index] = eigvecs[:,i]
            prin_comps_filt[:,index] = prin_comps[:,i]
            index += 1

    index += 1
    filtered_data = np.dot(prin_comps_filt, eigvecs_filt.T)

    return filtered_data, index



def find_b0s(data, bvals):
    # Determine Number of B0s
    b0_cnt = 0
    for b in bvals:
        if b <= 100:
            b0_cnt += 1

    b0 = np.zeros((data.shape[0],data.shape[1],data.shape[2],b0_cnt))
    idx = 0
    b0_cnt = 0
    for b in bvals:
        if b <= 100:
            b0[:,:,:,b0_cnt] = data[:,:,:,idx]
            b0_cnt += 1
            idx += 1

    if b0_cnt >= 2:
        data = b0
    else:
        pass

    return data


def noise_map(data, bvals, kernel_size=3, threshold_factor=2.3):
    data = find_b0s(data, bvals)

    # Assume smallest principal component is noise
    data_reshape = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]))
    eigvals,_,prin_comps,_ = pca(data_reshape)
    for i in range(eigvals.shape[0]):
        if eigvals[i] == np.min(eigvals):
            noise_est = np.reshape(prin_comps[:,i], (data.shape[0], data.shape[1], data.shape[2]))
            break

    # Find STD of the noise estimate and AVG signal
    noise = np.zeros((data.shape[0]+2, data.shape[1]+2, data.shape[2]+2))
    noise[1:data.shape[0]+1,1:data.shape[1]+1,1:data.shape[2]+1] = noise_est
    signal = np.zeros((data.shape[0]+2, data.shape[1]+2, data.shape[2]+2))
    signal[1:data.shape[0]+1,1:data.shape[1]+1,1:data.shape[2]+1] = np.mean(data,axis=3)
    noise_map = np.zeros((data.shape[0],data.shape[1],data.shape[2]))


    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                std = np.std(noise[i:i+3,j:j+3,k:k+3])
                sig = np.mean(signal[i:i+3,j:j+3,k:k+3])

                if sig == 0:
                    snr = 0
                elif std == 0 and sig != 0:
                    snr = sig / 0.01
                else:
                    snr = sig/std

                # Correct for Rician Noise
                I0 = scipy.special.jv(0,snr**2/4)
                I1 = scipy.special.jv(1,snr**2/4)
                adjusted_snr = 2 + snr**2 - np.pi/8 * np.exp(-(snr**2/2)) * ((2+snr**2)*I0 + snr**2*I1)**2
                if adjusted_snr > 1:
                    adjusted_snr = 1

                noise_map[i,j,k] = (std ** 2 / adjusted_snr) ** 0.5

    # Filter the Noise Map
    noise_map = (scipy.ndimage.filters.gaussian_filter(noise_map,7.5)) ** 2

    print("Done making noise map")
    return (noise_map * threshold_factor ** 2)


def pca(data):
    # Zero Center the Data
    mean_values = data.mean(axis=0)
    data = data - mean_values

    # Diagonalize the Covariance Matrix
    eigvals, eigvecs = np.linalg.eigh(np.cov(data.T))

    # Calculate the Principal Components
    prin_comps = np.dot(data,eigvecs)

    return eigvals, eigvecs, prin_comps, mean_values
