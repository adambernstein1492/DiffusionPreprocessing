import numpy as np
import dipy.io
import nibabel as nib
import os

def write_index_acqp(dwi_file, bval_file, bvec_file, rpe_file, echo_spacing=0.05):
    # Create text file containing as many ones as there are diffusion images
    bvals,_ = dipy.io.read_bvals_bvecs(bval_file, bvec_file)
    index_file = np.ones(bvals.shape)

    np.savetxt('index.txt', index_file, delimiter=' ', fmt='%d')

    # Create text file with acquisition parameters for b0s
    dwi = nib.load(dwi_file)
    dwi_data = dwi.get_data()
    rpe = nib.load(rpe_file)
    rpe_data = rpe.get_data()

    AP = 0
    for i in bvals:
        if i <= 100:
            AP += 1

    if len(rpe.shape) == 3:
        PA = 1

        # Same number of AP and PA images used for TOPUP
        if AP > 1:
            AP = 1

    elif len(rpe.shape) == 4:
        PA = rpe.shape[3]

        # Same number of AP and PA images used for TOPUP
        if AP > PA:
            AP = PA

    acqp_file = np.zeros((AP + PA, 4))

    for i in range(AP + PA):
        if i < AP:
            acqp_file[i,1] = -1
        else:
            acqp_file[i,1] = 1

        acqp_file[i,3] = echo_spacing

    np.savetxt('acqp.txt', acqp_file, delimiter=' ', fmt = '%f')

    # Combine all B0 images and normalize by the mean
    B0s = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], AP + PA))
    index = 0
    count = 0
    for i in bvals:
        if ((i <= 100) and (count < AP)):
            B0s[:,:,:,index] = dwi_data[:,:,:,count]/np.mean(dwi_data[:,:,:,count])
            index += 1
        count += 1

    if PA > 1:
        for i in range(PA):
            B0s[:,:,:,i+index] = rpe_data[:,:,:,i]/np.mean(rpe_data[:,:,:,i])
    elif PA == 1:
        B0s[:,:,:,index] = rpe_data/np.mean(rpe_data)

    # Write out B0s file
    b0s_img = nib.Nifti1Image(B0s, dwi.affine, dwi.header)
    nib.save(b0s_img, 'B0s.nii')


def no_topup_index_acqp(bval_file, bvec_file):
    acqp_file = np.zeros((1, 4))
    acqp_file[0,1] = -1
    acqp_file[0,3] = 0.05
    np.savetxt('acqp.txt', acqp_file, delimiter=' ', fmt = '%f')

    bvals,_ = dipy.io.read_bvals_bvecs(bval_file, bvec_file)
    index_file = np.ones((1, bvals.shape[0]))

    np.savetxt('index.txt', index_file, delimiter=' ', fmt = '%d')


def create_bvals_bvecs_rpe(rpe_file):
    rpe = nib.load(rpe_file)
    if len(rpe.shape) == 3:
        num_imgs = 1
    elif len(rpe.shape) == 4:
        num_imgs = rpe.shape[3]

    bvals = np.zeros((1,num_imgs))
    bvecs = np.zeros((3,num_imgs))

    np.savetxt('rpe_bval', bvals, delimiter=' ', fmt='%d')
    np.savetxt('rpe_bvec', bvecs, delimiter=' ', fmt='%d')


def create_avg_b0(dwi_file, bval_file, bvec_file):
    dwi = nib.load(dwi_file)
    dwi_data = dwi.get_data()

    bvals,_ = dipy.io.read_bvals_bvecs(bval_file, bvec_file)

    count = 0
    for i in bvals:
        if i < 50:
            count += 1

    b0s = np.zeros((dwi_data.shape[0], dwi_data.shape[1], dwi_data.shape[2], count))

    index = 0
    count = 0
    for i in bvals:
        if i < 50:
            b0s[:,:,:,index] = dwi_data[:,:,:,count]
            index += 1
        count += 1

    b0s = np.mean(b0s,axis=3)
    mean_b0 = nib.Nifti1Image(b0s, dwi.affine, dwi.header)
    nib.save(mean_b0, 'mean_b0.nii')


def apply_bias_field(dwi_file, bias_field_file):
    dwi = nib.load(dwi_file)
    dwi_data = dwi.get_data()

    bias_field = nib.load(bias_field_file)
    bias_data = bias_field.get_data()

    for i in range(dwi.shape[3]):
        dwi_data[:,:,:,i] = dwi_data[:,:,:,i] / bias_data

    dwi = nib.Nifti1Image(dwi_data, dwi.affine, dwi.header)
    nib.save(dwi, 'N4CorrectedDWI.nii')

def remove_header(img_path, save_path):
    img = nib.load(img_path)
    img_data = img.get_data()
    new_img = nib.Nifti1Image(img_data, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    nib.save(new_img, save_path)

def organize_by_bval(dwi_file, bval_file, bvec_file):
    dwi = nib.load(dwi_file)
    dwi_data = dwi.get_data()
    bvals,bvecs = dipy.io.read_bvals_bvecs(bval_file, bvec_file)

    # Count number of b=0 and other images
    low_count = 0
    high_count = 0
    for i in bvals:
        if i < 50:
            low_count += 1
        if i >= 50:
            high_count +=1

    # Allocate Space
    b0s = np.zeros((dwi.shape[0],dwi.shape[1],dwi.shape[2],low_count))
    diffs = np.zeros((dwi.shape[0],dwi.shape[1],dwi.shape[2],high_count))
    bvals_low = np.zeros((1,low_count))
    bvecs_low = np.zeros((low_count,3))
    bvals_high = np.zeros((1,high_count))
    bvecs_high = np.zeros((high_count,3))

    low_index = 0
    high_index = 0
    index = 0
    for i in bvals:
        if i < 50:
            b0s[:,:,:,low_index] = dwi_data[:,:,:,index]
            bvals_low[0,low_index] = bvals[index]
            bvecs_low[low_index,:] = bvecs[index,:]
            low_index += 1
        if i >= 50:
            diffs[:,:,:,high_index] = dwi_data[:,:,:,index]
            bvals_high[0,high_index] = bvals[index]
            bvecs_high[high_index,:] = bvecs[index,:]
            high_index += 1
        index += 1

    # Concatenate everything
    dwi_data = np.concatenate((b0s,diffs), axis=3)
    bvals = np.concatenate((bvals_low,bvals_high), axis=1)
    bvecs = np.concatenate((bvecs_low,bvecs_high), axis=0)

    # Save new data
    dwi = nib.Nifti1Image(dwi_data, dwi.affine, dwi.header)
    nib.save(dwi, dwi_file)
    np.savetxt(bval_file, bvals, delimiter=' ', fmt='%d')
    np.savetxt(bvec_file, bvecs.T, delimiter=' ', fmt='%f')


def filter_bvals(dwi, bvals, bvecs, bval_threshold=10000):

    count = 0
    for i in range(bvals.shape[0]):
        if bvals[i] <= bval_threshold:
            count += 1

    bval_thresh = np.zeros((count))
    bvec_thresh = np.zeros((count,3))
    dwi_thresh = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], count))

    index = 0
    for i in range(bvals.shape[0]):
        if bvals[i] <= bval_threshold:
            bval_thresh[index] = bvals[i]
            bvec_thresh[index,:] = bvecs[i,:]
            dwi_thresh[:,:,:,index] = dwi[:,:,:,i]
            index += 1

    return dwi_thresh, bval_thresh, bvec_thresh
