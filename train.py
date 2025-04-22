import time, sys, os
from copy import deepcopy
from random import shuffle, randint
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import models1
import losses
import weight_init

import torchvision.models as pretrained_models

#import utils
torch.cuda.set_device(0)

def augment_image(img, augmentation_type, augmentation_magnitude, img_type):
    """
    This function augments the image. It can be unchanged, mirrored or traslated.
    """

    # Create empty image according to image type
    if img_type == 'mri' or img_type == 'skin':
        augmented_img = np.zeros_like(img)
    elif img_type == 'ct':
        augmented_img = np.ones_like(img)*-1000

    # Modify image according to the augmentation type
    if augmentation_type == -1:
        augmented_img[:,:] = img[:,:]
    elif augmentation_type == 0:
        augmented_img[int(augmentation_magnitude+1):,:] = img[:-int(augmentation_magnitude+1),:]
    elif augmentation_type == 1:
        augmented_img[:-int(augmentation_magnitude+1),:] = img[int(augmentation_magnitude+1):,:]
    elif augmentation_type == 2:
        augmented_img[:,int(augmentation_magnitude+1):] = img[:,:-int(augmentation_magnitude+1)]
    elif augmentation_type == 3:
        augmented_img[:,:-int(augmentation_magnitude+1)] = img[:,int(augmentation_magnitude+1):]
    elif augmentation_type == 4:
        augmented_img[:,:] = img[:,::-1]
    elif augmentation_type == 5:
        augmented_img[:,:] = img[::-1,:]
    elif augmentation_type == 6:
        augmented_img[:,:] = img[::-1,::-1]

    return augmented_img


def binarize_img(img):
    """
    This function binarizes the label image given as input.
    """

    img = img[:,:,:].astype(np.float32)
    img -= img.min()
    img /= img.max()
    img[img>=0.5]=1
    img[img!=1]=0

    return img


def rotate_image(img, orientation):
    """
    This function swaps axes in order to have the data in SAC format.
    """

    if orientation == "SAC": # Sagittal, Axial, Coronal
        return img
    elif orientation == "ACS": # Axial, Coronal, Sagittal:
        m1 = np.swapaxes(img, 0, 1) # CAS
        return np.swapaxes(m1, 0, 2) # SAC
    else:
        print("Unknow orientation...Exit")
        exit()


def reshape_image(img, background_value=0.0):
    """
    This function reshapes the image to a cubic matrix.
    """

    img_shape = img.shape
    max_image_shape = max(img_shape)
    offsets = (int(np.floor(max_image_shape-img_shape[0])/2.0), int(np.floor(max_image_shape-img_shape[1])/2.0), int(np.floor(max_image_shape-img_shape[2])/2.0))

    reshaped_img = np.ones((max_image_shape, max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
    reshaped_img[offsets[0]:offsets[0]+img_shape[0],offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:]

    return reshaped_img.astype(np.float32)


def read_image(img_fn, orientation, binary_img=False, background_value=0.0):
    """
    This function reads, roatates and binarizes (if needed) the image.
    """

    img = sitk.GetArrayFromImage(sitk.ReadImage(img_fn)).astype(np.float32)
    img = rotate_image(img, orientation)
    img = reshape_image(img, background_value)

    if binary_img:
        return binarize_img(img).astype(np.float32)
    else:
        return img.astype(np.float32)


def generate_synthetic_data(ct, mri, skin, thr=200):

    # this function works only with single channel MRI
    assert ct.shape == mri.shape

    original_shape = ct.shape

    ct = ct.reshape((original_shape[0], original_shape[2], original_shape[3]))
    mri = mri.reshape((original_shape[0], original_shape[2], original_shape[3]))
    skin = skin.reshape((original_shape[0], original_shape[2], original_shape[3]))

    # Define what we want to keep and to be unchanged into the synthetic image
    to_keep = deepcopy(skin)
    to_keep[ct<thr]=0
    to_keep[ct<-800]=1
    to_keep[ct<-999]=0

    # Coords to be used for random sampling
    x_s,y_s,z_s = np.nonzero(to_keep)

    # Define the area to be filled with random samples
    to_fill = (1 - to_keep)*skin
    x_f,y_f,z_f = np.nonzero(to_fill)

    # prepare the images to be changed
    sct = deepcopy(ct)
    sct[to_keep!=1]=-1000

    smri = deepcopy(mri)
    smri[to_keep!=1]=0

    # Replace values
    for i in range(x_f.shape[0]):
        random_index = randint(0,x_s.shape[0]-1)
        sample_coords = x_s[random_index], y_s[random_index], z_s[random_index]

        sct[x_f[i],y_f[i],z_f[i]]=ct[sample_coords[0],sample_coords[1],sample_coords[2]]
        smri[x_f[i],y_f[i],z_f[i]]=mri[sample_coords[0],sample_coords[1],sample_coords[2]]

    return sct.reshape(original_shape), smri.reshape(original_shape)


def compute_l1_norm(model, lambda1=0.5):

    l1_regularization = torch.tensor(0).float().cuda()

    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)

    return lambda1 * l1_regularization


def create_list(cases, max_image_shape, validation=False):
    """
    This function creates a list for training/validation.
    """

    list_to_return=[] # {train_case, view, slice_number, augmentation_type}
    for case in range(len(cases)):
        for view in ["sagittal", "axial", "coronal"]:
            for slice_index in range(max_image_shape):
                if validation == False: # so it is training
                    for augmentation_type in range(-1,7):
                        list_to_return.append({"case": case, "view":view, "slice_index":slice_index, "augmentation_type":augmentation_type})
                elif validation == True: # so it is validation
                    list_to_return.append({"case": case, "view":view, "slice_index":slice_index, "augmentation_type":-1})
    return list_to_return


def run_training(model, trainCases, epoch, lp, max_image_shape):
    """
    This is the function that reads the training case images, augments the data, creates batches and trains the model.
    """

    epochStartTime = time.process_time()

    # Define metric and optimizer
    if lp["loss"]=="global":
        loss_function = losses.GlobalMAE()
    elif lp["loss"] == "masked":
        loss_function = losses.MaskedMAE()
    elif lp["loss"]=="weighted-masked":
        loss_function = losses.MaskedWeightedMAE()
    elif lp["loss"]=="masked-FFT":
        loss_function = losses.MaskedMAEPlusFFT()
    elif lp["loss"]=="masked-SSIM":
        loss_function = losses.MaskedMAEPlusSSIM()
    elif lp["loss"]=="alternated-MAE":
        loss_function_plain = losses.MaskedMAE()
        loss_function_weighted = losses.MaskedWeightedMAE()
        plain_weighted_ratio = int(np.ceil(epoch/2.0))
    elif lp["loss"] == "synthetic-MAE":
        loss_function = losses.MaskedMAE()
        plain_weighted_ratio = int(np.ceil(epoch/2.0))
    elif lp["loss"] == "content":
        loss_function_content = torch.nn.MSELoss()
        loss_function_maskedMAE = losses.MaskedMAE()

        vgg = pretrained_models.vgg16(pretrained=True)
        vgg.cuda()

        vgg.eval()

    lr = lp["learningRate"][-1]
    if epoch < len(lp["learningRate"]):
        lr = lp["learningRate"][epoch]

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay= lp["lambda"], momentum = 0.9, nesterov = True)
    
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= lp["lambda"])

    #optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay= lp["lambda"])
    #optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay= lp["lambda"])

    # Set training status
    model.train()

    print ("Run training function....")
    reportPath = lp["netPath"] + lp["selectedView"] + "_report.txt"

    # Read and store training images
    mris = np.zeros((len(trainCases), len(lp["MRIchannels"]), max_image_shape, max_image_shape, max_image_shape), dtype='float32')
    cts = np.zeros((len(trainCases), max_image_shape, max_image_shape, max_image_shape), dtype='float32')
    skins = np.zeros((len(trainCases), max_image_shape, max_image_shape, max_image_shape), dtype='float32')

    for train_case_index, train_case in enumerate(trainCases):
        print ("{t})loading images...".format(t=train_case_index))
        train_case_path = lp["dataPath"] + train_case + os.sep
        cts[train_case_index,:,:,:] = read_image(train_case_path+lp["CT"], lp["orientation"], background_value=-1000)
        skins[train_case_index,:,:,:] = read_image(train_case_path+lp["skin"], lp["orientation"], binary_img=True, background_value=0)
        for channel_index, channel in enumerate(lp["MRIchannels"]):
            mris[train_case_index,channel_index,:,:,:] = read_image(train_case_path+channel, lp["orientation"], background_value=0)

    # Create lists for training
    list_for_training=create_list(trainCases, max_image_shape) # {train_case, view, slice_number, augmentation_type}
    indexes_shuf = list(range(len(list_for_training)))
    shuffle(indexes_shuf)
    augmentation_magnitudes = np.random.randint(15, size=len(list_for_training))

    # Initialize data batch for training
    mris_batch = np.zeros((lp["batchSize"], len(lp["MRIchannels"]), max_image_shape, max_image_shape), dtype='float32')
    ct_batch = np.zeros((lp["batchSize"], 1, max_image_shape, max_image_shape), dtype='float32')
    skin_batch = np.zeros((lp["batchSize"], 1, max_image_shape, max_image_shape), dtype='float32')
    losses_list=[]

    # Loop over all the computed combinations
    batchIndex=0
    number_of_slices_to_show = len(indexes_shuf)
    balancing_index=0
    synthetic_batch = False
    for i, index_shuf in enumerate(indexes_shuf):

        # Extract once all the variables to use for the current combination
        current_train_case = list_for_training[index_shuf]["case"]
        current_view = list_for_training[index_shuf]["view"]
        current_slice_index = list_for_training[index_shuf]["slice_index"]
        current_augmentation_type = list_for_training[index_shuf]["augmentation_type"]
        current_augmentation_magnitude = augmentation_magnitudes[index_shuf]

        ct = None
        skin = None
        mri_channels = []

        # Sagittal view
        if current_view=="sagittal" and lp["selectedView"] in ["sagittal", "mixed"]:
            if 1 in skins[current_train_case, current_slice_index, :, :]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_train_case, channel, current_slice_index, :, :]))
                # CT
                ct = deepcopy(cts[current_train_case, current_slice_index, :, :])
                # SKIN
                skin = deepcopy(skins[current_train_case, current_slice_index, :, :])

        # Axial view
        elif current_view=="axial" and lp["selectedView"] in ["axial", "mixed"]:
            if 1 in skins[current_train_case, :, current_slice_index, :]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_train_case, channel, :, current_slice_index, :]))
                # CT
                ct = deepcopy(cts[current_train_case,:,current_slice_index,:])
                # SKIN
                skin = deepcopy(skins[current_train_case, :, current_slice_index, :])

        # Coronal view
        elif current_view=="coronal" and lp["selectedView"] in ["coronal", "mixed"]:
            if 1 in skins[current_train_case, :, :, current_slice_index]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_train_case, channel, :, :, current_slice_index]))
                # CT
                ct = deepcopy(cts[current_train_case, :, :, current_slice_index])
                # Skin
                skin = deepcopy(skins[current_train_case, :, :, current_slice_index])

        # Assign images to batch
        if ct is not None and skin is not None and not len(mri_channels) == 0:
            for mri_channel_index, mri_channel in enumerate(mri_channels):
                mris_batch[batchIndex,mri_channel_index,:,:] = deepcopy(augment_image(mri_channel, current_augmentation_type, current_augmentation_magnitude, 'mri'))
            ct_batch[batchIndex,0,:,:] = deepcopy(augment_image(ct, current_augmentation_type, current_augmentation_magnitude, 'ct'))
            skin_batch[batchIndex,0,:,:] = deepcopy(augment_image(skin, current_augmentation_type, current_augmentation_magnitude, 'skin'))
            batchIndex+=1

        # If batch is full, start training
        if batchIndex == lp["batchSize"]:

            optimizer.zero_grad()

            #if lp["loss"] == "synthetic-MAE" and epoch!=0 and synthetic_batch: # CODE FOR ALTERNATE TRAINING
            if balancing_index == 0 and epoch!=0 and lp["loss"]=="syntetic-MAE":
                ct_batch, mris_batch = generate_synthetic_data(ct_batch, mris_batch, skin_batch, 300)

            # Convert numpy object to pytorch tensor
            ct_batch_tensor = torch.from_numpy(ct_batch).float().cuda()
            mris_batch_tensor = torch.from_numpy(mris_batch).float().cuda()
            skin_batch_tensor = torch.from_numpy(skin_batch).float().cuda()

            output = model(mris_batch_tensor)

            if lp["loss"]=="global":
                loss = loss_function(ct_batch_tensor, output)
                loss_type_str = "MAE"
            elif lp["loss"] == "masked":
                loss = loss_function(ct_batch_tensor, output, skin_batch_tensor)
                loss_type_str = "maskedMAE"
            elif lp["loss"] == "weighted-masked":
                loss = loss_function(ct_batch_tensor, output, skin_batch_tensor, lp["maeWeight"])
                loss_type_str = "weightedMAE"
            elif lp["loss"]=="masked-FFT":
                loss = loss_function(ct_batch_tensor, output, skin_batch_tensor, lp["fftWeight"])
                loss_type_str = "maskedFFT"
            elif lp["loss"]=="masked-SSIM":
                loss = loss_function(ct_batch_tensor, output, skin_batch_tensor, lp["ssimWeight"])
                loss_type_str = "maskedSSIM"
            elif lp["loss"] == "synthetic-MAE":
                """
                # CODE FOR ALTERNATE TRAINING
                if epoch == 0 or (epoch!=0 and not synthetic_batch):
                    loss = loss_function(ct_batch_tensor, output, skin_batch_tensor)
                    loss_type_str = "maskedMAE"
                elif epoch != 0 and synthetic_batch:
                    loss = loss_function(ct_batch_tensor, output, skin_batch_tensor, lp["syntheticWeight"])
                    loss_type_str = "syntheticMAE"

                if synthetic_batch == False:
                    synthetic_batch = True
                elif synthetic_batch == True:
                    synthetic_batch = False
                """

                if balancing_index == 0 and epoch!=0:
                    loss = loss_function(ct_batch_tensor, output, skin_batch_tensor, lp["syntheticWeight"])# SYNTHETIC
                    loss_type_str = "syntheticMAE"
                else:
                    loss = loss_function(ct_batch_tensor, output, skin_batch_tensor) # MASKED
                    loss_type_str = "maskedMAE"

                if balancing_index == plain_weighted_ratio:
                    balancing_index = 0
                else:
                    balancing_index += 1

            elif lp["loss"]=="alternated-MAE":
                if balancing_index == 0:
                    loss = loss_function_plain(ct_batch_tensor, output, skin_batch_tensor) # MASKED
                    loss_type_str = "maskedMAE"
                else:
                    loss = loss_function_weighted(ct_batch_tensor, output, skin_batch_tensor, lp["maeWeight"]) # WEIGHTED
                    loss_type_str = "weightedMAE"

                if balancing_index == plain_weighted_ratio:
                    balancing_index = 0
                else:
                    balancing_index += 1

            elif lp["loss"]=="content":
                loss_type_str = "content"

                loss_maskedMAE = loss_function_maskedMAE(ct_batch_tensor, output, skin_batch_tensor)

                ct_batch_vgg = deepcopy(ct_batch)
                ct_batch_vgg[skin_batch==0]=-1000
                ct_batch_vgg -= -1000.0
                ct_batch_vgg /= 2200.0
                ct_batch_vgg = np.concatenate((ct_batch_vgg, ct_batch_vgg, ct_batch_vgg),1)
                ct_batch_vgg_tensor = torch.from_numpy(ct_batch_vgg).float().cuda()

                output_for_vgg_tensor = output.clone()
                output_for_vgg_tensor[skin_batch_tensor==0]=-1000
                output_for_vgg_tensor -= -1000.0
                output_for_vgg_tensor /= 2200.0
                output_for_vgg_tensor = torch.cat((output_for_vgg_tensor, output_for_vgg_tensor, output_for_vgg_tensor),1)

                features_gt = vgg.features[0](ct_batch_vgg_tensor)
                features_output = vgg.features[0](output_for_vgg_tensor)

                loss_content = loss_function_content(features_gt, features_output)

                loss = loss_maskedMAE + (lp["contentWeight"] * loss_content)

                print(" masked MAE = %.3f  --  content = %.3f  --  total (NO l1) = %.3f" % (loss_maskedMAE, lp["contentWeight"] * loss_content, loss))

            # Add l1 penalty
            loss += compute_l1_norm(model, lp["lambda"])

            loss.backward()
            optimizer.step()

            losses_list.append(loss.item())

            print (">>> Epoch %d -- Train case index: %d -- Example %d/%d -- Loss (%s) = %.3f" % (epoch, current_train_case, i, number_of_slices_to_show, loss_type_str, loss.item()))
            batchIndex=0

    losses_mean=np.mean(losses_list)

    # Print on screen and write into the report the obtained epoch statistics 
    print ("TRAIN - epoch time %.3f -- mean loss = %.3f" % ((time.process_time()-epochStartTime)/60.0, losses_mean))
    with open(reportPath, "a") as report:
        report.write("Train - epoch %d ===== mean loss = %.3f\n" % (epoch, losses_mean))


def run_validation(valid_fn, validCases, epoch, lp, max_image_shape):
    """
    This is the function that reads the validation case images, augments the data, creates batches and valid the model.
    """

    epochStartTime = time.process_time()

    print ("Run training function....")
    reportPath = lp["netPath"] + lp["selectedView"] + "_report.txt"

    # Define metric and optimizer
    if lp["loss"]=="global":
        loss_function = losses.GlobalMAE()
    elif lp["loss"] in ["masked", "masked-FFT", "masked-SSIM", "weighted-masked", "alternated-MAE", "synthetic-MAE", "content"]:
        loss_function = losses.MaskedMAE()

    # Set validation status
    model.eval()

    # Read and store validation images
    mris = np.zeros((len(validCases), len(lp["MRIchannels"]), max_image_shape, max_image_shape, max_image_shape), dtype='float32')
    cts = np.zeros((len(validCases), max_image_shape, max_image_shape, max_image_shape), dtype='float32')
    skins = np.zeros((len(validCases), max_image_shape, max_image_shape, max_image_shape), dtype='float32')

    for valid_case_index, valid_case in enumerate(validCases):
        print ("{t})loading images...".format(t=valid_case_index))
        valid_case_path = lp["dataPath"] + valid_case + os.sep
        cts[valid_case_index,:,:,:] = read_image(valid_case_path+lp["CT"], lp["orientation"], background_value=-1000)
        skins[valid_case_index,:,:,:] = read_image(valid_case_path+lp["skin"], lp["orientation"], binary_img=True, background_value=0)
        for channel_index, channel in enumerate(lp["MRIchannels"]):
            mris[valid_case_index,channel_index,:,:,:] = read_image(valid_case_path+channel, lp["orientation"], background_value=0)

    # Create lists for validation
    list_for_validation=create_list(validCases, max_image_shape, validation=True) # {train_case, view, slice_number, augmentation_type}
    indexes_shuf = list(range(len(list_for_validation)))
    shuffle(indexes_shuf)
    augmentation_magnitudes = np.random.randint(15, size=len(list_for_validation))

    # Initialize data batch for training
    mris_batch = np.zeros((lp["batchSize"], len(lp["MRIchannels"]), max_image_shape, max_image_shape), dtype='float32')
    ct_batch = np.zeros((lp["batchSize"], 1, max_image_shape, max_image_shape), dtype='float32')
    skin_batch = np.zeros((lp["batchSize"], 1, max_image_shape, max_image_shape), dtype='float32')
    losses_list=[]

    # Loop over all the computed combinations
    batchIndex=0

    number_of_slices_to_show = len(indexes_shuf)

    for i, index_shuf in enumerate(indexes_shuf):

        # Extract once all the variables to use for the current combination
        current_valid_case = list_for_validation[index_shuf]["case"]
        current_view = list_for_validation[index_shuf]["view"]
        current_slice_index = list_for_validation[index_shuf]["slice_index"]
        current_augmentation_type = list_for_validation[index_shuf]["augmentation_type"]
        current_augmentation_magnitude = augmentation_magnitudes[index_shuf]

        ct = None
        skin = None
        mri_channels = []

        # Sagittal view
        if current_view=="sagittal" and lp["selectedView"] in ["sagittal", "mixed"]:
            if 1 in skins[current_valid_case, current_slice_index, :, :]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_valid_case, channel, current_slice_index, :, :]))
                # CT
                ct = deepcopy(cts[current_valid_case, current_slice_index, :, :])
                # SKIN
                skin = deepcopy(skins[current_valid_case, current_slice_index, :, :])

        # Axial view
        elif current_view=="axial" and lp["selectedView"] in ["axial", "mixed"]:
            if 1 in skins[current_valid_case, :, current_slice_index, :]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_valid_case, channel, :, current_slice_index, :]))
                # CT
                ct = deepcopy(cts[current_valid_case,:,current_slice_index,:])
                # SKIN
                skin = deepcopy(skins[current_valid_case, :, current_slice_index, :])

        # Coronal view
        elif current_view=="coronal" and lp["selectedView"] in ["coronal", "mixed"]:
            if 1 in skins[current_valid_case, :, :, current_slice_index]: # If the slice does not contain patient, skip
                # MRI
                for channel in range(len(lp["MRIchannels"])): # Loop over all the possible input MRI channels
                    mri_channels.append(deepcopy(mris[current_valid_case, channel, :, :, current_slice_index]))
                # CT
                ct = deepcopy(cts[current_valid_case, :, :, current_slice_index])
                # Skin
                skin = deepcopy(skins[current_valid_case, :, :, current_slice_index])

        # Assign images to batch
        if ct is not None and skin is not None and not len(mri_channels) == 0:
            for mri_channel_index, mri_channel in enumerate(mri_channels):
                mris_batch[batchIndex,mri_channel_index,:,:] = deepcopy(augment_image(mri_channel, current_augmentation_type, current_augmentation_magnitude, 'mri'))
            ct_batch[batchIndex,0,:,:] = deepcopy(augment_image(ct, current_augmentation_type, current_augmentation_magnitude, 'ct'))
            skin_batch[batchIndex,0,:,:] = deepcopy(augment_image(skin, current_augmentation_type, current_augmentation_magnitude, 'skin'))
            batchIndex+=1

        # If batch is full, start validing
        if batchIndex == 1: # lp["batchSize"]:

            # Convert numpy object to pytorch tensor
            ct_batch_tensor = torch.from_numpy(ct_batch).float().cuda()
            mris_batch_tensor = torch.from_numpy(mris_batch).float().cuda()
            skin_batch_tensor = torch.from_numpy(skin_batch).float().cuda()

            output = model(mris_batch_tensor)

            if lp["loss"]=="global":
                loss = loss_function(ct_batch_tensor, output)
                loss_type_str = "global"
            elif lp["loss"] in ["masked", "masked-FFT", "masked-SSIM", "weighted-masked", "alternated-MAE", "synthetic-MAE", "content"]:
                loss = loss_function(ct_batch_tensor, output, skin_batch_tensor)
                loss_type_str = "masked"

            losses_list.append(loss.item())
            print (">>> VALIDATION Epoch %d -- Train case index: %d -- Example %d/%d -- Loss (%s) = %.3f" % (epoch, current_valid_case, i, number_of_slices_to_show, loss_type_str, loss.item()))
            batchIndex=0

    losses_mean=np.mean(losses_list)

    # Print on screen and write into the report the obtained epoch statistics 
    print ("VALID - epoch time = %.3f -- mean loss (%s) = %.3f" % ((time.process_time()-epochStartTime)/60.0, loss_type_str, losses_mean))
    with open(reportPath, "a") as report:
        report.write("Validation - epoch %d ===== mean loss (%s) = %.3f\n" % (epoch, loss_type_str, losses_mean))

    return losses_mean


def load_parameters(fname):
    """
    This function loads a model from file.
    """

    model = torch.load(fname)
    model.eval()

    return model


def save_parameters(model, fname):
    """
    This function writes the model into a file.
    """

    torch.save(model, fname)


def load_cases(cases_file_name):
    """
    This function parse the text file where cases IDs are reported.
    """

    # Open and read cases file
    with open(cases_file_name) as f:
        cases_lines = f.readlines()

    # Clean and return lines
    cases = [case.strip() for case in cases_lines if not (case.strip() == "" or case.strip().startswith("#"))]
    return cases


def load_learning_parameters(par_file_name):
    """
    This function parse the text file where learning parameters are defined.
    """

    # Define the parameters dict
    LearningParameters = {
        "dataPath":"",
        "netPath":"",
        "filtersInitNum":0,
        "learningRate":0.0,
        "lambda":0.0,
        "fftWeight":1.0,
        "maeWeight":1.0,
        "ssimWeight":2000.0,
        "syntheticWeight":1.0,
        "contentWeight":1.0,
        "epochs":0,
        "batchSize":0,
        "doBatchNorm":None,
        "MRIchannels":[],
        "CT":"",
        "skin":"",
        "loss":"global",
        "continueEpoch": -1,
        "selectedView": "mixed",
        "tile": 0,
        "orientation": "SAC",
        "dropout":0.0,
        "toCrop":0}

    # Open and read parameter file
    with open(par_file_name) as f:
        par_lines = f.readlines()

    # Parse each line
    for line_number, par_line in enumerate(par_lines):

        # Skip empty lines or comment lines
        if par_line.strip() == "" or par_line.strip().startswith("#"):
            continue

        splitted_line = par_line.strip().split("=")

        if len(splitted_line) != 2: # Each line must have the format "key=value"
            print("Error in line %d...skipped" % (line_number))
            continue

        # Split key and value
        key, value = splitted_line[0].strip(), splitted_line[1].strip()

        # Cast each value into the correct type
        if key in ["epochs", "batchSize", "tile", "toCrop", "filtersInitNum"]: # Int value
            LearningParameters[key] = int(value)
        elif key in ["lambda", "dropout", "fftWeight", "ssimWeight", "maeWeight", "syntheticWeight", "contentWeight"]: # Float value
            LearningParameters[key] = float(value)
        elif key in ["loss", "selectedView", "orientation", "CT", "skin"]: # String value
            LearningParameters[key] = value
        elif key in ["dataPath", "netPath"]: # String value (path)
            LearningParameters[key] = os.path.abspath(value) + os.sep
        elif key in ["doBatchNorm"]: # Boolean value
            if value == "False": LearningParameters[key] = False
            elif value == "True": LearningParameters[key] = True
        elif key in ["learningRate"]: # List of floats
            LearningParameters[key] = [float(v) for v in value.replace("[","").replace("]","").split(",")]
        elif key in ["MRIchannels"]: # List of strings
            LearningParameters[key] = value.replace("[","").replace("]","").split(",")
        elif key in ["continueEpoch"]: # Custom
            if value == "-":
                LearningParameters[key] = -1
            else:
                LearningParameters[key] = int(value)
        else:
            print("Unknown key in line %d" % (line_number))

    return LearningParameters


# Let's start
print ("Start!")

# Get parameters from command line
#lp = load_learning_parameters(os.path.abspath(sys.argv[1]))
lp = load_learning_parameters('examplel_net1_ax/learning_parameters.txt')


# Parse text files
trainCases = load_cases(os.path.abspath(lp["netPath"] + "train.txt"))
validCases = load_cases(os.path.abspath(lp["netPath"] + "valid.txt"))

# Compute image shape for the model. "to_crop" is the amount of slices to crop along axial direction
max_image_shape = max(sitk.GetArrayFromImage(sitk.ReadImage(lp["dataPath"] + trainCases[0] + os.sep + lp["MRIchannels"][0])).shape)

# Create model
model = models1.Net1(lp).cuda()
model.apply(weight_init.weight_init)


# Figure out if training starts from scratch or not
starting_epoch = 0
best_valid_metric = np.inf

if lp["continueEpoch"] != -1:
    print ("Loading to continue from %d" % (lp["continueEpoch"]))
    model = load_parameters(lp["netPath"] + lp["selectedView"] + "_" + str(lp["continueEpoch"]) + ".pt")
    print ("Loaded! :)")
    starting_epoch = lp["continueEpoch"] + 1

# Loop over epochs
for epoch in range(starting_epoch, lp["epochs"]):
    print ("===============================================now epoch", epoch)

    # Run train
    run_training(model, trainCases, epoch, lp, max_image_shape)

    # Run validation
    current_valid_metric = run_validation(model, validCases, epoch, lp, max_image_shape)
   
    if current_valid_metric <= best_valid_metric:
        best_valid_metric = current_valid_metric
        # Save model for the current epoch
        save_parameters(model, lp["netPath"] + lp["selectedView"] + "_" + str(epoch) + ".pt")
    
