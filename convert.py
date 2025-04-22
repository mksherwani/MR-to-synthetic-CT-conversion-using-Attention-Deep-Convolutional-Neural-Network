import time, sys, os
from copy import deepcopy
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn

import models

torch.cuda.set_device(0)

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

def rotate_back_image(img, orientation):
    """
    This function swaps axes in order to return the data in the same format as the original.
    """

    if orientation == "SAC": # Sagittal, Axial, Coronal
        return img
    elif orientation == "ACS": # Axial, Coronal, Sagittal:
        mm1 = np.swapaxes(img, 2, 0)
        return np.swapaxes(mm1, 1, 0)
    else:
        print("Unknow orientation...Exit")
        exit()

def reshape_image(img, background_value=0.0):
    """
    This function reshapes the image to a cubic matrix and returns also the offsets.
    """

    img_shape = img.shape
    max_image_shape = max(img_shape)
    offsets = (int(np.floor(max_image_shape-img_shape[0])/2.0), int(np.floor(max_image_shape-img_shape[1])/2.0), int(np.floor(max_image_shape-img_shape[2])/2.0))

    reshaped_img = np.ones((max_image_shape, max_image_shape, max_image_shape), dtype=np.float32) * float(background_value)
    reshaped_img[offsets[0]:offsets[0]+img_shape[0],offsets[1]:offsets[1]+img_shape[1],offsets[2]:offsets[2]+img_shape[2]]=img[:,:,:]

    return reshaped_img.astype(np.float32), offsets


def prepare_img(img_sitk, orientation, background_value=0.0):
    """
    This function converts, roatates and reshapes (if needed) the sitk image.
    """

    img = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    img = rotate_image(img, orientation)
    original_img_shape = img.shape
    img, offsets = reshape_image(img, background_value)

    return img.astype(np.float32), original_img_shape, offsets


def convert(model, testing_case_path, lp, max_image_shape, reconstructing_view):
    """
    This function execute the conversion from MR to CT
    """

    model.eval()

    # Read the input image(s)
    mris = np.zeros((1, len(lp["MRIchannels"]), max_image_shape, max_image_shape, max_image_shape), dtype='float32')
    for channel_index, channel in enumerate(lp["MRIchannels"]):
        mri_sitk = sitk.ReadImage(testing_case_path+channel)
        mri, original_img_shape, offsets = prepare_img(mri_sitk, lp["orientation"], background_value=0)
        mris[0,channel_index,:,:,:] = mri

    # Create the batch
    mris_batch = np.zeros((1, len(lp["MRIchannels"]), max_image_shape, max_image_shape), dtype='float32')

    # Create an empty matrix for storing the converted slices
    sCT_raw = np.ones((max_image_shape, max_image_shape, max_image_shape), dtype="float32")*-1000.0

    # Loop over all the slices (according to the selected view) and convert them.
    for slice_index in range(max_image_shape):
        if reconstructing_view == "sagittal":
            mris_batch[0,:,:,:] = mris[0, :, slice_index, :, :]
            mris_batch_tensor = torch.from_numpy(mris_batch).float().cuda()
            sCT_slice = model(mris_batch_tensor).cpu().data.numpy()
            sCT_raw[slice_index,:,:] = sCT_slice[0,0,:,:]

        elif reconstructing_view == "axial":
            mris_batch[0,:,:,:] = mris[0, :, :, slice_index, :]
            mris_batch_tensor = torch.from_numpy(mris_batch).float().cuda()
            sCT_slice = model(mris_batch_tensor).cpu().data.numpy()
            sCT_raw[:,slice_index,:] = sCT_slice[0,0,:,:]

        elif reconstructing_view == "coronal":
            mris_batch[0,:,:,:] = mris[0, :, :, :, slice_index]
            mris_batch_tensor = torch.from_numpy(mris_batch).float().cuda()
            sCT_slice = model(mris_batch_tensor).cpu().data.numpy()
            sCT_raw[:,:,slice_index] = sCT_slice[0,0,:,:]

    # Crop and rotate the image to the original shape/orientation
    sCT_raw_crop = sCT_raw[offsets[0]:offsets[0]+original_img_shape[0],offsets[1]:offsets[1]+original_img_shape[1],offsets[2]:offsets[2]+original_img_shape[2]]
    sCT = rotate_back_image(sCT_raw_crop, lp["orientation"]).astype(np.float32)
    sCT_sitk = sitk.GetImageFromArray(sCT)
    sCT_sitk.CopyInformation(mri_sitk)

    return sCT_sitk


def load_parameters(model, fname):
    """
    This function loads a model from file.
    """

    model = torch.load(fname)
    model.eval()

    return model

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
        "epochs":0,
        "batchSize":0,
        "doBatchNorm":None,
        "MRIchannels":[],
        "CT":"",
        "skin":"",
        "loss":"global",
        "continueEpoch": -1,
        "selectedView": "axial",
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
        elif key in ["lambda", "dropout"]: # Float value
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

    return LearningParameters

# Let's start
print ("Start!")

# Get parameters from command line
lp = load_learning_parameters(os.path.abspath(sys.argv[1]))
selected_epoch = int(sys.argv[2])
reconstructing_view = sys.argv[3]
assert reconstructing_view in ["axial", "sagittal", "coronal"]

# Parse text files
testing_cases = load_cases(os.path.abspath(lp["netPath"] + "testing.txt"))

# Compute image shape for the network. "to_crop" is the amount of slices to crop along axial direction
max_image_shape = max(sitk.GetArrayFromImage(sitk.ReadImage(lp["dataPath"] + testing_cases[0] + os.sep + lp["MRIchannels"][0])).shape)

# Create model
model = models.Net1(lp).cuda()
model = load_parameters(model, lp["netPath"] + lp["selectedView"] + "_" + str(selected_epoch) + ".pt")

# Loop through testing cases, convert and save
for testing_case in testing_cases:
    print("Converting case %s along %s direction" % (testing_case, reconstructing_view))
    testing_case_path = lp["dataPath"] + testing_case + os.sep
    sCT_fn = os.path.abspath("%ssCT_%s_e%d.nrrd" % (testing_case_path, reconstructing_view, selected_epoch))
    sCT_sitk = convert(model, testing_case_path, lp, max_image_shape, reconstructing_view)
    sitk.WriteImage(sCT_sitk, sCT_fn)

print("Converted!")
