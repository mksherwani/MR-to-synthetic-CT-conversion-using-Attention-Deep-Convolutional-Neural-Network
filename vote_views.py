import sys

import numpy as np
import SimpleITK as sitk

# Figure out file names
v1_fn=sys.argv[1]
v2_fn=sys.argv[2]
v3_fn=sys.argv[3]
skin_fn=sys.argv[4]
out_fn=sys.argv[5]

# Set background value
background = -1000.0

# Read images and convert into numpy format
v1_sitk = sitk.ReadImage(v1_fn)
v2_sitk = sitk.ReadImage(v2_fn)
v3_sitk = sitk.ReadImage(v3_fn)
skin_sitk = sitk.ReadImage(skin_fn)

v1 = sitk.GetArrayFromImage(v1_sitk)
v2 = sitk.GetArrayFromImage(v2_sitk)
v3 = sitk.GetArrayFromImage(v3_sitk)
skin = sitk.GetArrayFromImage(skin_sitk)

# Vote views and mask
matrix4d_for_median = np.stack((v1, v2, v3))
out_median = np.median(matrix4d_for_median, axis=0)
out_median[skin==0] = background

# Write file
out_median_sitk=sitk.GetImageFromArray(out_median)
out_median_sitk.CopyInformation(v1_sitk)
sitk.WriteImage(out_median_sitk, out_fn)
