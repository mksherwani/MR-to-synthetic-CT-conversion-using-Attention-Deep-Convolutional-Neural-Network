import sys
import numpy as np
import SimpleITK as sitk

def bin_img(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img[img>0.5]=1
    img[img<=0.5]=0

    return img.astype(np.uint8)

# Read images
ct_sitk = sitk.ReadImage(sys.argv[1])
ct = sitk.GetArrayFromImage(ct_sitk)

mask_sitk = sitk.ReadImage(sys.argv[2])
mask = bin_img(sitk.GetArrayFromImage(mask_sitk))

output_fn = sys.argv[3]

# Mask Image
ct[mask==0]=-1000

# Write image
output_sitk = sitk.GetImageFromArray(ct)
output_sitk.CopyInformation(ct_sitk)
sitk.WriteImage(output_sitk, output_fn)
