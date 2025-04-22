import sys
import numpy as np
import SimpleITK as sitk

# Read images
ct_sitk = sitk.ReadImage(sys.argv[1])
ct = sitk.GetArrayFromImage(ct_sitk)

pct_sitk = sitk.ReadImage(sys.argv[2])
pct = sitk.GetArrayFromImage(pct_sitk)

mask_sitk = sitk.ReadImage(sys.argv[3])
mask = sitk.GetArrayFromImage(mask_sitk)

# Compute MAE just inside the mask
diff_mae = np.abs(ct - pct)
diff_mae[mask==0]=np.nan
mae = np.nanmean(diff_mae)

# Compute BIAS just inside the mask
diff_bias = ct - pct
diff_bias[mask==0]=np.nan
bias = np.nanmean(diff_bias)

# Print the results
print("MAE = %.3f" % mae)
print("BIAS = %.3f" % bias)
