import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
# 16 11
# 17 12
# 4 from right
# 17 from left
# 11 * 20 + 4 - 1
# Load the image
url = "/Users/albert2/Downloads/d1_reimage/fov_3_MMStack_4-Pos016_013.ome.tif"
tif_stack = tif.imread(url)

# Get the blue channel
bluechannel = tif_stack[263][0]
plt.imshow(bluechannel, cmap='Blues_r')
plt.show()

# Apply thresholding
buffer = 7.4
threshold = np.max(bluechannel) / buffer
binary = bluechannel > threshold

# Calculate the distance transform
distance = ndi.distance_transform_edt(binary)

# Find local maxima
local_maxi = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)
local_maxi_mask = np.zeros(distance.shape, dtype=bool)
local_maxi_mask[tuple(local_maxi.T)] = True

# Create markers for watershed
markers, _ = ndi.label(local_maxi_mask)

# Apply watershed
labels = watershed(-distance, markers, mask=binary)

# Count cells and get their sizes
unique, counts = np.unique(labels, return_counts=True)
cells = len(unique) - 1  # Subtract 1 to exclude background
sizes = counts[1:]  # Exclude background size

print(f"Number of cells before removing outliers: {cells}")
print(f"Average cell size before removing outliers: {np.mean(sizes):.2f}")

# Calculate outlier threshold
outlier_threshold = np.mean(sizes) + (1.5 * np.std(sizes))
print(f"Outlier threshold: {outlier_threshold:.2f}")

# Identify outliers
outliers = np.where(sizes < outlier_threshold)[0] + 1  # Add 1 because background is labeled 0

# Remove outliers
labels_without_outliers = labels.copy()
for outlier in outliers:
    labels_without_outliers[labels == outlier] = 0

# Count cells and get their sizes after removing outliers
unique_without_outliers, counts_without_outliers = np.unique(labels_without_outliers, return_counts=True)
cells_without_outliers = len(unique_without_outliers) - 1  # Subtract 1 to exclude background
sizes_without_outliers = counts_without_outliers[1:]  # Exclude background size

print(f"Number of cells after removing outliers: {cells_without_outliers}")
print(f"Average cell size after removing outliers: {np.mean(sizes_without_outliers):.2f}")

# Create masks for visualization
mask_before = (labels > 0).astype(int)
mask_after = (labels_without_outliers > 0).astype(int)

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(mask_before, cmap='Blues_r')
ax1.set_title('Segmented Cells (Before)')

ax2.imshow(mask_after, cmap='Blues_r')
ax2.set_title('Segmented Cells (After)')

plt.tight_layout()
plt.show()

# Save the masks
tif.imwrite('mask_before_outlier_removal.tif', mask_before.astype(np.uint8) * 255)
tif.imwrite('mask_after_outlier_removal.tif', mask_after.astype(np.uint8) * 255)