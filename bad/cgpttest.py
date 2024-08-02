import cv2
import numpy as np
import tifffile as tiff
from skimage import filters
import matplotlib.pyplot as plt

# Load the TIFF file
file_path = "/Users/albert2/Downloads/d1_reimage/fov_3_MMStack_4-Pos012_001.ome.tif"
image_stack = tiff.imread(file_path)

# Check the shape of the image stack
print("Image stack shape:", image_stack.shape)

# Extract the image at the specified index
index = 27
if index < len(image_stack):
    image = image_stack[index]
else:
    print("Index out of range.")
    exit()

# Process a single channel
def count_cells_in_channel(image, channel_index):
    channel = image[:, :, channel_index]
    blurred = cv2.GaussianBlur(channel, (5, 5), 0)
    thresh_val = filters.threshold_otsu(blurred)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)

    # Distance transform and watershed algorithm
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = np.int32(markers)  # Convert markers to int32 for watershed
    color_image = cv2.merge([channel, channel, channel])
    color_image = np.uint8(color_image)
    markers = cv2.watershed(color_image, markers)

    return len(np.unique(markers)) - 1  # Subtract one for the background marker

# Process overlapping cells between two channels
def count_overlapping_cells(image, channel1_index, channel2_index):
    channel1 = image[:, :, channel1_index]
    channel2 = image[:, :, channel2_index]
    blurred1 = cv2.GaussianBlur(channel1, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(channel2, (5, 5), 0)
    thresh_val1 = filters.threshold_otsu(blurred1)
    thresh_val2 = filters.threshold_otsu(blurred2)
    _, binary1 = cv2.threshold(blurred1, thresh_val1, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(blurred2, thresh_val2, 255, cv2.THRESH_BINARY)
    binary1 = np.uint8(binary1)
    binary2 = np.uint8(binary2)
    overlap = cv2.bitwise_and(binary1, binary2)

    # Distance transform and watershed algorithm
    dist_transform = cv2.distanceTransform(overlap, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(overlap, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = np.int32(markers)  # Convert markers to int32 for watershed
    color_image = cv2.merge([overlap, overlap, overlap])
    color_image = np.uint8(color_image)
    markers = cv2.watershed(color_image, markers)

    return len(np.unique(markers)) - 1  # Subtract one for the background marker

# Calculate counts
blue_cells = count_cells_in_channel(image, 0)
red_cells = count_cells_in_channel(image, 2)
green_cells = count_cells_in_channel(image, 1)
blue_red_overlap = count_overlapping_cells(image, 0, 2)
blue_green_overlap = count_overlapping_cells(image, 0, 1)

# Print results
print(f"Image index: {index}")
print(f"Blue cells: {blue_cells}")
print(f"Red cells: {red_cells}")
print(f"Green cells: {green_cells}")
print(f"Blue-Red overlapping cells: {blue_red_overlap}")
print(f"Blue-Green overlapping cells: {blue_green_overlap}")

# Plotting the results
labels = ['Blue Cells', 'Red Cells', 'Green Cells', 'Blue-Red Overlap', 'Blue-Green Overlap']
counts = [blue_cells, red_cells, green_cells, blue_red_overlap, blue_green_overlap]

plt.bar(labels, counts, color=['blue', 'red', 'green', 'purple', 'cyan'])
plt.xlabel('Cell Types')
plt.ylabel('Counts')
plt.title(f'Cell Counts for Image Index {index}')
plt.show()
