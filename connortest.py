import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np

"""
THESE VARIABLES CHANGE THE VALUES, CHANGE THESE FOR DIFFERENT IMAGES:
"""
bluebuffer = 7.4 # buffer for background, higher buffer means lower threshold for background removal (divided)
thresholdC = 140 # minimum cell size blue
thresholdMultiplier = 1.5 # threshold multipler for outlier calculation, increase to increase threshold (less cells are considered and split)
# ^ cont: 1.5 is normal calculation for stats, can change if data set is weird
greenbuffer = 2.3 # buffer for background, but for green channel instead (divided)
thresholdG = 20 # minimum cell size overlap with green
redbuffer = 200 # buffer for background, but for red channel instead (added)
minthresholdR = 15 # minimum cell size red
thresholdR = 90 # maxumim cell size red
"""
END
"""

url = "/Users/albert2/Downloads/d1_reimage/fov_3_MMStack_4-Pos012_001.ome.tif"
tif_stack = tif.imread(url)
posIndex = 27
# Show image, not needed tehe
# image = np.stack((tif_stack[2], tif_stack[1], tif_stack[0]), axis = -1)
# image = ((image - 181) / 4666.0 * 255).astype(np.uint8)
# plt.imshow(image)
# plt.show()

bluechannel = tif_stack[posIndex][0]
# plt.imshow(bluechannel, cmap='Blues_r')
# plt.show()

threshold = np.max(bluechannel)/bluebuffer
new = np.copy(bluechannel)
new[np.where(new < threshold)] = 0
# plt.imshow(new, cmap='Blues_r')
# plt.show()

dx = [-1, 1, 0, 0, 1,-1,-1, 1]
dy = [ 0, 0, 1,-1, 1, 1,-1,-1]

def fill(x, y):
    hist = [(x, y)]
    stack = [(x, y)]
    size = 0

    while stack:
        cx, cy = stack.pop()
        if new[cx][cy] == 0:
            continue
        
        new[cx][cy] = 0
        size += 1

        for i in range(8):
            nx, ny = cx + dx[i], cy + dy[i]
            if 0 <= nx < 1200 and 0 <= ny < 1200 and new[nx][ny] != 0:
                stack.append((nx, ny))
                hist.append((nx, ny))

    return size, list(set(hist))

cells = 0
px = []
py = []
sizes = []
histories = []

for i in range(1200):
    for j in range(1200):
        if new[i][j] != 0:
            size, history = fill(i, j)
            if size > thresholdC:  # Only record cells
                cells += 1
                px.append(i)
                py.append(j)
                sizes.append(size)
                histories.append(history)

# print(px)
# print(py)
# print(sizes)
# print(np.mean(sizes))

print("Found Cells:", cells)

mask = np.copy(bluechannel)
mask += 2
for i in range(cells):
    for j in range(sizes[i]):
        mask[histories[i][j]] = 1

mask[np.where(mask != 1)] = 0

plt.imshow(mask, cmap='Blues_r')
plt.show()

maskCellCount = np.zeros(cells)
maskCellCount += 1

outlierThreshold = np.mean(sizes) + (thresholdMultiplier * np.std(sizes))

avgCellSizeOurlierless = np.mean(np.array(sizes)[np.where(sizes < outlierThreshold)])
print("Average Cell Size Excluding Outliers:", avgCellSizeOurlierless)

outlierCells = np.array(sizes)[np.where(sizes > outlierThreshold)]
indexs = np.where(sizes > outlierThreshold)
# print(indexs)
for i in range(len(outlierCells)):
    cells += int(outlierCells[i]//avgCellSizeOurlierless - 1)
    maskCellCount[indexs[0][i]] += outlierCells[i]//avgCellSizeOurlierless - 1

print("Found Cells With Split:", cells)

"""
THIS IS THE END OF WORKING WITH BLUE MOSTLY, MOVING ONTO GREEN
"""

greenchannel = tif_stack[posIndex][1]

# plt.imshow(greenchannel, cmap='Greens_r')
# plt.show()

# print(np.max(greenchannel))
# print(np.min(greenchannel))
# print(np.mean(greenchannel))
# print(np.std(greenchannel))

new = np.copy(greenchannel)
# plt.imshow(new, cmap='Greens_r')
# plt.show()

threshold = np.max(greenchannel)/greenbuffer
# print(threshold)
new = np.copy(greenchannel)
new *= mask
new[np.where(new < threshold)] = 0

# plt.imshow(new, cmap='Greens_r')
# plt.show()

gocells = 0
gox = []
goy = []

for i in range(1200):
    for j in range(1200):
        if new[i][j] != 0:
            size, history = fill(i, j)
            if size > thresholdG:  # Only record cells
                gocells += 1
                gox.append(i)
                goy.append(j)
                # print(i, j)

print("Overlap With Green Channel:", gocells)

"""
THIS IS THE END OF WORKING WITH GREEN, MOVING ONTO RED
"""

redchannel = tif_stack[posIndex][2]

# plt.imshow(greenchannel, cmap='Greens_r')
# plt.show()

# print(np.max(greenchannel))
# print(np.min(greenchannel))
# print(np.mean(greenchannel))
# print(np.std(greenchannel))

new = np.copy(redchannel)
# plt.imshow(new, cmap='Reds_r')
# plt.show()

# print(np.mean(redchannel))
threshold = np.mean(redchannel) + redbuffer
# print(threshold)
new = np.copy(redchannel)
new *= mask
new[np.where(new < threshold)] = 0

plt.imshow(new, cmap='Reds_r')
plt.show()

rocells = 0
rox = []
roy = []

for i in range(1200):
    for j in range(1200):
        if new[i][j] != 0:
            size, history = fill(i, j)
            if minthresholdR < size and size < thresholdR:  # Only record cells
                rocells += 1
                rox.append(i)
                roy.append(j)
                # print(i, j)

print("Overlap With Red Channel:", rocells)