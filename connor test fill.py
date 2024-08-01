import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np

# blue = "/Users/connor/Downloads/temp/image_1.tif"
# green = "/Users/connor/Downloads/temp/image_2.tif"
# red = "/Users/connor/Downloads/temp/image_3.tif"
# img = plt.imread(blue)
# plt.imshow(img, cmap="Blues_r")
# plt.show()
# img = plt.imread(green)
# plt.imshow(img, cmap="Greens_r")
# plt.show()
# img = plt.imread(red)
# plt.imshow(img, cmap="Reds_r")
# plt.show()

url = "/Users/albert2/Downloads/d1_reimage/fov_3_MMStack_4-Pos016_013.ome.tif"
tif_stack = tif.imread(url)

# Show image, not needed tehe
# image = np.stack((tif_stack[2], tif_stack[1], tif_stack[0]), axis = -1)
# image = ((image - 181) / 4666.0 * 255).astype(np.uint8)
# plt.imshow(image)
# plt.show()

bluechannel = tif_stack[263][0]
# plt.imshow(bluechannel, cmap='Blues_r')
# plt.show()

buffer = 7.4
threshold = np.max(bluechannel)/buffer
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
thresholdC = 140
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

print(cells)

mask = np.copy(bluechannel)
mask += 2
for i in range(cells):
    for j in range(sizes[i]):
        mask[histories[i][j]] = 1

mask[np.where(mask != 1)] = 0

plt.imshow(mask, cmap='Blues_r')
plt.show()

outlierThreshold = np.mean(sizes) + (1.5 * np.std(sizes))

histories[np.where(histories)]