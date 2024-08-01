import os
import tifffile as tif
from PIL import Image

tiff_stack_path = 'path to tif file :O'
output_dir = 'path to folder where tif stack opened should go'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tif_stack = tif.imread(tiff_stack_path)

num_images = tif_stack.shape[0]
print(f'The stack contains {num_images} images.')

for i in range(num_images):
    image = tif_stack[i]
    output_path = os.path.join(output_dir, f'image_{i+1}.tif')
    Image.fromarray(image).save(output_path)

print(f'All {num_images} images have been saved in the folder "{output_dir}".')
