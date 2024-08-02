import matplotlib.pyplot as plt
import tifffile as tif
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from skimage import measure

@dataclass
class ChannelParams:
    buffer: float
    min_threshold: int
    max_threshold: Optional[int] = None

@dataclass
class ImageParams:
    blue: ChannelParams
    green: ChannelParams
    red: ChannelParams
    threshold_multiplier: float
    image_shape: Tuple[int, int]
    plot: bool = False

def load_image(url: str, pos_index: int) -> np.ndarray:
    """Load the image from the given URL and position index."""
    try:
        tif_stack = tif.imread(url)
        if pos_index >= len(tif_stack):
            raise IndexError(f"pos_index {pos_index} is out of bounds for tif_stack with length {len(tif_stack)}")
        return tif_stack[pos_index]
    except FileNotFoundError:
        print(f"Error: File not found at {url}")
        raise
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        raise

def find_cells(channel: np.ndarray, min_threshold: int, max_threshold: Optional[int] = None) -> Tuple[int, List[int], List[int], List[int], List[List[Tuple[int, int]]]]:
    """Find cells in the given channel using scikit-image."""
    labeled_channel, num_cells = measure.label(channel > 0, return_num=True, connectivity=2)
    cell_props = measure.regionprops(labeled_channel)

    cells = 0
    px, py, sizes, histories = [], [], [], []

    for prop in cell_props:
        if prop.area > min_threshold and (max_threshold is None or prop.area < max_threshold):
            cells += 1
            px.append(int(prop.centroid[0]))
            py.append(int(prop.centroid[1]))
            sizes.append(prop.area)
            histories.append(prop.coords.tolist())

    return cells, px, py, sizes, histories

def create_mask(channel: np.ndarray, histories: List[List[Tuple[int, int]]]) -> np.ndarray:
    """Create a binary mask from cell histories."""
    mask = np.zeros_like(channel, dtype=bool)
    for history in histories:
        mask[tuple(zip(*history))] = True
    return mask.astype(np.uint8)

def process_channel(image: np.ndarray, params: ChannelParams, mask: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray, List[int], List[int], List[int]]:
    """Process a single channel of the image."""
    if params.buffer:
        threshold = np.max(image) / params.buffer
    else:
        threshold = np.mean(image) + params.buffer  # For red channel

    new = np.copy(image)
    if mask is not None:
        new *= mask
    new[new < threshold] = 0

    cells, px, py, sizes, histories = find_cells(new, params.min_threshold, params.max_threshold)
    
    if mask is None:
        mask = create_mask(image, histories)

    return cells, mask, px, py, sizes

def split_large_cells(sizes: List[int], threshold_multiplier: float) -> Tuple[float, int]:
    """Split large cells and calculate new cell count."""
    sizes_array = np.array(sizes)
    outlier_threshold = np.mean(sizes_array) + (threshold_multiplier * np.std(sizes_array))
    normal_sizes = sizes_array[sizes_array < outlier_threshold]
    avg_cell_size = np.mean(normal_sizes)
    
    outlier_cells = sizes_array[sizes_array > outlier_threshold]
    additional_cells = np.sum(outlier_cells // avg_cell_size - 1)
    
    return avg_cell_size, len(sizes) + int(additional_cells)

def display_labeled_mask(mask: np.ndarray, title: str, cell_centers: List[Tuple[int, int]], cmap: str = 'nipy_spectral'):
    """Display a labeled mask with centered labels."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(mask, cmap=cmap)
    
    for i, (y, x) in enumerate(cell_centers):
        ax.text(x, y, str(i+1), fontsize=8, color='white', 
                ha='center', va='center')
        
        # Optionally, add a small circle around the centroid
        circle = plt.Circle((x, y), 3, color='red', fill=False, linewidth=1)
        ax.add_patch(circle)
    
    ax.set_title(title)
    plt.show()

def main(url: str, pos_index: int, params: ImageParams):
    """Main function to process the image."""
    image = load_image(url, pos_index)

    # Process blue channel
    blue_cells, blue_mask, blue_px, blue_py, blue_sizes = process_channel(image[0], params.blue)
    print(f"Found Cells in Blue Channel: {blue_cells}")

    if params.plot:
        cell_centers = list(zip(blue_py, blue_px))
        display_labeled_mask(blue_mask, "Blue Channel Mask with Labeled Cells", cell_centers, cmap='Blues_r')

    avg_cell_size, split_cells = split_large_cells(blue_sizes, params.threshold_multiplier)
    print(f"Average Cell Size Excluding Outliers: {avg_cell_size:.2f}")
    print(f"Found Cells With Split: {split_cells}")

    # Process green channel
    green_cells, _, _, _, _ = process_channel(image[1], params.green, blue_mask)
    print(f"Overlap With Green Channel: {green_cells}")

    # Process red channel
    red_cells, red_mask, _, _, _ = process_channel(image[2], params.red, blue_mask)
    print(f"Overlap With Red Channel: {red_cells}")

    if params.plot:
        display_labeled_mask(red_mask, "Red Channel Mask", [], cmap='Reds_r')  # No cell centers for red channel

if __name__ == "__main__":
    params = ImageParams(
        blue=ChannelParams(buffer=7.4, min_threshold=140),
        green=ChannelParams(buffer=2.3, min_threshold=20),
        red=ChannelParams(buffer=200, min_threshold=15, max_threshold=90),
        threshold_multiplier=1.5,
        image_shape=(1200, 1200),
        plot=True
    )
    url = "/Users/albert2/Downloads/d1_reimage/fov_3_MMStack_4-Pos012_001.ome.tif"
    pos_index = 27
    main(url, pos_index, params)