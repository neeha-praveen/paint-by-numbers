import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")  # suppress sklearn convergence warnings


def quantize_image(pil_image: Image.Image, n_colors: int = 15) -> tuple:
    """
    Reduce an image to n_colors using K-Means clustering.

    Args:
        pil_image : a PIL Image object (the uploaded image)
        n_colors  : how many colors to reduce to (default 15)

    Returns:
        quantized_image : PIL Image with only n_colors unique colors
        palette         : list of (R, G, B) tuples — the chosen colors
        color_map       : dict mapping color_index (1-based) → (R, G, B)
    """

    # --- Step 1: Ensure image is in RGB mode ---
    # Some images are RGBA (with transparency) or grayscale.
    # We convert everything to RGB so every pixel = 3 numbers (R, G, B).
    image = pil_image.convert("RGB")

    # --- Step 2: Resize for faster processing ---
    MAX_DIM = 400
    w, h = image.size
    scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)  # never upscale
    small = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # --- Step 3: Reshape image into a 2D array of pixels ---
    # image shape: (height, width, 3)
    # We flatten to: (height*width, 3) — a list of [R, G, B] rows
    small_array = np.array(small, dtype=np.float32)
    pixels = small_array.reshape(-1, 3)  # shape: (num_pixels, 3)

    # --- Step 4: Run K-Means ---
    # n_init=10 means it tries 10 random starting points, picks the best
    # This avoids getting stuck in a bad local minimum
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # cluster_centers_ is an array of shape (n_colors, 3)
    # Each row is the average R, G, B of that cluster = our palette colors
    palette = [tuple(map(int, center)) for center in kmeans.cluster_centers_]

    # --- Step 5: Build color_map (1-based index → RGB) ---
    # We use 1-based because the paint sheet will show numbers 1, 2, 3...
    color_map = {i + 1: color for i, color in enumerate(palette)}

    # --- Step 6: Apply palette to the FULL-SIZE image ---
    # Now we recolor every pixel in the original full image
    # by finding its nearest palette color
    full_array = np.array(image, dtype=np.float32)
    full_pixels = full_array.reshape(-1, 3)

    # For each pixel, compute distance to every palette color
    # Then assign the index of the closest one
    palette_array = np.array(palette, dtype=np.float32)  # shape: (n_colors, 3)

    # Broadcasting trick:
    # full_pixels[:, None, :] shape: (N, 1, 3)
    # palette_array[None, :, :] shape: (1, K, 3)
    # difference shape: (N, K, 3) — distance from each pixel to each color
    diff = full_pixels[:, None, :] - palette_array[None, :, :]
    distances = np.sum(diff ** 2, axis=2)  # (N, K) — squared Euclidean distance
    labels = np.argmin(distances, axis=1)  # (N,) — index of closest color

    # --- Step 7: Reconstruct the quantized image ---
    quantized_pixels = palette_array[labels]  # replace each pixel with its palette color
    quantized_array = quantized_pixels.reshape(full_array.shape).astype(np.uint8)
    quantized_image = Image.fromarray(quantized_array, mode="RGB")

    # Also return the label grid (same shape as image, each value = color index 0-based)
    label_grid = labels.reshape(full_array.shape[:2])  # shape: (height, width)

    return quantized_image, palette, color_map, label_grid