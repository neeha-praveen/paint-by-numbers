# segment.py
import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage


def generate_paint_sheet(
    label_grid: np.ndarray,
    color_map: dict,
    min_region_size: int = 150,
    outline_thickness: int = 2,
) -> tuple:

    height, width = label_grid.shape
    outline_img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(outline_img)
    region_data = []

    # ── Occupancy grid for number placement ──────────────────────
    # Each cell covers GRID_CELL x GRID_CELL pixels
    # When a number is placed, we mark surrounding cells as occupied
    GRID_CELL = 14  # pixels per grid cell — tune this if needed
    grid_cols = width // GRID_CELL + 1
    grid_rows = height // GRID_CELL + 1
    occupied = np.zeros((grid_rows, grid_cols), dtype=bool)

    def is_position_free(cx, cy, num_text):
        """Check if placing a number at (cx, cy) would overlap existing numbers."""
        # Estimate how many grid cells this number occupies
        char_w = len(num_text) * GRID_CELL // 2
        char_h = GRID_CELL

        gx1 = max(0, (cx - char_w) // GRID_CELL)
        gx2 = min(grid_cols - 1, (cx + char_w) // GRID_CELL)
        gy1 = max(0, (cy - char_h) // GRID_CELL)
        gy2 = min(grid_rows - 1, (cy + char_h) // GRID_CELL)

        return not np.any(occupied[gy1:gy2+1, gx1:gx2+1])

    def mark_position(cx, cy, num_text):
        """Mark grid cells around (cx, cy) as occupied."""
        char_w = len(num_text) * GRID_CELL // 2
        char_h = GRID_CELL

        # Add padding so numbers breathe
        pad = GRID_CELL // 2
        gx1 = max(0, (cx - char_w - pad) // GRID_CELL)
        gx2 = min(grid_cols - 1, (cx + char_w + pad) // GRID_CELL)
        gy1 = max(0, (cy - char_h - pad) // GRID_CELL)
        gy2 = min(grid_rows - 1, (cy + char_h + pad) // GRID_CELL)

        occupied[gy1:gy2+1, gx1:gx2+1] = True

    def find_best_position(region_mask, cx, cy, num_text):
        """
        Try to find a free position for the number inside the region.
        Strategy:
          1. Try center of mass first
          2. If occupied, spiral outward in small steps
          3. At each candidate point, verify it's inside the region mask
          4. Return the first free-and-inside point found
          5. Fall back to center of mass if nothing found
        """
        if is_position_free(cx, cy, num_text) and region_mask[cy, cx]:
            return cx, cy

        # Spiral search outward from center
        # Steps increase gradually so we stay as close to center as possible
        step = GRID_CELL // 2
        max_radius = min(width, height) // 4

        for radius in range(step, max_radius, step):
            # Sample points on a circle of this radius
            num_samples = max(8, int(2 * np.pi * radius / step))
            for j in range(num_samples):
                angle = 2 * np.pi * j / num_samples
                tx = int(cx + radius * np.cos(angle))
                ty = int(cy + radius * np.sin(angle))

                # Must be inside image bounds
                if not (0 <= tx < width and 0 <= ty < height):
                    continue

                # Must be inside this region
                if not region_mask[ty, tx]:
                    continue

                # Must not overlap existing numbers
                if is_position_free(tx, ty, num_text):
                    return tx, ty

        # Nothing found — fall back to original center (overlap is unavoidable)
        return cx, cy

    # ── Sort regions largest-first ────────────────────────────────
    # Place numbers in large regions first so they get priority
    # Small regions get whatever space is left
    all_regions = []
    unique_labels = np.unique(label_grid)

    for label_idx in unique_labels:
        color_number = label_idx + 1
        mask = (label_grid == label_idx).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        labeled_array, num_features = ndimage.label(mask)

        for region_id in range(1, num_features + 1):
            region_mask = (labeled_array == region_id).astype(np.uint8)
            region_size = int(np.sum(region_mask))
            if region_size < min_region_size:
                continue

            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )

            cy_f, cx_f = ndimage.center_of_mass(region_mask)
            cx_i, cy_i = int(cx_f), int(cy_f)
            cx_i = max(2, min(cx_i, width - 2))
            cy_i = max(2, min(cy_i, height - 2))

            if not region_mask[cy_i, cx_i]:
                ys, xs = np.where(region_mask)
                distances = (xs - cx_i) ** 2 + (ys - cy_i) ** 2
                closest = np.argmin(distances)
                cx_i, cy_i = int(xs[closest]), int(ys[closest])

            all_regions.append({
                "color_number": color_number,
                "region_mask": region_mask,
                "contours": contours,
                "cx": cx_i,
                "cy": cy_i,
                "size": region_size,
            })

    # Sort largest regions first for priority placement
    all_regions.sort(key=lambda r: r["size"], reverse=True)

    # ── Draw outlines first (all regions) ────────────────────────
    for region in all_regions:
        for contour in region["contours"]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            smoothed = cv2.approxPolyDP(contour, epsilon, True)
            pts = smoothed.reshape(-1, 2)
            if len(pts) < 3:
                continue
            pts_list = [(int(p[0]), int(p[1])) for p in pts]
            draw.polygon(pts_list, fill=(255, 255, 255), outline=None)
            draw.line(pts_list + [pts_list[0]], fill=(0, 0, 0), width=outline_thickness)

    # ── Place numbers (largest first, with overlap avoidance) ────
    for region in all_regions:
        num_text = str(region["color_number"])
        cx, cy = find_best_position(
            region["region_mask"], region["cx"], region["cy"], num_text
        )

        # Adaptive font size based on region area
        font_size = int(0.45 * (region["size"] ** 0.38))
        font_size = max(9, min(font_size, 20))

        text_x = cx - len(num_text) * 3
        text_y = cy - font_size // 2

        # White halo for readability
        for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1),(-1,0),(1,0),(0,-1),(0,1)]:
            draw.text((text_x + dx, text_y + dy), num_text, fill=(255, 255, 255))
        draw.text((text_x, text_y), num_text, fill=(30, 30, 30))

        mark_position(cx, cy, num_text)

        region_data.append({
            "color_number": region["color_number"],
            "size": region["size"],
            "center": (cx, cy),
        })

    colorkey_img = _generate_color_key(color_map)
    return outline_img, colorkey_img, region_data


def _generate_color_key(color_map: dict) -> Image.Image:
    """Clean color key with filled swatches and contrast-aware text."""
    n = len(color_map)
    swatch_w, swatch_h = 80, 40
    label_w = 24
    padding = 10
    cols = 4
    rows = (n + cols - 1) // cols

    key_width = cols * (swatch_w + label_w + padding) + padding
    key_height = rows * (swatch_h + padding) + padding

    key_img = Image.new("RGB", (key_width, key_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(key_img)

    for i, (num, rgb) in enumerate(sorted(color_map.items())):
        col = i % cols
        row = i // cols
        x = padding + col * (swatch_w + label_w + padding)
        y = padding + row * (swatch_h + padding)

        draw.rectangle([x, y, x + swatch_w, y + swatch_h], fill=rgb, outline=(0, 0, 0))
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (255, 255, 255) if brightness < 140 else (0, 0, 0)
        draw.text((x + swatch_w // 2 - 5, y + swatch_h // 2 - 7), str(num), fill=text_color)

    return key_img