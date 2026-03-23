import io
from PIL import Image
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader


def pil_to_reportlab_image(pil_img: Image.Image, max_width_mm: float, max_height_mm: float):
    """
    Converts a PIL image to a ReportLab Image object that fits within given dimensions.
    ReportLab works in points (1 mm = 2.835 points), so we convert carefully.
    """
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # Convert mm to points
    max_w_pt = max_width_mm * mm
    max_h_pt = max_height_mm * mm

    # Scale image to fit while preserving aspect ratio
    img_w, img_h = pil_img.size
    aspect = img_w / img_h

    if aspect > (max_w_pt / max_h_pt):
        # Width is the limiting dimension
        final_w = max_w_pt
        final_h = max_w_pt / aspect
    else:
        # Height is the limiting dimension
        final_h = max_h_pt
        final_w = max_h_pt * aspect

    return RLImage(buf, width=final_w, height=final_h)


def rgb_to_hex(rgb: tuple) -> str:
    """Convert (R, G, B) tuple to '#RRGGBB' hex string for ReportLab."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def generate_pdf(
    outline_img: Image.Image,
    color_map: dict,
    page_size: str = "A4",
    title: str = "Paint by Numbers",
) -> bytes:
    """
    Generates a complete paint-by-numbers PDF.

    Page 1: Full-page outline sheet with title
    Page 2: Color key — numbered swatches in a clean grid

    Args:
        outline_img : PIL Image — the black & white numbered outline
        color_map   : dict {1-based number → (R, G, B)}
        page_size   : "A4" or "LETTER"
        title       : title printed at top of sheet

    Returns:
        bytes — the complete PDF file as a byte string (ready to download)
    """

    # --- Setup ---
    size = A4 if page_size == "A4" else LETTER
    buf = io.BytesIO()

    # We'll draw pages manually using canvas for precise control
    c = rl_canvas.Canvas(buf, pagesize=size)
    page_w, page_h = size

    margin = 15 * mm

    # PAGE 1 — The outline sheet

    # Draw title at the top
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(black)
    title_y = page_h - margin - 8 * mm
    c.drawCentredString(page_w / 2, title_y, title)

    # Subtitle
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(
        page_w / 2,
        title_y - 6 * mm,
        f"{len(color_map)} colors  •  Fill each region with its numbered color"
    )

    # Draw a thin separator line
    c.setStrokeColorRGB(0.8, 0.8, 0.8)
    c.setLineWidth(0.5)
    sep_y = title_y - 10 * mm
    c.line(margin, sep_y, page_w - margin, sep_y)

    # Draw the outline image below the title
    available_w_mm = (page_w - 2 * margin) / mm
    available_h_mm = (sep_y - margin) / mm  # space below separator to bottom margin

    # Convert PIL image to bytes and draw it
    img_buf = io.BytesIO()
    outline_img.save(img_buf, format="PNG")
    img_buf.seek(0)

    img_w, img_h = outline_img.size
    aspect = img_w / img_h

    max_w = page_w - 2 * margin
    max_h = sep_y - margin

    if aspect > (max_w / max_h):
        draw_w = max_w
        draw_h = max_w / aspect
    else:
        draw_h = max_h
        draw_w = max_h * aspect

    # Center horizontally, align to top of available space
    x = (page_w - draw_w) / 2
    y = sep_y - draw_h  # ReportLab y=0 is bottom-left

    c.drawImage(
    ImageReader(img_buf), x, y,
    width=draw_w, height=draw_h,
    preserveAspectRatio=True
    )

    # PAGE 2 — Color Key
    c.showPage()

    # Page 2 title
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(black)
    c.drawCentredString(page_w / 2, page_h - margin - 8 * mm, "Color Key")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawCentredString(
        page_w / 2,
        page_h - margin - 15 * mm,
        "Match each number to its color when painting"
    )

    # Grid layout
    cols = 4
    swatch_w = 30 * mm
    swatch_h = 16 * mm
    h_gap = 10 * mm
    v_gap = 12 * mm   # extra vertical gap to fit hex label below

    grid_w = cols * swatch_w + (cols - 1) * h_gap
    start_x = (page_w - grid_w) / 2
    start_y = page_h - margin - 28 * mm

    sorted_colors = sorted(color_map.items())

    for i, (num, rgb) in enumerate(sorted_colors):
        col = i % cols
        row = i // cols

        x = start_x + col * (swatch_w + h_gap)
        y = start_y - row * (swatch_h + v_gap)

        # ── Draw filled swatch ──────────────────────────────
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        c.setFillColorRGB(r, g, b)
        c.setStrokeColorRGB(0, 0, 0)
        c.setLineWidth(1)
        c.rect(x, y - swatch_h, swatch_w, swatch_h, fill=1, stroke=1)

        # ── Number centered inside swatch ───────────────────
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if brightness < 140:
            c.setFillColorRGB(1, 1, 1)   # white text on dark swatch
        else:
            c.setFillColorRGB(0, 0, 0)   # black text on light swatch
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(x + swatch_w / 2, y - swatch_h + 4 * mm, str(num))

        # ── Hex code below swatch ───────────────────────────
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.setFont("Helvetica", 7)
        hex_str = "#{:02X}{:02X}{:02X}".format(*rgb)
        c.drawCentredString(x + swatch_w / 2, y - swatch_h - 4 * mm, hex_str)

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(0.6, 0.6, 0.6)
    c.drawCentredString(page_w / 2, margin / 2, "Generated with Paint by Numbers Maker")

    # Finalize
    c.save()
    buf.seek(0)
    return buf.read()