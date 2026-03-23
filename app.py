# app.py
import streamlit as st
from PIL import Image

from quantize import quantize_image
from segment import generate_paint_sheet
from pdf_gen import generate_pdf

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Paint by Numbers Maker",
    page_icon="🎨",
    layout="wide"
)

# ── Limit image size (fix giant preview) ───────────────────────
st.markdown("""
<style>
img {
    max-height: 400px;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)

# ── Session state flag ─────────────────────────────────────────
if "generated" not in st.session_state:
    st.session_state["generated"] = False

# ── Title ─────────────────────────────────────────────────────
st.title("🎨 Paint by Numbers Maker")
st.markdown("Upload any image and get a printable paint-by-numbers PDF!")
st.divider()

# ── Sidebar — controls ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    n_colors = st.slider("Number of colors", 5, 25, 12)
    min_region = st.slider("Minimum region size", 50, 500, 150, step=25)
    page_size = st.radio("PDF page size", ["A4", "LETTER"])
    title_input = st.text_input("Sheet title", "My Paint by Numbers")

    st.divider()
    st.markdown("**💡 Tips**")
    st.markdown("""
    - Start with **12 colors** for a good balance
    - Portraits work best with **15–20 colors**
    - Landscapes work well with **10–15 colors**
    - If sheet looks too busy, increase min region size
    """)

# ── MAIN AREA ──────────────────────────────────────────────────

# ================= BEFORE GENERATION =================
if not st.session_state["generated"]:

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started!")

    else:
        pil_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(pil_image, width=500)  # 👈 smaller preview
        st.caption(f"Size: {pil_image.size[0]} × {pil_image.size[1]} px")

        st.divider()

        if st.button("🚀 Generate Paint by Numbers", type="primary", use_container_width=True):

            with st.spinner("Step 1/3 — Analyzing and reducing colors..."):
                quantized, palette, color_map, label_grid = quantize_image(
                    pil_image, n_colors=n_colors
                )

            with st.spinner("Step 2/3 — Finding regions and drawing outlines..."):
                outline, colorkey, regions = generate_paint_sheet(
                    label_grid,
                    color_map,
                    min_region_size=min_region
                )

            with st.spinner("Step 3/3 — Building your PDF..."):
                pdf_bytes = generate_pdf(
                    outline,
                    color_map,
                    page_size=page_size,
                    title=title_input
                )

            # store results
            st.session_state["quantized"] = quantized
            st.session_state["outline"] = outline
            st.session_state["colorkey"] = colorkey
            st.session_state["regions"] = regions
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["color_map"] = color_map

            # 🔥 switch UI state
            st.session_state["generated"] = True
            st.rerun()

# ================= AFTER GENERATION =================
else:
    st.success("✅ Done! Your paint-by-numbers sheet is ready.")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🖊️ Paint Sheet Preview")
        st.image(st.session_state["outline"], width=450)

    with col_right:
        st.subheader("🎨 Quantized Preview")
        st.image(st.session_state["quantized"], width=450)
        st.caption("This is what your painting should look like when finished")

    # Color key
    st.divider()
    st.subheader("🔑 Color Key Preview")
    st.image(st.session_state["colorkey"], width=500)

    # Stats
    st.divider()
    regions = st.session_state["regions"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total regions", len(regions))
    col2.metric("Colors used", len(st.session_state["color_map"]))  # ✅ FIXED
    avg_size = sum(r["size"] for r in regions) // max(len(regions), 1)
    col3.metric("Avg region size", f"{avg_size} px")

    # Download
    st.divider()
    st.download_button(
        label="⬇️ Download PDF",
        data=st.session_state["pdf_bytes"],
        file_name=f"{title_input.replace(' ', '_')}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    # Optional: reset button
    if st.button("🔁 Create New", use_container_width=True):
        st.session_state.clear()
        st.rerun()