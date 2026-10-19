import os
import zipfile
from io import BytesIO

import numpy as np
import streamlit as st

import inference

# ------------------------------------------------------
# 1. Global Page Configuration
# ------------------------------------------------------
st.set_page_config(
    page_title="Building Footprint Extractor",
    page_icon="🏗️",
    layout="wide"
)

# ------------------------------------------------------
# 2. Model Loading (Using st.cache_resource)
# ------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model (first run downloads it)...")
def load_model():
    return inference.load_model()

model = load_model()

# ------------------------------------------------------
# 3. Session State Setup
# ------------------------------------------------------
# Each result holds the image, its building probabilities and any georeferencing.
if "results" not in st.session_state:
    st.session_state.results = []

# ------------------------------------------------------
# 4. Sidebar Configuration
# ------------------------------------------------------
st.sidebar.header("Configuration Options")

uploaded_files = st.sidebar.file_uploader(
    "Choose image(s)",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
    help="RGB images. Georeferenced GeoTIFFs keep their location in the downloads."
)

threshold_value = st.sidebar.slider(
    "Building probability threshold",
    0.05, 0.95, 0.20, 0.05,
    help=("Pixels at or above this probability are marked as buildings. Lower it to "
          "catch more buildings, raise it to keep only the most certain ones.")
)
overlay_option = st.sidebar.checkbox("Create overlay imagery", value=True)

# ------------------------------------------------------
# 5. App Title and Description (Main Area)
# ------------------------------------------------------
st.title("Building Footprint Extractor")
st.write("Upload Sentinel-2 imagery to extract building footprints.")

st.markdown(
    """
    **Instructions:**

    1. **Upload Imagery:** Use the file uploader in the sidebar to choose your image(s).
       RGB images at about 10 m per pixel work best, such as a Sentinel-2 true-colour (TCI) GeoTIFF.
    2. **Run Processing:** Click the **Run Segmentation** button to process the uploaded images.
    3. **Adjust Results:** Use the sidebar slider to change the building probability threshold.
       The results update without re-running the model.
    4. **Download Results:** Choose the outputs you want and click **Download ZIP**.
    """
)
st.caption("At 10 m per pixel a house covers only a pixel or two, so treat the results as a guide to "
           "where buildings are rather than exact outlines. A rebuilt version of this tool is in progress.")

# ------------------------------------------------------
# 6. Step 1: Run Segmentation
# ------------------------------------------------------
if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded successfully!**")

    if st.button("Run Segmentation", type="primary"):
        st.session_state.results = []
        progress_bar = st.progress(0.0, text="Running segmentation...")
        total_images = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            def report(fraction, idx=idx, filename=uploaded_file.name):
                progress_bar.progress((idx + fraction) / total_images,
                                      text=f"Running segmentation on {filename}...")

            try:
                rgb, georef, note = inference.read_image(uploaded_file.getvalue(), uploaded_file.name)
            except inference.ImageTooLargeError as error:
                st.error(f"{uploaded_file.name}: {error}")
                continue
            except Exception:
                st.error(f"Could not read {uploaded_file.name}. Please upload an RGB image.")
                continue

            # Keep output names unique when two uploads share a name, like scene.tif and scene.png.
            base = os.path.splitext(uploaded_file.name)[0]
            name, suffix = base, 2
            while any(result["name"] == name for result in st.session_state.results):
                name, suffix = f"{base}_{suffix}", suffix + 1

            scores = inference.predict_scores(model, rgb, progress=report)
            st.session_state.results.append({
                "name": name,
                "rgb": rgb,
                "prob": inference.calibrate(scores),
                "georef": georef,
                "note": note,
            })

        progress_bar.empty()
        if st.session_state.results:
            st.success("Segmentation complete!")

# ------------------------------------------------------
# 7. Step 2: Results (Interactive Refinement)
# ------------------------------------------------------
def to_8bit(prob):
    return np.round(prob * 255).astype(np.uint8)

def create_zip(results, choices, threshold):
    """Encode the selected outputs for every image into one ZIP archive."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            name, rgb, prob, georef = result["name"], result["rgb"], result["prob"], result["georef"]
            mask = (prob >= threshold).astype(np.uint8) * 255
            if "Masks (PNG)" in choices:
                zf.writestr(f"mask_{name}.png", inference.png_bytes(mask))
            if "Probability maps (PNG)" in choices:
                zf.writestr(f"probability_{name}.png", inference.png_bytes(to_8bit(prob)))
            if "Overlays (PNG)" in choices:
                zf.writestr(f"overlay_{name}.png", inference.png_bytes(inference.overlay(rgb, mask)))
            if georef and "Masks (GeoTIFF)" in choices:
                zf.writestr(f"mask_{name}.tif", inference.geotiff_bytes(mask, georef))
            if georef and "Probability maps (GeoTIFF)" in choices:
                zf.writestr(f"probability_{name}.tif", inference.geotiff_bytes(to_8bit(prob), georef))
    return zip_buffer.getvalue()

if st.session_state.results:
    st.markdown("### Results")

    for result in st.session_state.results:
        name, rgb, prob, georef = result["name"], result["rgb"], result["prob"], result["georef"]
        mask = (prob >= threshold_value).astype(np.uint8) * 255

        st.markdown(f"#### {name}")
        if result["note"]:
            st.info(result["note"])

        columns = st.columns(3 if overlay_option else 2)
        columns[0].image(rgb, caption="Original", width="stretch")
        columns[1].image(mask, caption=f"Buildings (probability ≥ {threshold_value:.2f})", width="stretch")
        if overlay_option:
            columns[2].image(inference.overlay(rgb, mask), caption="Overlay", width="stretch")

        area_m2 = inference.building_area_m2(prob, georef)
        summary = f"{np.mean(mask > 0):.1%} of pixels are marked as buildings."
        if area_m2 is not None:
            summary += f" Rough estimate of the area covered by buildings: {area_m2 / 1e6:,.2f} km²."
        st.caption(summary)

    # ------------------------------------------------------
    # 8. Step 3: Download Options (ZIP File Download)
    # ------------------------------------------------------
    st.markdown("### Download Options")
    options = ["Masks (PNG)", "Probability maps (PNG)", "Overlays (PNG)"]
    if any(result["georef"] for result in st.session_state.results):
        options += ["Masks (GeoTIFF)", "Probability maps (GeoTIFF)"]
    download_choices = st.multiselect(
        "Select what you would like to download:",
        options,
        default=options[:1],
        help="Probability maps store each pixel's building probability scaled to 0-255."
    )

    # The archive is built only when the button is clicked, with the current settings.
    results, threshold = st.session_state.results, threshold_value
    st.download_button(
        label="Download ZIP",
        data=lambda: create_zip(results, download_choices, threshold),
        file_name="building_footprints.zip",
        mime="application/zip",
        disabled=not download_choices,
        on_click="ignore"
    )
