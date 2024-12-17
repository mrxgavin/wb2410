import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from Algorithms.BandDetection import detect_bands
from Algorithms.Analyze_band_intensity import analyze_band_intensity
from ImageProcessing.Processor import *



# Page config
st.set_page_config(
    page_title="Western Blot Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)



def main():
    st.title("Western Blot Band Analyzer")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Sidebar controls
    st.sidebar.header("Settings")

    # Add option to select test image or upload image
    image_source = st.sidebar.radio("Image Source", ("Test Image", "Upload Image"))

    if image_source == "Test Image":
        test_images = load_test_images()
        
        if not test_images:
            st.error("No test images found in 'pics' directory")
            return
            
        selected_image = st.sidebar.selectbox("Select Test Image", test_images)
        
        # Construct proper path with os.path.join
        testimage_path = os.path.join(script_dir, "ImageProcessing/pics", selected_image)
        image = safe_read_image(testimage_path)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
            image = safe_read_image(temp_file_path)
        else:
            image = None

    # Advanced settings expander
    advanced_settings = st.sidebar.expander("Advanced Settings", expanded=False)
    with advanced_settings:
        min_distance = st.slider(
            "Minimum Distance Between Contours",
            0, 20, 2,
            help="Larger values reduce the number of detected bands by requiring them to be further apart. Smaller values increase the number of detected bands by allowing them to be closer together."
        )
        min_area = st.slider(
            "Minimum Contour Area",
            0, 100, 50,
            help="Larger values reduce the number of detected bands by considering only larger contours. Smaller values increase the number of detected bands by considering smaller contours as well."
        )
        clip_limit = st.slider(
            "CLAHE Clip Limit",
            1.0, 4.0, 2.0,
            help="Higher values increase the contrast of the image, making bands more prominent. Lower values decrease the contrast, making the image smoother."
        )
        tile_grid_size = st.slider(
            "CLAHE Tile Grid Size",
            1, 16, 8,
            help="Larger values process larger areas of the image, making the contrast equalization effect more pronounced. Smaller values process smaller areas, making the effect more detailed."
        )

    if image is None:
        return
    
    # Process the image
    processed, standardized = process_image(image, clip_limit, tile_grid_size)
    bands = detect_bands(processed, min_distance, min_area)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(standardized, channels="BGR", use_container_width=True)
        st.caption(f"Standardized Size: {standardized.shape[1]}x{standardized.shape[0]}")

    # Draw detection results
    result = standardized.copy()
    for i, cnt in enumerate(bands):
        # Draw contours
        cv2.drawContours(result, [cnt], -1, (0,255,255), 2)
        
        # Draw bounding boxes and labels
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,255), 1)
        cv2.putText(result, f"#{i+1}", (x,y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    with col2:
        st.subheader("Detected Bands")
        st.image(result, channels="BGR", use_container_width=True)

    # Analysis results
    if len(bands) > 0:
        st.subheader("Band Analysis Results")
        analysis_results = analyze_band_intensity(processed, bands)
        
        # Create DataFrame
        df = pd.DataFrame(analysis_results)
        
        # Add relative integrated density
        max_density = df['integrated_density'].max()
        df['relative_density'] = (df['integrated_density'] / max_density * 100)
        
        # Set display columns and round to two decimal places
        display_cols = [
            'band_number', 
            'area',
            'mean_intensity',
            'integrated_density',
            'relative_density'
        ]
        
        # Round all numerical columns to two decimal places
        df = df.round(2)
        
        col1, col2 = st.columns([3, 7])
        
        with col1:
            st.dataframe(df[display_cols])
        
        with col2:
            # st.subheader("Integrated Density Distribution")
            fig = go.Figure()
            
            # Add integrated density bar chart
            fig.add_trace(go.Bar(
                x=df['band_number'],
                y=df['relative_density'],
                name='Relative Integrated Density',
                marker_color='rgb(55, 83, 109)'
            ))
            
            fig.update_layout(
                xaxis_title="Band Number",
                yaxis_title="Relative Integrated Density (%)",
                showlegend=False,  # Hide the legend
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
    
