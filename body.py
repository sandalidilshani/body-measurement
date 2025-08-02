import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Body Measurement Tool",
    page_icon="📏",
    layout="wide"
)

st.title("📏 Body Measurement Tool")
st.markdown("Upload front and side view images to get accurate body measurements")

# Initialize session state
if 'measurements_done' not in st.session_state:
    st.session_state.measurements_done = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Helper Functions
@st.cache_data
def create_silhouette(image_array):
    """Create silhouette from image array"""
    image = image_array.copy()
    h, w, _ = image.shape
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, w - 100, h - 100)
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    silhouette = np.ones_like(image, np.uint8) * 255
    silhouette[mask2 == 1] = [0, 0, 0]
    return image, silhouette, mask2

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def process_measurements(front_img, side_img, real_height_cm):
    """Process both images and return measurements"""
    results = {
        'shoulder_cm': None,
        'waist_cm': None,
        'depth_cm': None,
        'front_annotated': None,
        'side_annotated': None
    }
    
    try:
        # Process Front View
        front_cv2 = pil_to_cv2(front_img)
        front_processed, sil_front, mask_front = create_silhouette(front_cv2)
        h1, w1 = sil_front.shape[:2]
        gray_front = cv2.cvtColor(sil_front, cv2.COLOR_BGR2GRAY)
        body_mask_front = gray_front < 128
        
        # Waist measurement from front view
        waist_y = int(h1 * 0.45)
        waist_row = body_mask_front[waist_y, :]
        x_coords_front = np.where(waist_row)[0]
        
        if len(x_coords_front) > 1:
            waist_px = x_coords_front[-1] - x_coords_front[0]
            px_per_cm = h1 / real_height_cm
            results['waist_cm'] = waist_px / px_per_cm
        
        # Shoulder measurement using MediaPipe
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
        rgb_front = cv2.cvtColor(front_processed, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_front)
        
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            def get_xy(idx): return int(lm[idx].x * w1), int(lm[idx].y * h1)
            
            left_shoulder = get_xy(11)
            right_shoulder = get_xy(12)
            nose = get_xy(0)
            heels = [get_xy(29)[1], get_xy(30)[1]]
            
            pixel_height = np.mean(heels) - nose[1]
            px_per_cm = pixel_height / real_height_cm
            shoulder_px = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
            results['shoulder_cm'] = shoulder_px / px_per_cm
        
        # Process Side View
        side_cv2 = pil_to_cv2(side_img)
        side_processed, sil_side, mask_side = create_silhouette(side_cv2)
        h2, w2 = sil_side.shape[:2]
        gray_side = cv2.cvtColor(sil_side, cv2.COLOR_BGR2GRAY)
        body_mask_side = gray_side < 128
        
        # Depth measurement
        depth_y = int(h2 * 0.45)
        depth_row = body_mask_side[depth_y, :]
        x_coords_side = np.where(depth_row)[0]
        
        if len(x_coords_side) > 1:
            depth_px = x_coords_side[-1] - x_coords_side[0]
            px_per_cm = h2 / real_height_cm
            results['depth_cm'] = depth_px / px_per_cm
        
        # Create annotated images
        front_draw = front_processed.copy()
        
        # Draw waist line on front
        if len(x_coords_front) > 1:
            pt1 = (x_coords_front[0], waist_y)
            pt2 = (x_coords_front[-1], waist_y)
            cv2.line(front_draw, pt1, pt2, (0, 0, 255), 3)
            cv2.putText(front_draw, f"Waist: {results['waist_cm']:.1f} cm", 
                       (pt1[0], pt1[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw shoulder line on front
        if pose_results.pose_landmarks and results['shoulder_cm']:
            cv2.line(front_draw, left_shoulder, right_shoulder, (255, 0, 0), 3)
            mid_point = ((left_shoulder[0] + right_shoulder[0]) // 2, left_shoulder[1] - 15)
            cv2.putText(front_draw, f"Shoulder: {results['shoulder_cm']:.1f} cm", 
                       mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw depth line on side
        side_draw = side_processed.copy()
        if len(x_coords_side) > 1:
            pt3 = (x_coords_side[0], depth_y)
            pt4 = (x_coords_side[-1], depth_y)
            cv2.line(side_draw, pt3, pt4, (0, 255, 0), 3)
            cv2.putText(side_draw, f"Depth: {results['depth_cm']:.1f} cm", 
                       (pt3[0], pt3[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Convert back to PIL for display
        results['front_annotated'] = Image.fromarray(cv2.cvtColor(front_draw, cv2.COLOR_BGR2RGB))
        results['side_annotated'] = Image.fromarray(cv2.cvtColor(side_draw, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")
        return None
    
    return results

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    real_height_cm = st.number_input(
        "Person's Real Height (cm)", 
        min_value=50, 
        max_value=250, 
        value=170,
        step=1,
        help="Enter the actual height of the person in centimeters"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a front view image
    2. Upload a side view image  
    3. Set the person's real height
    4. Click 'Start Measurements'
    
    **Tips:**
    - Ensure person is standing straight
    - Good lighting and clear background
    - Person should fill most of the frame
    """)

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Front View Image")
    front_uploaded = st.file_uploader(
        "Upload front view image", 
        type=['png', 'jpg', 'jpeg'],
        key="front_upload"
    )
    
    if front_uploaded:
        front_image = Image.open(front_uploaded)
        st.image(front_image, caption="Front View", use_column_width=True)

with col2:
    st.subheader("Side View Image")
    side_uploaded = st.file_uploader(
        "Upload side view image", 
        type=['png', 'jpg', 'jpeg'],
        key="side_upload"
    )
    
    if side_uploaded:
        side_image = Image.open(side_uploaded)
        st.image(side_image, caption="Side View", use_column_width=True)

# Measurement button and processing
st.markdown("---")
col_center1, col_center2, col_center3 = st.columns([1, 1, 1])

with col_center2:
    if st.button("Start Measurements", type="primary", use_container_width=True):
        if front_uploaded and side_uploaded:
            with st.spinner("Processing images and calculating measurements..."):
                front_image = Image.open(front_uploaded)
                side_image = Image.open(side_uploaded)
                
                results = process_measurements(front_image, side_image, real_height_cm)
                
                if results:
                    st.session_state.results = results
                    st.session_state.measurements_done = True
                    st.success("✅ Measurements completed successfully!")
        else:
            st.error("Please upload both front and side view images first!")

# Display results
if st.session_state.measurements_done and st.session_state.results:
    st.markdown("---")
    st.header(" Measurement Results")
    
    # Results summary
    col_r1, col_r2, col_r3 = st.columns(3)
    
    results = st.session_state.results
    
    with col_r1:
        if results['shoulder_cm']:
            st.metric("🟦 Shoulder Width", f"{results['shoulder_cm']:.1f} cm")
        else:
            st.metric("🟦 Shoulder Width", "Not detected")
    
    with col_r2:
        if results['waist_cm']:
            st.metric("🟥 Waist Width", f"{results['waist_cm']:.1f} cm")
        else:
            st.metric("🟥 Waist Width", "Not detected")
    
    with col_r3:
        if results['depth_cm']:
            st.metric("🟩 Body Depth", f"{results['depth_cm']:.1f} cm")
        else:
            st.metric("🟩 Body Depth", "Not detected")
    
    # Annotated images
    st.subheader(" Annotated Images")
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        if results['front_annotated']:
            st.image(results['front_annotated'], caption="Front View with Measurements", use_column_width=True)
    
    with col_a2:
        if results['side_annotated']:
            st.image(results['side_annotated'], caption="Side View with Measurements", use_column_width=True)
    
    # Download results
    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
    
    with col_d2:
        # Create results text
        results_text = f"""Body Measurement Results
========================
Height Reference: {real_height_cm} cm

Measurements:
- Shoulder Width: {results['shoulder_cm']:.1f} cm if results['shoulder_cm'] else 'Not detected'
- Waist Width: {results['waist_cm']:.1f} cm if results['waist_cm'] else 'Not detected'
- Body Depth: {results['depth_cm']:.1f} cm if results['depth_cm'] else 'Not detected'

Generated by Body Measurement Tool
"""
        
        st.download_button(
            label="📄 Download Results",
            data=results_text,
            file_name="body_measurements.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("*Body Measurement Tool - Upload images and get accurate measurements using computer vision*")