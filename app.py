import streamlit as st
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import os

st.set_page_config(page_title="Lane Detection", layout="centered")

# -------- Lane Detection Logic --------
def video_processing(input_path, output_path):
    def interested_region(img, vertices):
        mask = np.zeros_like(img)
        mask_color_ignore = (255,) * img.shape[2] if len(img.shape) > 2 else 255
        cv2.fillPoly(mask, vertices, mask_color_ignore)
        return cv2.bitwise_and(img, mask)

    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            lines_drawn(line_img, lines)
        return line_img

    def lines_drawn(img, lines, color=[255, 0, 0], thickness=6):
        left_fit, right_fit = [], []
        img_height = img.shape[0]

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope, intercept = _calculate_slope_intercept(x1, y1, x2, y2)
                if slope is None:
                    continue
                (left_fit if slope < -0.5 else right_fit if slope > 0.5 else []).append((slope, intercept))

        for fit in [left_fit, right_fit]:
            if fit:
                avg = np.average(fit, axis=0)
                coords = make_coordinates(img_height, avg, 0.6)
                if coords:
                    cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)

    def _calculate_slope_intercept(x1, y1, x2, y2):
        if x1 == x2:
            return None, None
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def make_coordinates(img_height, line_parameters, y1_scale):
        slope, intercept = line_parameters
        y1 = img_height
        y2 = int(img_height * y1_scale)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]

    def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def process_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        mask_yellow = cv2.inRange(img_hsv, np.array([20, 100, 100], dtype="uint8"), np.array([30, 255, 255], dtype="uint8"))
        mask_white = cv2.inRange(gray_image, 200, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

        gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)
        canny_edges = cv2.Canny(gauss_gray, 50, 150)

        imshape = image.shape
        vertices = [np.array([
            [imshape[1] / 9, imshape[0]],
            [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10],
            [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10],
            [imshape[1] - imshape[1] / 9, imshape[0]]
        ], dtype=np.int32)]

        roi_image = interested_region(canny_edges, vertices)
        theta = np.pi / 180
        line_image = hough_lines(roi_image, 2, theta, 15, 40, 20)
        return weighted_img(line_image, image)

    clip1 = VideoFileClip(input_path)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_path, audio=False, verbose=False, logger=None)

# --------- Streamlit UI ---------
st.title("🚗 Road Lane-Line Detection")

# Use default demo video
default_video_path = "road_video.mp4"
output_path = "processed_demo_video.mp4"

# Only process if not already processed
if not os.path.exists(output_path):
    st.info("Processing default demo video...")
    video_processing(default_video_path, output_path)
    st.success("Processing complete!")



col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Video**")
    st.video(default_video_path)

with col2:
    st.markdown("**Processed Video**")
    st.video(output_path)

# Optional user upload section
st.markdown("---")
st.subheader("📁 Or Upload Your Own Video")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    input_user_path = f"input_{uploaded_file.name}"
    output_user_path = f"output_{uploaded_file.name}"
    with open(input_user_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info("Processing your uploaded video...")
    video_processing(input_user_path, output_user_path)
    st.success("Processing complete!")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Your Original Video**")
        st.video(input_user_path)

    with col4:
        st.markdown("**Your Processed Video**")
        st.video(output_user_path)
