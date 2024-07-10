import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def video_processing(input_path, output_path):
    def interested_region(img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            mask_color_ignore = (255,) * img.shape[2]
        else:
            mask_color_ignore = 255

        cv2.fillPoly(mask, vertices, mask_color_ignore)
        return cv2.bitwise_and(img, mask)

    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            lines_drawn(line_img, lines)
        return line_img

    def lines_drawn(img, lines, color=[255, 0, 0], thickness=6):
        left_lines = []
        right_lines = []

        img_height, img_width = img.shape[:2]
        left_fit = []
        right_fit = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope, intercept = _calculate_slope_intercept(x1, y1, x2, y2)

                if slope is None or intercept is None:
                    continue

                if slope < -0.5:
                    left_fit.append((slope, intercept))
                elif slope > 0.5:
                    right_fit.append((slope, intercept))

        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_lines = make_coordinates(img_height, left_fit_average, 0.6)  # Adjust the scale factor as needed

        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_lines = make_coordinates(img_height, right_fit_average, 0.6)  # Adjust the scale factor as needed

        if left_lines:
            cv2.line(img, (left_lines[0], left_lines[1]), (left_lines[2], left_lines[3]), color, thickness)
        if right_lines:
            cv2.line(img, (right_lines[0], right_lines[1]), (right_lines[2], right_lines[3]), color, thickness)

    def _calculate_slope_intercept(x1, y1, x2, y2):
        if x1 == x2:
            return None, x1
    
        try:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
        except ZeroDivisionError:
            return 0, y1
    
        return slope, intercept

    def make_coordinates(img_height, line_parameters, y1_scale):
        slope, intercept = line_parameters
        if slope is None or intercept is None:
            return []
    
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

        lower_yellow = np.array([20, 100, 100], dtype="uint8")
        upper_yellow = np.array([30, 255, 255], dtype="uint8")

        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(gray_image, 200, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

        gauss_gray = cv2.GaussianBlur(mask_yw_image, (5, 5), 0)

        canny_edges = cv2.Canny(gauss_gray, 50, 150)

        imshape = image.shape
        lower_left = [imshape[1] / 9, imshape[0]]
        lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
        top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
        top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        roi_image = interested_region(canny_edges, vertices)

        theta = np.pi / 180

        line_image = hough_lines(roi_image, 2, theta, 15, 40, 20)
        result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
        return result

    #white_output = 'road_clip_output.mp4'
    clip1 = VideoFileClip(input_path)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output_path, audio=False)


# Deploy on streamlit
st.title("Road Lane-Line Detection")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])


if uploaded_file is not None:
    input_path = f"input_{uploaded_file.name}"
    output_path = f"output_{uploaded_file.name}"

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.text("Processing video...")
    video_processing(input_path, output_path)
    st.text("Video processing complete!")

    st.text("Original Video")
    st.video(input_path)

    st.text("Lane Detected Video")
    st.video(output_path)


