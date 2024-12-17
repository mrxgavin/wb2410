import cv2
import numpy as np

def detect_bands(image, min_distance=2, min_area=50):
    """Improved band detection function"""
    # Preprocess to reduce shadow effects
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu thresholding to better separate bands and background
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove noise and shadows
    kernel_v = np.ones((15, 1), np.uint8)  # Vertical kernel
    kernel_h = np.ones((1, 5), np.uint8)   # Horizontal kernel

    # Open operation to remove small noise
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # Close operation to fill gaps inside bands
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)

    # Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # Merge contours that are too close
    merged_contours = []

    # Sort contours by x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    current_contour = None
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filter out contours that are too small (likely noise)
        if area < min_area:  # Minimum area threshold
            continue

        # Filter out contours with abnormal aspect ratios (likely shadows)
        aspect_ratio = w / h
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # Limit aspect ratio
            continue

        if current_contour is None:
            current_contour = contour
        else:
            # Get the bounding box of the current contour
            curr_x, _, curr_w, _ = cv2.boundingRect(current_contour)

            # If two contours are close enough, merge them
            if x - (curr_x + curr_w) < min_distance:
                # Create a merged contour
                combined_contour = np.vstack((current_contour, contour))
                current_contour = combined_contour
            else:
                merged_contours.append(current_contour)
                current_contour = contour

    # Add the last contour
    if current_contour is not None:
        merged_contours.append(current_contour)

    # Optimize each merged contour
    final_contours = []
    for cnt in merged_contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Slightly expand the ROI area
        roi = image[max(0, y-5):min(image.shape[0], y+h+5),
                    max(0, x-5):min(image.shape[1], x+w+5)]

        # Reapply thresholding within the ROI
        _, roi_thresh = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the largest contour within the ROI
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

        if roi_contours:
            # Choose the largest contour
            max_cnt = max(roi_contours, key=cv2.contourArea)
            # Adjust coordinates back to the original image
            max_cnt = max_cnt + \
                np.array([max(0, x-5), max(0, y-5)])[None, None, :]
            final_contours.append(max_cnt)

    return final_contours
