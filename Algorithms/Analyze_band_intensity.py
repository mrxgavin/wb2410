import cv2
import numpy as np

def analyze_band_intensity(image, contours):
    """Improved band intensity analysis function"""
    results = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Expand ROI to include surrounding background
        roi_y1 = max(0, y - 10)
        roi_y2 = min(image.shape[0], y + h + 10)
        roi_x1 = max(0, x - 10)
        roi_x2 = min(image.shape[1], x + w + 10)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Create a mask
        mask = np.zeros_like(roi)
        roi_cnt = cnt - np.array([roi_x1, roi_y1])[None, None, :]
        cv2.drawContours(mask, [roi_cnt], -1, 255, -1)
        
        # Calculate background (using the edge region of the ROI)
        bg_mask = cv2.bitwise_not(mask)
        background = np.median(roi[bg_mask > 0])
        
        # Calculate band intensity
        band_pixels = roi[mask > 0]
        mean_intensity = np.mean(band_pixels) - background
        
        # Adjust numerical scale
        mean_intensity = (mean_intensity / 100)  # Divide intensity by 100
        area = cv2.contourArea(cnt) / 100  # Divide area by 100
        total_intensity = mean_intensity * len(band_pixels)
        
        # Calculate integrated density
        integrated_density = total_intensity * area
        
        results.append({
            'band_number': i+1,
            'area': area,
            'mean_intensity': mean_intensity,
            'total_intensity': total_intensity,
            'integrated_density': integrated_density
        })
    
    return results
