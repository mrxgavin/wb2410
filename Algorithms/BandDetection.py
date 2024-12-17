import cv2
import numpy as np
# import gaussian filter1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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


def detect_valleys(profile, min_prominence=10, min_width=5, max_width=100):
    """
    改进的波谷检测算法
    
    参数:
    profile: 强度曲线
    min_prominence: 最小显著度
    min_width: 最小波谷宽度
    max_width: 最大波谷宽度
    """
    from scipy.signal import find_peaks

    # 寻找波谷（通过寻找-profile的波峰）
    inverted_profile = -profile
    peaks, properties = find_peaks(
        inverted_profile,
        prominence=min_prominence,  # 最小显著度
        width=(min_width, max_width),  # 波谷宽度范围
        distance=50  # 最小距离
    )
    
    return peaks, properties

def split_merged_bands(image, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    band_region = image[y:y+h, x:x+w]
    
    # 计算和平滑垂直profile
    vertical_profile = np.mean(band_region, axis=0)
    smoothed_profile = gaussian_filter1d(vertical_profile, sigma=2)

    significant_minima, properties = detect_valleys(
        smoothed_profile,
        min_prominence=np.std(smoothed_profile),  # 使用标准差作为显著度阈值
        min_width=5,
        max_width=w
    )
    
    # ######################################
    # # 创建figure对象
    # fig = go.Figure()

    # # 添加平滑轮廓线
    # fig.add_trace(go.Scatter(
    #     x=list(range(len(smoothed_profile))),
    #     y=smoothed_profile,
    #     mode='lines',
    #     name='Smoothed Profile',
    #     line=dict(color='blue')
    # ))

    # # 添加显著极小值点
    # fig.add_trace(go.Scatter(
    #     x=significant_minima,
    #     y=smoothed_profile[significant_minima],
    #     mode='markers',
    #     name='Significant Minima',
    #     marker=dict(
    #         color='red',
    #         size=10,
    #         symbol='circle'
    #     ),
    #     hovertemplate='Position: %{x}<br>Intensity: %{y:.2f}<extra></extra>'
    # ))

    # # 更新布局
    # fig.update_layout(
    #     title='Band Intensity Profile with Detected Minima',
    #     xaxis_title='Position (pixels)',
    #     yaxis_title='Intensity',
    #     hovermode='x',
    #     showlegend=True
    # )

    # # 显示图表
    # st.plotly_chart(fig, use_container_width=True)
    # ######################################

    # 将significant_minima转换为原始图像坐标系
    significant_minima = [x + m for m in significant_minima]
    
    new_bands = []
    cnt = cnt.reshape(-1, 2)  # 将轮廓点转换为(N,2)形状
    
    # 根据minima分割轮廓点
    if len(significant_minima) > 0:
        # 添加起始和结束x坐标
        split_points = [x] + significant_minima + [x + w]
        
        # 遍历每个分割区间
        for i in range(len(split_points) - 1):
            # 获取当前区间的x范围
            x_start = split_points[i]
            x_end = split_points[i + 1]
            
            # 根据x坐标筛选轮廓点
            mask = (cnt[:, 0] >= x_start) & (cnt[:, 0] < x_end)
            if np.any(mask):
                new_cnt = cnt[mask]
                # 确保轮廓点格式正确
                new_cnt = new_cnt.reshape(-1, 1, 2)
                new_bands.append(new_cnt)
    else:
        # 如果没有显著的minima,保持原始轮廓不变
        new_bands.append(cnt.reshape(-1, 1, 2))
    
    return new_bands