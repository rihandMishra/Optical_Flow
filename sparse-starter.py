import cv2 as cv
import numpy as np
from scipy.signal import convolve2d

corner_params = {
    "maxCorners": 300,
    "qualityLevel": 0.2,
    "minDistance": 2,
    "blockSize": 7
}

video = cv.VideoCapture("Test.mp4")
if not video.isOpened():
    print("Error: Couldn't open video file!")
    exit()

flow_color = (255, 0, 0)

success, prev_frame = video.read()
if not success:
    print("Error: Couldn't read the first frame!")
    exit()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_corners = cv.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)
trail_mask = np.zeros_like(prev_frame)

def get_flow_derivatives(prev_img, curr_img):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    dx = convolve2d(prev_img, x_kernel, mode='same') + convolve2d(curr_img, x_kernel, mode='same')
    dy = convolve2d(prev_img, y_kernel, mode='same') + convolve2d(curr_img, y_kernel, mode='same')
    dt = convolve2d(curr_img, t_kernel, mode='same') - convolve2d(prev_img, t_kernel, mode='same')

    return dx, dy, dt

def track_points_lk(prev_img, curr_img, points, window_size=15):
    half_w = window_size // 2
    h, w = prev_img.shape
    dx, dy, dt = get_flow_derivatives(prev_img, curr_img)

    num_points = points.shape[0]
    new_points = np.copy(points).astype(np.float32)
    tracked_status = np.zeros(num_points, dtype=np.uint8)

    for i, point in enumerate(points):
        x, y = point[0]
        x_int, y_int = int(x), int(y)

        if x_int - half_w < 0 or x_int + half_w >= w or y_int - half_w < 0 or y_int + half_w >= h:
            continue

        win_dx = dx[y_int - half_w: y_int + half_w + 1, x_int - half_w: x_int + half_w + 1].flatten()
        win_dy = dy[y_int - half_w: y_int + half_w + 1, x_int - half_w: x_int + half_w + 1].flatten()
        win_dt = dt[y_int - half_w: y_int + half_w + 1, x_int - half_w: x_int + half_w + 1].flatten()

        A = np.vstack((win_dx, win_dy)).T
        b = -win_dt

        if np.linalg.det(A.T @ A) > 1e-6:
            flow_vector = np.linalg.inv(A.T @ A) @ A.T @ b
            new_points[i, 0, 0] = x + flow_vector[0]
            new_points[i, 0, 1] = y + flow_vector[1]
            tracked_status[i] = 1

    return new_points, tracked_status

print("Starting optical flow tracking. Press 'q' to quit.")
while True:
    success, current_frame = video.read()
    if not success:
        print("End of video.")
        break

    current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

    if prev_corners is not None and len(prev_corners) > 0:
        if prev_corners.shape[0] < corner_params["maxCorners"] // 2:
            prev_corners = cv.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)

        next_corners, status = track_points_lk(prev_gray, current_gray, prev_corners)
        status = status.reshape(-1, 1)

        good_old = prev_corners[status.flatten() == 1]
        good_new = next_corners[status.flatten() == 1]

        for new, old in zip(good_new, good_old):
            new_x, new_y = new.ravel().astype(int)
            old_x, old_y = old.ravel().astype(int)

            trail_mask = cv.line(trail_mask, (new_x, new_y), (old_x, old_y), flow_color, 2)
            current_frame = cv.circle(current_frame, (new_x, new_y), 3, flow_color, -1)

        output_frame = cv.add(current_frame, trail_mask)
        prev_gray = current_gray.copy()
        prev_corners = good_new.reshape(-1, 1, 2)
    else:
        output_frame = current_frame
        prev_corners = cv.goodFeaturesToTrack(prev_gray, mask=None, **corner_params)

    cv.imshow("Lucas-Kanade Optical Flow", output_frame)

    if cv.waitKey(10) & 0xFF == ord('q'):
        print("Exiting...")
        break

video.release()
cv.destroyAllWindows()
print("Finished.")
