# main_opticalflow_navigation.py

from ai2thor.controller import Controller
import numpy as np
import cv2

# ==== Constants ====
WINDOW_SIZE = 15
ALPHA = 1.0  # Target attraction constant
GAMMA = 2.0  # Obstacle repulsion constant

# ==== Setup Controller ====
controller = Controller(scene="FloorPlan1", width=640, height=480, renderInstanceSegmentation=True)
controller.reset("FloorPlan1")

# ==== State Variables ====
prev_gray = None
prev_corners = None
goal_coords = (320, 240)  # Center pixel for now

# ==== Optical Flow Functions ====
def get_sparse_flow(prev_img, curr_img, prev_pts):
    lk_params = dict(winSize=(WINDOW_SIZE, WINDOW_SIZE), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)
    good_old = prev_pts[status == 1]
    good_new = next_pts[status == 1]
    return good_old, good_new

def estimate_foe(good_old, good_new):
    A = []
    b = []
    for (x1, y1), (x2, y2) in zip(good_old, good_new):
        vx = x2 - x1
        vy = y2 - y1
        A.append([vy, -vx])
        b.append(x1 * vy - y1 * vx)
    A = np.array(A)
    b = np.array(b)
    foe = np.linalg.lstsq(A, b, rcond=None)[0]
    return tuple(foe)

# ==== Potential Field ====
def compute_target_field(foe, goal, alpha=ALPHA):
    dx = goal[0] - foe[0]
    dy = goal[1] - foe[1]
    norm = np.linalg.norm([dx, dy]) + 1e-6
    return alpha * dx / norm, alpha * dy / norm

def compute_obstacle_field(flow_map, foe, gamma=GAMMA):
    # Using Otsu threshold on flow magnitude
    mag = np.sqrt(flow_map[..., 0] ** 2 + flow_map[..., 1] ** 2)
    _, thresh = cv2.threshold(mag.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    obstacle_mask = cv2.GaussianBlur(thresh, (15, 15), 0)
    grad_y, grad_x = np.gradient(obstacle_mask.astype(np.float32))
    rep_force = gamma * np.array([np.mean(grad_x), np.mean(grad_y)])
    return tuple(rep_force)

def get_control_action(force_vec):
    fx, fy = force_vec
    if abs(fx) < 5:
        return "MoveAhead"
    elif fx < 0:
        return "RotateLeft"
    else:
        return "RotateRight"

# ==== Main Loop ====
for step in range(30):
    event = controller.step(action="Pass")
    rgb = event.frame
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.3, minDistance=7)
        continue

    good_old, good_new = get_sparse_flow(prev_gray, gray, prev_corners)
    flow_map = np.zeros((gray.shape[0], gray.shape[1], 2))
    for (x1, y1), (x2, y2) in zip(good_old, good_new):
        flow_map[int(y1), int(x1)] = [x2 - x1, y2 - y1]

    foe = estimate_foe(good_old, good_new)
    f_att = compute_target_field(foe, goal_coords)
    f_rep = compute_obstacle_field(flow_map, foe)
    total_force = np.array(f_att) - np.array(f_rep)

    action = get_control_action(total_force)
    print(f"Step {step}: FOE={foe}, Force={total_force}, Action={action}")
    controller.step(action=action)

    prev_gray = gray
    prev_corners = good_new.reshape(-1, 1, 2)

controller.stop()
