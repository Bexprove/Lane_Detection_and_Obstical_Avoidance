import numpy as np
import cv2
import os
import yaml

# -------- CONFIG --------
INVERSE_DEPTH_PATH = os.path.expanduser("~/project_ws/midas_frame.npy")
SCALE_SAVE_PATH = os.path.expanduser("~/project_ws/midas_scale.yaml")
ABS_DEPTH_SAVE_PATH = os.path.expanduser("~/project_ws/midas_abs_depth.npy")
WINDOW_NAME = "Click on point with known distance"
# ------------------------

# Load inverse depth map
inverse_depth = np.load(INVERSE_DEPTH_PATH)
abs_depth = None
scale = None

def on_mouse(event, x, y, flags, param):
    global scale, abs_depth

    if event == cv2.EVENT_LBUTTONDOWN:
        inv_val = inverse_depth[y, x]
        print(f"\n📍 Clicked at ({x}, {y}) - Inverse depth: {inv_val:.6f}")

        try:
            real_distance = float(input("Enter real-world distance at this point (in meters): "))
            scale = real_distance * inv_val  # ✅ since rel = 1/inv, scale = d * inv
            print(f"✅ Computed scale = {scale:.4f}")

            abs_depth = scale / inverse_depth  # ✅ final depth in meters
            np.save(ABS_DEPTH_SAVE_PATH, abs_depth)
            print(f"💾 Absolute depth saved to: {ABS_DEPTH_SAVE_PATH}")

            with open(SCALE_SAVE_PATH, 'w') as f:
                yaml.dump({'scale': float(scale)}, f)
            print(f"💾 Scale saved to: {SCALE_SAVE_PATH}")

            print("✅ Calibration complete. Press ESC to exit.")
        except:
            print("❌ Invalid input. Try again.")

# Visualize inverse depth
norm_vis = cv2.normalize(inverse_depth, None, 0, 255, cv2.NORM_MINMAX)
vis_color = cv2.applyColorMap(norm_vis.astype(np.uint8), cv2.COLORMAP_JET)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

while True:
    cv2.imshow(WINDOW_NAME, vis_color)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

