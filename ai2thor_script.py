import ai2thor.controller
import numpy as np
import cv2

# Run AI2-THOR in headless mode
controller = ai2thor.controller.Controller(scene="FloorPlan1", headless=True,width=640,height=480,renderInstanceSegmentation=False)

# Start the environment
controller.reset("FloorPlan1")

# Take an action

for _ in range(5):  # Run only for 5 steps
    event = controller.step(action="Pass")

rgb_frame=event.frame
bgr_frame=cv2.cvtColor(rgb_frame , cv2.COLOR_RGB2BGR)
cv2.imshow(bgr_frame)
cv2.waitkey(0)
cv2.destroyAllWindows()


# Stop the controller
controller.stop()
