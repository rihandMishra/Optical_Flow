import cv2 as cv
import numpy as np

# Function to compute polynomial expansion for a neighborhood
def polynomial_expansion(image, window_size=5, sigma=1.1):
    # Apply Gaussian blur to smooth the image
    smoothed = cv.GaussianBlur(image, (window_size, window_size), sigma)
    
    # Compute gradients using Sobel operators
    grad_x = cv.Sobel(smoothed, cv.CV_32F, 1, 0, ksize=window_size)
    grad_y = cv.Sobel(smoothed, cv.CV_32F, 0, 1, ksize=window_size)
    
    # Compute the polynomial coefficients
    A = grad_x * grad_x
    B = grad_y * grad_y
    C = grad_x * grad_y
    
    return A, B, C

# Function to compute optical flow using Farneback's method
def compute_optical_flow(prev_frame, curr_frame, window_size=5, sigma=1.1):
    # Convert frames to grayscale
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
    
    # Compute polynomial expansion for both frames
    A1, B1, C1 = polynomial_expansion(prev_gray, window_size, sigma)
    A2, B2, C2 = polynomial_expansion(curr_gray, window_size, sigma)
    
    # Compute the displacement (optical flow)
    delta_A = A2 - A1
    delta_B = B2 - B1
    delta_C = C2 - C1
    
    # Solve for the flow vectors
    flow_x = (delta_A + delta_B) / (2 * (A1 + B1 + 1e-10))
    flow_y = (delta_C + delta_B) / (2 * (A1 + B1 + 1e-10))
    
    # Combine the flow components into a single flow matrix
    flow = np.stack((flow_x, flow_y), axis=-1)
    
    return flow

# Main function to process the video
def main():
    # Read the video
    cap = cv.VideoCapture("Test.mp4")
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return
    
    prev_frame = first_frame
    
    # Create a mask for visualization
    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255  # Set saturation to maximum
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute optical flow
        flow = compute_optical_flow(prev_frame, frame)
        
        # Convert flow to magnitude and angle
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Scale the magnitude to make it more visible
        magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        magnitude = np.clip(magnitude * 3, 0, 255)  # Increase magnitude scaling
        
        # Apply thresholding to remove small flow vectors (noise)
        min_magnitude_threshold = 10  # Adjust this value as needed
        magnitude[magnitude < min_magnitude_threshold] = 0
        
        # Update the mask for visualization
        mask[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
        mask[..., 2] = magnitude  # Value represents magnitude (brightness)
        
        # Convert HSV to RGB for display
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        
        # Display the input and output frames
        cv.imshow("input", frame)
        cv.imshow("dense optical flow", rgb)
        
        # Update the previous frame
        prev_frame = frame
        
        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()