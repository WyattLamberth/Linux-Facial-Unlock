import cv2
import os

def capture_ir_video():
    # Open webcam in IR sensor, this value MUST be 2 for Logitech BRIO - DO NOT CHANGE
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create directory to save images
    if not os.path.exists("IR_Images"):
        os.makedirs("IR_Images")

    frame_count = 0
    video_length = 15  # Length of video to capture in seconds
    frame_interval = 2  # Interval to save frames (every 15th frame)

    while frame_count < video_length * 30:  # Assuming 30 frames per second
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Display IR image
        cv2.imshow('IR Image', frame)

        # Save every 15th frame
        if frame_count % frame_interval == 0:
            filename = f"IR_Images/frame_{frame_count // frame_interval}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame {frame_count // frame_interval} saved as {filename}")

        frame_count += 1

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_ir_video()
