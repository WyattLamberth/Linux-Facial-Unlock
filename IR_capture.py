import cv2
import os

def capture_ir_image():
    # Open webcam in IR sensor this value MUST be 2 for logitech BRIO - DO NOT CHANGE
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

    image_count = 1  # Initialize image count

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Display IR image
        cv2.imshow('IR Image', frame)

        # Press 'q' to exit
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 32:  # Press Spacebar to capture and save image
            filename = f"/home/wyatt/Developer/tf-learning/face_images/IR_images/image_{image_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"IR image saved as {filename}")
            image_count += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_ir_image()
