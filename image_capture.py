import cv2

cap = cv2.VideoCapture(0)  # Use 0 for the first webcam
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)

    if k%256 == 27:  # ESC to exit
        break
    elif k%256 == 32:  # SPACE to capture an image
        cv2.imwrite(f'face_{count}.png', frame)
        count += 1

cap.release()
cv2.destroyAllWindows()
