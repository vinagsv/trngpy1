import cv2
import torch
from torchvision import transforms

camera = cv2.VideoCapture(2)

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)
# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Transformation pipeline
transform = transforms.Compose([transforms.ToTensor()])


def generate_frames():
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        # Perform person detection using YOLOv5
        results = model(frame)

        # Process each detected person
        for detection in results.xyxy[0]:
            if detection[5] == 0:  # 0 corresponds to 'person' class
                xmin, ymin, xmax, ymax, confidence = detection[:5].cpu().numpy()

                # Convert the person area to grayscale for face detection
                person_area_gray = cv2.cvtColor(
                    frame[int(ymin) : int(ymax), int(xmin) : int(xmax)],
                    cv2.COLOR_BGR2GRAY,
                )

                # Perform face detection within the person area
                faces = face_cascade.detectMultiScale(
                    person_area_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                # If face detected, draw a green rectangle around the face
                if len(faces) > 0:
                    for fx, fy, fw, fh in faces:
                        cv2.rectangle(
                            frame,
                            (int(xmin) + fx, int(ymin) + fy),
                            (int(xmin) + fx + fw, int(ymin) + fy + fh),
                            (0, 255, 0),
                            2,
                        )  # Green color

                    # Draw blue rectangle around the whole person
                    cv2.rectangle(
                        frame,
                        (int(xmin), int(ymin)),
                        (int(xmax), int(ymax)),
                        (255, 0, 0),
                        2,
                    )  # Blue color
                    cv2.putText(
                        frame,
                        "Verifiable",
                        (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,  #text label font size
                        (255, 0, 0),
                        1,
                    )

                else:
                    # Draw rectangles around detected persons
                    cv2.rectangle(
                        frame,
                        (int(xmin), int(ymin)),
                        (int(xmax), int(ymax)),
                        (0, 165, 255),
                        2,
                    )  # Orange color
                    cv2.putText(
                        frame,
                        "Person",
                        (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,  #text label font size
                        (0, 165, 255),
                        1,
                    )

        try:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        except cv2.error:
            continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def cleanup():
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        for frame in generate_frames():
            # Display the resulting frame
            cv2.imshow("Frame", cv2.imdecode(frame, cv2.IMREAD_COLOR))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cleanup()
