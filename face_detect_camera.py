import cv2
import mediapipe as mp

face_detect = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with face_detect.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detect:
    while True:
        _, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detect.process(frame_rgb)
        H, W, _ = frame.shape

        if out.detections:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h, = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)



        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()