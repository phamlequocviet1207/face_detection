# watching video from "https://www.youtube.com/watch?v=DRMBqhrfxXg&t=834s"

import cv2
import mediapipe as mp
import os

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


img_path = './data/dog.jpg'
img = cv2.imread(img_path)

H,W, _ = img.shape

face_detect = mp.solutions.face_detection


# face detect
with face_detect.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detect:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detect.process(img_rgb)

    print(out.detections)

    for detection in out.detections:
        location_data = detection.location_data
        bbox = location_data.relative_bounding_box

        x1,y1,w,h, = bbox.xmin, bbox.ymin, bbox.width, bbox.height

        x1 = int(x1*W)
        y1 = int(y1*H)
        w = int(w*W)
        h = int(h*H)

        cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (0,255,0), 10)
        img[y1:y1 + h, x1: x1 + w, :] = cv2.blur(img[y1:y1 + h, x1: x1 + w, :], (90,90), )

    cv2.imshow("img", img)
    cv2.waitKey(0)

#save img
cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
cv2.destroyAllWindows()

