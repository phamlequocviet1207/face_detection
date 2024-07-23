import os
import argparse

import cv2
import mediapipe as mp


def process_img(img, face_detect):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detect.process(img_rgb)
    H, W, _ = img.shape
    if out.detections:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h, = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)

            #blur
            img[y1: y1 + h, x1:x1 + w, :] = cv2.blur(img[y1: y1 + h, x1:x1 + w, :], (100, 100))
    return img


args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default='None')

args = args.parse_args()

face_detect = mp.solutions.face_detection

#for live camera
cap = cv2.VideoCapture(0)

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with face_detect.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detect:
    if args.mode in ["image"]:
        # read img
        img = cv2.imread(args.filePath)
        img = process_img(img, face_detect)


        # save img
        cv2.imwrite(os.path.join(output_dir, 'output_blur_img.png'), img)

    elif args.mode in ["video"]:


        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_vid = cv2.VideoWriter(os.path.join(output_dir, "output_blur_video.mp4"),
                                     cv2.VideoWriter_fourcc(*'MP4V'),
                                     25,
                                     (frame.shape[1], frame.shape[0]))

        while True:
            frame = process_img(frame, face_detect)

            output_vid.write(frame)

            ret, frame = cap.read()
        cap.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detect)

            cv2.imshow("frame", frame)
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
