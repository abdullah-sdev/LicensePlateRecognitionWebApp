import cv2
import time
import datetime

from ANPRsystem import lprsDetector, lprsReader, saveImglp, getpicturesFromFiles

def video(videoPath):
    # Paths and configurations

    confidence_threshold = 0.2

    start = time.time_ns()
    frame_count = 0
    fps = -1

    cap = cv2.VideoCapture(videoPath)

    while True:
        _, frame = cap.read()

        # image_PreProcess
        image = cv2.imread(frame)

        # Perform object detection
        detections, imglp = lprsDetector(image)

        if imglp:
            lp_texts = lprsReader(imglp)
            saveImglp(detections, frame, lp_texts, 'static/predict')

        # Count fps
        frame_count += 1

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.imshow("Camera", frame)

        # if cv2.waitKey(1) == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()


# ../TestDataRecording/VID_20230530_143713.mp4

if __name__ == '__main__':
    videoPath = 'D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording/VID_AJKFA1.mp4'
    video(videoPath)