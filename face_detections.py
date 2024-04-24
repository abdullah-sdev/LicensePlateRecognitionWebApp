import cv2
import matplotlib.pyplot as plt
from datetime import datetime

#nameing Files
def nameyourFile():
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
    return current_time

def apply_brightness_contrast(input_img, brightness=0, contrast=0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

# imagePath = 'D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording\DataSet-CarLicensePlate\DataSet-CarLicensePlate\DSC_1071.JPG'
# img = cv2.imread(imagePath)
# print(img.shape)
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)
#
#
#
# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )
# enhanced_image = apply_brightness_contrast(gray_image, brightness=50, contrast=50)
# face = face_classifier.detectMultiScale(
#     enhanced_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 90)
# )
#
# count = 0
# for (x, y, w, h) in face:
#     face = img[y:y + h, x:x + w]  # slice the face from the image
#     filename = nameyourFile()
#     cv2.imwrite('static/faces/cropped/' + filename +str(count) + '.jpg', face)  # save the image
#     count += 1
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

# facedetections, faceimgs = faceDetector(image)
def faceDetector(image):
    # img = cv2.imread(image)
    img = image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    enhanced_image = apply_brightness_contrast(gray_image, brightness=0, contrast=0)
    face = face_classifier.detectMultiScale(
        enhanced_image, scaleFactor=1.1, minNeighbors=5, minSize=(80, 90)
    )

    count = 0
    for (x, y, w, h) in face:
        face = img[y:y + h, x:x + w]  # slice the face from the image
        filename = nameyourFile()
        cv2.imwrite('static/faces/cropped/' + filename + str(count) + '.jpg', face)  # save the image
        count += 1
        image_copy = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
        cv2.imwrite('static/faces/detected/' + filename + '.jpg', image_copy)

    return face
if __name__ == '__main__':
    image = 'D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording\DataSet-CarLicensePlate\DataSet-CarLicensePlate\DSC_1071.JPG'
    facesbbox = faceDetector(image)

    # drawfacebbox(facesbbox)