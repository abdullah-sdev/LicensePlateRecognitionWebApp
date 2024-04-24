import cv2
import easyocr
from lprs_utils import lprsDetector, lprsReader, saveImglp, getpicturesFromFiles


print('Program Started...')
# image_path = 'C:/Users/BlueSpirit\Downloads\Project\DataSet-CarLicensePlate/WhatsappImg.jpeg'
filesArrayForPics = getpicturesFromFiles('D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording\DataSet-CarLicensePlate\DataSet-CarLicensePlate - Copy\images')

for picture in filesArrayForPics:
    print(f'For Picture... {picture}')
    image = cv2.imread(picture)
    detections, imglp = lprsDetector(image)
    lp_texts = lprsReader(imglp)
    # output
    # to save pictures
    saveImglp(detections, image, lp_texts)