
import cv2
import pytesseract as tess
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
import torch
import openpyxl
import time
import matplotlib.pyplot as plt
import easyocr
import glob
import os
import re

model_path = '../yolov5/models/best.pt'
# Load the pre-trained YOLOv5 model
model = torch.hub.load('..\yolov5', 'custom', path=model_path, source='local')
reader = easyocr.Reader(['en'])


# function which takes image to detect a numberplate and return cordinates
def lprsDetector(image):

    threshold_lpd = 0

    #Detect Lisence Plate from image by YOLO-Model
    lps_bb_c = model(image)   # Bounding boxs and confidences of image (maybe multiple)
    # lps_bb_c.show()
    bounding_boxes = lps_bb_c.xyxy[0][:, :4].tolist()
    confidences = lps_bb_c.xyxy[0][:, 4].tolist()

    bbwc = lps_bb_c.xyxy[0][:, :5].tolist()

    number_plates = []

    for detectedLP in bbwc:

        if detectedLP[4] > threshold_lpd: # confidence check
            bboxs_ = detectedLP[:4]
            print(f'BoundingBox: {bboxs_}')
            print(f'Confidence: {detectedLP[4]} \n')

            x1, y1, x2, y2 = map(int, bboxs_)
            x, y, width, height = x1 + 5, y1 - 5, x2 - x1 - 10, y2 - y1 + 10

            # Perform bounds checking
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                print("Invalid region coordinates")
                continue

            # print(f'(x: {x} \n y: {y} \n width: {width} \n height: {height} )')
            # print(image)
            image_np = np.array(image)
            # print(image_np)
            # Crop the number plate region from the image
            number_plate_region = image_np[y:y + height, x:x + width]
            # print(number_plate_region)
            # break


            # Display using cv2.imshow
            # plt.imshow(number_plate_region)
            # plt.title('Cropped Image (Matplotlib)')
            # plt.show()

            # Convert the NumPy array to a PIL Image
            number_plate_image = Image.fromarray(number_plate_region)
            # append in numberplates
            number_plates.append(number_plate_image)

            # Display using Matplotlib
            plt.imshow(number_plate_image)
            plt.title('Cropped Image (Matplotlib)')
            # plt.pause(4)  # Add this line
            plt.show()
            # cv2.imshow(number_plate_image)

            # cv2.imshow('Cropped Image (OpenCV)', number_plate_region)

    return bbwc, number_plates


# function which takes image and cordinates to read the numberplate and return the read data
def lprsReader(lp_images):

    lp_text = []

    if (len(lp_images) > 0 ):
        for img in lp_images:
            image_np = np.array(img)

            roi_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)

            ocrReadings = reader.readtext(magic_color)
            lp_text_cache = ''
            if ocrReadings:
                for txt in range(len(ocrReadings)):
                    lp_text_cache += ocrReadings[txt][1]
                    word = re.sub('^-+|-+$|[^A-Za-z0-9-]', '-', lp_text_cache)
                    word = alter_string(word)
                lp_text.append(word)
            else:
                lp_text.append('Not Recognized')

    print(f'License Plates: {lp_text}')
    return lp_text


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

def alter_string(word):
    # Replace consecutive hyphens with a single hyphen
    word = re.sub(r'-+', '-', word)

    # Remove leading and trailing hyphens
    word = re.sub(r'^-|-$', '', word)

    # Ensure it starts and ends with an alphabet or number and has only one hyphen in between
    if not re.match(r'^[A-Za-z0-9]+(-[A-Za-z0-9]+)?$', word):
        word = '-'

    return word


def draw_rect_bg(img, numberplate_text, font_scale, font, cords, rectangle_bgr, font_color):
    # Calculate the text size for each line
    # (detect_width, detect_height) = cv2.getTextSize(f'{numberplate_detect_conf}', font, fontScale=font_scale, thickness=1)[0]
    # (text_width, text_height) = cv2.getTextSize(f'{numberplate_text_conf}', font, fontScale=font_scale, thickness=1)[0]
    (plate_width, plate_height) = cv2.getTextSize(f'{numberplate_text}', font, fontScale=font_scale, thickness=1)[0]

    # Set the text start position
    cords_as_int = [int(value) for value in cords]
    text_offset_x = cords_as_int[0]
    text_offset_y = cords_as_int[1]

    # Draw a rectangle for each line
    for text, width, height in [(f'{numberplate_text}', plate_width, plate_height)]:
        # Make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + width + 2, text_offset_y - height - 2))
        cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=font_color, thickness=1)
        # Move the y-coordinate for the next line
        text_offset_y -= height

    return img

#save sata in image
def saveImglp(cords, image, lp_texts):
    counter_lp = 0
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
    filesavename = current_time + '-(' + str(counter_lp) + ')'
    for display in lp_texts:
        numberplate_text = 'Text: ' + lp_texts[counter_lp]
        # print(detections[counter_lp])
        cords_as_int = [int(value) for value in cords[counter_lp]]
        # print(cords_as_int)
        x1, y1, x2, y2, _ = cords_as_int
        image_copy = cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 1)
        # print(f'{numberplate_detect_conf} \n {numberplate_text_conf} \n {numberplate_text}')
        image_copy = draw_rect_bg(image_copy, numberplate_text, 0.6,
                                  cv2.FONT_HERSHEY_DUPLEX, cords[counter_lp],
                                  (0, 0, 0), (36, 255, 12))

        # cv2.imwrite('../TestDataRecording/' + '/NumberPlate', image_copy)



        counter_lp += 1
    if counter_lp > 0:
        cv2.imwrite('runs/rexs/' + filesavename + '.jpg', image_copy)



# writes the data in Excel
def saveToExcel():
    return 0

# writes the data in Database
def saveToDb():
    return 0



# def getpicturesFromFiles(filePath):
#     image_types = ['*.jpg', '*.jpeg', '*.png']
#     # jpeg_pics = [f for f in glob.glob(filePath+"\*.jpeg")]
#     image_files = [f for image_type in image_types for f in glob.glob(os.path.join(filePath, image_type))]
#     print(image_files)
#     return image_files


def getpicturesFromFiles(file_path):
    image_types = ['*.jpg', '*.jpeg', '*.png']
    image_files = [f for image_type in image_types for f in glob.glob(os.path.join(file_path, image_type))]
    print(image_files)
    return image_files

# Run Main
if __name__ == '__main__':


    # image_path = 'C:/Users/BlueSpirit\Downloads\Project\DataSet-CarLicensePlate/WhatsappImg.jpeg'
    image_path = 'D:\Projects\Pythons\FaceRecognotion_c4.0\TestDataRecording\DataSet-CarLicensePlate\DataSet-CarLicensePlate - Copy\images\DSC_0329.jpg'

    image = cv2.imread(image_path)

    print('Program Started...')
    detections, imglp = lprsDetector(image)

    lp_texts = lprsReader(imglp)
    # output
    saveImglp(detections, image, lp_texts)