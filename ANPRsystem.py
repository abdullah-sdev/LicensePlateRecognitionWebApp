
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

model_path = 'models/best.pt'
# Load the pre-trained YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
reader = easyocr.Reader(['en'])

def image_processing(img):
    # Resize the image while maintaining aspect ratio
    width = 640
    height = int(img.shape[0] * width / img.shape[1])
    resized_image = cv2.resize(img, (width, height))

    norm_img_filter = np.zeros((resized_image.shape[0], resized_image.shape[1]))
    norm_image = cv2.normalize(resized_image, norm_img_filter, 0, 255, cv2.NORM_MINMAX)

    # deskew = deskew_image(norm_image)




    # resized_image = img
    # Convert image to RGB
    # image_grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    image_grey = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    # Convert image to PIL format
    image_pil = Image.fromarray(image_grey)
    # image = Image.open(img).resize((64, 64))
    return norm_image


def split_string_on_number(input_string):
    # Split the string when a number comes
    result = re.split(r'(\d+)', input_string)

    # Filter out empty strings from the result
    result = list(filter(None, result))

    return result

# function which takes image to detect a numberplate and return cordinates
def lprsDetector(image):

    threshold_lpd = 0

    # image640 = image_processing(image)
    #Detect Lisence Plate from image by YOLO-Model
    lps_bb_c = model(image)   # Bounding boxs and confidences of image (maybe multiple)
    # lps_bb_c.show()
    bounding_boxes = lps_bb_c.xyxy[0][:, :4].tolist()
    confidences = lps_bb_c.xyxy[0][:, 4].tolist()

    bbwc = lps_bb_c.xyxy[0][:, :5].tolist()

    number_plates = []

    for detectedLP in bbwc:

        if detectedLP[4] > threshold_lpd:
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
            print(f'imagesize: {image_np.size}')
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
            magic_color = apply_brightness_contrast(gray, brightness=0, contrast=0)

            ocrReadings = reader.readtext(magic_color)
            lp_text_cache = ''
            if ocrReadings:
                precds = process_license_plate(ocrReadings)
                print(precds)
                for words in precds:
                    word = words

                    lp_text.append(word)
            else:
                lp_text.append('Not Recognized')

    print(f'License Plates: {lp_text}')
    return lp_text



def process_license_plate(result):
    bbox = []
    text = []
    conf = []

    for i in result:
        bbox.append(i[0])
        text.append(i[1])
        conf.append(i[2])

    # print(conf)

    text_word = []
    for texts in text:
        word = re.sub('^-+|-+$|[^A-Za-z0-9-]', '-', texts)

        # Replace consecutive hyphens with a single hyphen
        word = re.sub(r'-+', '-', word)

        # Remove leading and trailing hyphens
        word = re.sub(r'^-|-$', '', word)

        # Remove all hyphens
        word = word.replace('-', '')

        text_word.append(word)

    print(f'texts: {text_word}')

    lp_chars = []
    slit_cache = ''
    for words in text_word:

        if (len(words) <= 7 and len(words) >= 2):
            splittedstring = split_string_on_number(words)

            for slit in splittedstring:

                if (slit.isdigit()):
                    if (len(slit) == 3 or len(slit) == 4):
                        slit_cache += slit
                        print(f'slit: {slit}')
                if (slit.isdigit() != True):
                    if (len(slit) == 2 or len(slit) == 3):
                        slit = slit.upper()
                        slit_cache += slit
                        print(f'slit: {slit}')


        elif (len(slit_cache) <= 7 and len(slit_cache) >= 5) == False:
            if (words.isdigit()):
                if (len(words) == 3 or len(words) == 4):
                    slit_cache += words

            else:
                if (len(words) == 2 or len(words) == 3):
                    words = words.upper()
                    slit_cache += words
    # print(slit_cache)
    lp_chars.append(slit_cache)

    # print(f'lp_chars: {lp_chars}')

    return lp_chars



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
def saveImglp(cords, image, lp_texts, filename):
    counter_lp = 0
    # current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
    # filename = nameyourFile()
    filename = filename
    # filesavename = filename + '-(' + str(counter_lp) + ')'
    filesavename = filename
    for display in lp_texts:
        numberplate_text = 'LP: ' + lp_texts[counter_lp]
        # print(detections[counter_lp])
        cords_as_int = [int(value) for value in cords[counter_lp]]
        # print(cords_as_int)
        x1, y1, x2, y2, _ = cords_as_int
        image_copy = cv2.rectangle(image, (x1, y1), (x2, y2), (36, 255, 12), 1)
        # print(f'{numberplate_detect_conf} \n {numberplate_text_conf} \n {numberplate_text}')
        image_copy = draw_rect_bg(image_copy, numberplate_text, 1,
                                  cv2.FONT_HERSHEY_SIMPLEX , cords[counter_lp],
                                  (0, 0, 0), (36, 255, 12))

        # cv2.imwrite('../TestDataRecording/' + '/NumberPlate', image_copy)



        counter_lp += 1
    if counter_lp >= 0:
        cv2.imwrite('static/predict/' + filesavename , image_copy)



# writes the data in Excel
def saveToExcel():
    return 0

# writes the data in Database
def saveToDb():
    return 0

#nameing Files
def nameyourFile():
    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
    return current_time

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


    image_path = 'C:/Users/BlueSpirit\Downloads\Project\DataSet-CarLicensePlate/WhatsappImg.jpeg'

    image = cv2.imread(image_path)

    print('Program Started...')
    detections, imglp = lprsDetector(image)

    lp_texts = lprsReader(imglp)
    # output
    saveImglp(detections, image, lp_texts, 'static/roi/')
