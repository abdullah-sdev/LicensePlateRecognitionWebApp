from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from ANPRsystem import lprsReader, lprsDetector, saveImglp
from face_detections import faceDetector



# Webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')


@app.route('/', methods=['POST', 'GET'])
def index():

    return render_template('index.html', upload=False)


@app.route('/picture', methods=['POST', 'GET'])
def picture():
    if request.method == 'POST':
        # get input
        upload_file = request.files['image_name']
        filename = upload_file.filename

        #save upload image
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)

        image = cv2.imread(path_save)
        # print(image.shape)
        detections, imglps = lprsDetector(image)

        # facedetections, faceimgs = faceDetector(image)

        text = lprsReader(imglps)

        facesbbox = faceDetector(image)
        # print(f'faces: {facesbbox}')

        # for (x, y, w, h) in facesbbox:
        #     print('face detected')
        #     x1, y1, x2, y2 = x, y, w + x, y + h
        #     image2 = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        #     cv2.imwrite('static/face/detected/' + filename, image2)

        saveImglp(detections, image, text, filename)


        licensecropped = np.array(imglps[0])
        cv2.imwrite('static/roi/' + filename, licensecropped)


        return render_template('mytemplate.html', upload=True, upload_image=filename, text=text)

    return render_template('mytemplate.html', upload=False)


@app.route('/video', methods=['POST', 'GET'])
def video():

    return 404




if __name__ == "__main__":
    app.run(debug=True)