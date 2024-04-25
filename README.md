# License Plate and Face Recognition Web App

This Python project utilizes YOLOv5 for license plate detection and face recognition integrated with Flask to create a web application.

## Features

- License plate detection using YOLOv5.
- Face recognition using a pre-trained model.
- Web interface for easy interaction.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/abdullah-sdev/LicensePlateRecognitionWebApp.git
    ```

2. Navigate to the project directory:

    ```bash
    cd LicensePlateRecognitionWebApp
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

5. Install project dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. Clone YOLOv5:

    ```bash
    git clone https://github.com/ultralytics/yolov5  # clone
    cd yolov5
    pip install -r requirements.txt  # install
    ```

## Usage

1. Navigate to the project directory:

    ```bash
    cd LicensePlateRecognitionWebApp
    ```

2. Open the project folder in VSCode.

3. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. Run the Flask app:

    ```bash
    python App.py
    ```

Now, you can use your web browser to access the application at `http://localhost:5000`.

## Screenshots

![Screenshot 1](/static/predict/DSC_0453.jpg)
<!-- ![Screenshot 2](/screenshots/screenshot2.png) -->

## Video Tutorial

Check out the video tutorial for a step-by-step guide on how to use the License Plate and Face Recognition Web App:

[![Video Tutorial](fypsubmission/VideoGuide.mp4)

## Dependencies

- Python 3.8
- YOLOv5
- Flask
- EasyOCR/PyTesseract
- Other dependencies listed in `requirements.txt`

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
