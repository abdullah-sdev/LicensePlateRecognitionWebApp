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

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask app:

    ```bash
    python app.py
    ```

2. Open a web browser and go to `http://127.0.0.1:5000` to access the application.

3. Upload an image containing license plates or faces to detect and recognize them.

## Screenshots

![Screenshot 1](/static/predict/DSC_0453.jpg)
<!-- ![Screenshot 2](/screenshots/screenshot2.png) -->

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
