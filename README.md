# AiPO-Project-Localisation-Guesser
Image recognition and processing project written in Python using OpenCV. Determines geographical position using given video clip.

# Installation

  Install dependencies and libraries that the project is using.
  Run the following commands:

## PyTesseract

  ```
  $ sudo apt install tesseract-ocr
  $ pip install pytesseract
  ```

## OpenCV

  ```
  $ pip install opencv-python
  ```

## Polyglot

  ```
  $ sudo apt-get install libicu-dev
  $ pip install numpy
  $ pip install polyglot
  ```

## Pytohn dependencies
  Python dependencies can be installed as well by using requirements.txt file
  ```
  pip install -r requirements.txt
  ```

# Tesseract Data

Tesseract requires external data files to properly detect languages.
Theese files can be downloaded form offical GitHub repository.
Setting of **TESSDATA_PREFIX** environment variable to the directory with data files is required.

**Warning:** The complete data library is large (~3GB).