# Import required packages
# https://www.linux.com/training-tutorials/using-tesseract-ubuntu/
import cv2
from pytesseract import image_to_string 
# from textblob import TextBlob
from polyglot.detect import Detector
 
# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = 'System_path_to_tesseract.exe'
 
# Read image from which text needs to be extracted
# img = cv2.imread("sample.jpg")
# img = cv2.imread("russian.jpg")
# img = cv2.imread("spanish.png")
img = cv2.imread("french.png")
# img = cv2.imread("books.jpg")
# img = cv2.imread("map.jpg")
# img = cv2.imread("new_york.jpg")
img = cv2.imread("new_york2.jpeg")
img = cv2.imread("broadway.jpeg")
img = cv2.imread("roadsigns.jpg")
img = cv2.imread("roadsing.jpeg")
img = cv2.imread("nachuj.jpg")
# img = cv2.imread("roadsign.jpg")
# img = cv2.imread("blanco.jpg")

# Preprocessing the image starts
 
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
 
# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
 
# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
# Creating a copy of image
im2 = img.copy()
 
# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()
 
# Looping through the identified contours
# Then rectangular part is cropped and passed on
# to pytesseract for extracting text from it
# Extracted text is then written into the text file
langs = ["eng", "rus", "ara", "chi_sim"]
for lang in langs:
    text = ""
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        
        # Open the file in append mode
        
        # # Apply OCR on the cropped image
        text += image_to_string(cropped, lang)
        # text += image_to_string(cropped, "rus") + "\n"
        # text += image_to_string(cropped, "ara") + "\n"
        # text += image_to_string(cropped, "chi_sim") + "\n"

        # b = TextBlob(text)
        # language = b.detect_language()


    text2 = ""
    text2 += image_to_string(img, lang) 
    # text2 += image_to_string(img, "rus") + "\n"
    # text2 += image_to_string(img, "ara") + "\n"
    # text2 += image_to_string(img, "chi_sim") + "\n"
    print("\n strings equal: ", text2 == text)
    text += " " + text2
    file = open("recognized.txt", "a")
    print("found text: ", text)
    # Appending the text into file
    file.write(text + "\n")

    languages = []
    try :
        languages = Detector(text).languages
    except Exception as e:
        print("Unexpected error: ", str(e))        

    for language in languages:
        print(language)
        file.write(str(language) + "\n")

    # Close the file
    file.close