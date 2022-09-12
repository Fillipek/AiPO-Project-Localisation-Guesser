from email import message
from re import M
import cv2
import imutils
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans
import pytesseract
import shutil
import os
import random
import sys

import text_detection.text_detection as txtdet
import plate_reading.plate_reading as plateread


def do_processing(frame):
    shape = frame.shape
    scale = shape[1] / 600
    height = int(shape[0]//scale)
    frame = cv2.resize(frame, (600, height) )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 11, 17, 17) 
    return gray


def find_yellow_plates(frame, frame_gray):
    # Supress pyplot for this usage
    def dummy():
        pyplot.close()
    supressed = pyplot.show
    pyplot.show = dummy

    contours = plateread.findContours(frame, frame_gray)
    plate = plateread.findCarPlate(frame, contours)
    plate = plateread.extractCarPlate(frame, plate)
    plate = plateread.rotatePlate(frame, plate)
    dominant_color = plateread.findDominantColour(plate)
    msg = plateread.checkIfPlatesAreYellow(dominant_color)

    # disable plot supression after detection
    pyplot.imshow = supressed

    return msg


def detect_registration_plates(video_name, path=""):
    langs = ["eng"]
    # langs = ["eng", "rus", "ara", "chi_sim"]
    messages = []
    
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    cap = cv2.VideoCapture(path + video_name)

    i = 0
    ret = True
    while ret:
        i += 1
        if i%10 != 0:
            continue
        ret, frame = cap.read()
        print(f"Frame {i}")

        frame_gray = do_processing(frame)
        arr1, arr2 = txtdet.detect_language_on_image(frame, langs)
        for elem in arr1:
            print(elem)
        for elem in arr2:
            print(elem)
        # messages.append(f"Frame {i}: " + find_yellow_plates(frame, frame_gray))
        
    cap.release()

    return messages

    



def main():
    if len(sys.argv) < 2:
        print("Please specify video name in argument, like: python3 main.py [path to video].")
        sys.exit(1)

    video_in = sys.argv[1]
    langs = ["eng", "rus", "ara", "chi_sim"]

    cap = cv2.VideoCapture(video_in)
    out_file = open("main_run.log", "w")

    i = 0
    ret = True
    while ret:
        i += 1
        if i%30 == 0:
            print(f"Frame {i}")
            out_file.write(f"Frame {i}")
            frame_gray = do_processing(frame)
            out_txt, out_lang = txtdet.detect_language_on_image(frame, langs)
            for out in out_txt:
                print(out.strip('\n'))
                print(out_lang)
                out_file.write(out.strip('\n'))
            try:
                m = f"Frame {i}: " + find_yellow_plates(frame, frame_gray)
                out_file.write(m)
                print(m)
            except Exception as e:
                m = "Failed to find register plates: " + str(e)
                print(m)
                out_file.write(m)
        ret, frame = cap.read()
        continue
        
    cap.release()

    out_file.close()

    # reg_plates_msgs = detect_registration_plates("new_york.mkv", "./data/video")
    
    # print("Plate detection results:")
    # for msg in reg_plates_msgs:
    #     print(msg)

if __name__ == "__main__":
    main()