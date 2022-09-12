import cv2
import imutils
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans
import pytesseract
import shutil
import os
import random


def readImage(name, path=""):
    # read image from file
    if len(path) > 0 and path[-1] != '/':
        path += '/'
    img = cv2.imread(path+name,cv2.IMREAD_COLOR)

    # resize image 
    shape = img.shape
    scale = shape[1] / 600
    height = int(shape[0]//scale)
    img = cv2.resize(img, (600, height) )

    #convert to grayscale and smoothen image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 11, 17, 17) 

    # show both original(only resized) and edited image in grayscale
    images = [img, gray]
    _, axs = pyplot.subplots(1, 2, figsize=(20, 10))
    axs = axs.flatten()
    for im, ax in zip(images, axs):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    pyplot.show()

    #return both original and grayscale image
    return img, gray


def findContours(img, gray):
    # detect edges on image in grayscale, with min and max threshold values set to 30 and 300
    # only edges with intensity gradient above and beyond this 30-300 boundaries will be displayed
    edged = cv2.Canny(gray, 30, 200) 

    # look for contours (edges with closed surfaces) on our edges-representing image
    contours, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    imgCpy=img.copy()
    cv2.drawContours(imgCpy,contours,-1,(0,255,0),3)

    # sort detected contours from biggest to smallest, then take first 30 of them 
    bestContours = sorted(contours, key = cv2.contourArea, reverse = True) [:30]
    imgCpy2 = img.copy()
    cv2.drawContours(imgCpy2,bestContours,-1,(0,255,0),3)

    #present results of above three steps - edges, contours and best contours
    images = [edged, imgCpy, imgCpy2]
    _, axs = pyplot.subplots(1, 3, figsize=(20, 10))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pyplot.show()

    return bestContours


def findCarPlate(img, bestContours):
    carPlate = None
    # iterate over best contures - we look for that one with rectangular shape and for edges - that should be the car plate
    for c in bestContours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
            carPlate = approx
            break

    # communicate if there is no car plate detected
    if carPlate is None:
        detected = 0
        print ("No contour detected")
    # show detected car plate marked on original image
    else:
        imgCpy3 = img.copy()
        cv2.drawContours(imgCpy3, [carPlate], -1, (0, 0, 255), 3)
        pyplot.imshow(cv2.cvtColor(imgCpy3, cv2.COLOR_BGR2RGB))
    return carPlate


def findDominantColour(finalPlate):
    clusters = 5 
    org_img = finalPlate.copy()
    flat_img = np.reshape(finalPlate,(-1,3))

    kmeans = KMeans(n_clusters=clusters,random_state=0)
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')

    percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]
    p_and_c = zip(percentages,dominant_colors)
    p_and_c = sorted(p_and_c,reverse=True)

    dominantColourRGB = p_and_c[0][1][::-1]

    block = np.ones((50,50,3),dtype='uint')
    pyplot.figure(figsize=(12,8))
    for i in range(clusters):
        pyplot.subplot(1,clusters,i+1)
        block[:] = p_and_c[i][1][::-1] # we have done this to convert bgr(opencv) to rgb(matplotlib) 
        pyplot.imshow(block)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.xlabel(str(round(p_and_c[i][0]*100,2))+'%')

    bar = np.ones((50,500,3),dtype='uint')
    pyplot.figure(figsize=(12,8))
    pyplot.title('Proportions of colors in the image')
    start = 0
    i = 1
    for p,c in p_and_c:
        end = start+int(p*bar.shape[1])
        if i==clusters:
            bar[:,start:] = c[::-1]
        else:
            bar[:,start:end] = c[::-1]
        start = end
        i+=1

    pyplot.imshow(bar)
    pyplot.xticks([])
    pyplot.yticks([])

    rows = 1000
    cols = int((org_img.shape[0]/org_img.shape[1])*rows)
    img = cv2.resize(org_img,dsize=(rows,cols),interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy,(rows//2-250,cols//2-90),(rows//2+250,cols//2+110),(255,255,255),-1)
    pyplot.show()

    return dominantColourRGB


def extractCarPlate(img, carPlate):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask with zeros in size of whole image
    mask = np.zeros(gray.shape,np.uint8)
    # create mask with white colour (1) where selected licence plate is and black (0) elsewhere
    new_image = cv2.drawContours(mask,[carPlate],0,255,-1,)
    # apply mask - as a result we get blask image with only car plate visible
    new_image = cv2.bitwise_and(img,img,mask=mask)

    # cut out only the part representing car plate
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    croppedCarPlate = img[topx:bottomx+1, topy:bottomy+1]

    # show and return car plate image
    pyplot.imshow(cv2.cvtColor(croppedCarPlate, cv2.COLOR_BGR2RGB))
    return croppedCarPlate


def rotatePlate(img, cnt):
    # find minimum area rectangle
    rect = cv2.minAreaRect(cnt)

    # find four corner vertices
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])

    # prepare 4 pairs of corresponding points - "before" and "after"
    # input 4 points
    src_pts = box.astype("float32")
    # output 4 points, for straight positioned plate - calculated with its size
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    # get transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    pyplot.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    shape = warped.shape
    # check if height is bigger than width - if so, then it is wrngly rotated
    if shape[0] > shape[1] :
        rotated = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        pyplot.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        return rotated
    return warped


def checkIfPlatesAreYellow(dominantColourRGB):
    # check colour boundaries
    if (dominantColourRGB > [0, 93, 22]).all() and (dominantColourRGB < [255, 255, 45]).all() :
        return "As the car plate background is yellow, it may be from countries like: \nEUROPE: Luxembourg, Netherlands, United Kingdom, also Denmark and Hungary but only for commercial vehicles. \nASIA: North Korea, Oman, Israel \nAFRICA: Namibia Tanzania, Zimbabwe, Gabon, Suriname"
    elif (dominantColourRGB > [4, 31, 86]).all() and (dominantColourRGB < [50, 150, 220]).all() :
        return "As the car plprocess_videoate background is red, it may be from Nepal, Bhutan"
    elif (dominantColourRGB < [38, 38, 38]).all() :
        return "As the car plate background is black, it may be from \nASIA: North Yemen, Pakistan, Papua New Guinea, Singapore, South Yemen, Timor Leste, Vietnam, Brunei, Myanmar, India, Indonesia, Malaysia \nAFRICA: Nigeria, Sierra Leone, Somalia, Tanganyika, Angola, Egypt, Eritrea, Ethiopia, Libya \nAMERICAS: Argentina, Guyana"
    return "Background of this carplate is white, thats the most popular case and doe's not help much"


def readTextFromPlates(pletes):
    pyplot.imshow(cv2.cvtColor(pletes, cv2.COLOR_BGR2RGB))
    gray = cv2.cvtColor(pletes, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 11, 17, 17) 
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    pyplot.imshow(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(croppedCarPlate, config='--psm 13')
    print("Detected license plate Number is:",text)

if __name__ == "__main__":
    def dummy():
        pyplot.close()

    # Supress pyplot windows, comment out if graphical results desired
    pyplot.show = dummy

    # Exampe 1
    img, gray = readImage('auto.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)

    # Exampe 2
    img, gray = readImage('auto3.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 3
    img, gray = readImage('auto5.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 4
    img, gray = readImage('auto6.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 5
    img, gray = readImage('auto9.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 6
    img, gray = readImage('auto10.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 7
    img, gray = readImage('auto17.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)

    # Example 8
    img, gray = readImage('auto18.jpg', "../data/cars_reg")
    bestContours = findContours(img, gray)
    carPlate = findCarPlate(img, bestContours)
    croppedCarPlate = extractCarPlate(img, carPlate)
    finalPlate = rotatePlate(img, carPlate)
    dominantColourRGB = findDominantColour(croppedCarPlate)
    msg = checkIfPlatesAreYellow(dominantColourRGB)
    print(msg)
    readTextFromPlates(finalPlate)