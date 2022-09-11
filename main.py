import sys
import cv2

import text_detection.text_detection as txtdet

def main():
    image = cv2.imread("data/text_rec/" + sys.argv[1])

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    langs = ["eng", "rus", "ara", "chi_sim"]
    arr1, arr2 = txtdet.detect_language_on_image(image, langs)
    print(arr1, arr2)

if __name__ == "__main__":
    main()