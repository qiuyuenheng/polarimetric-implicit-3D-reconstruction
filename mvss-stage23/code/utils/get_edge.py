from PIL import Image
import numpy as np
import cv2

if __name__ == '__main__':
    img_name = "D:\Desktop\wwbottle\mask/029_pseudo.png"
    # img = np.array(Image.open("D:\Desktop/"+img_name))[:,:,:3]
    # edge_img = img.copy()

    img = cv2.imread(img_name, 0)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    gaussImg = 255 - cv2.Canny(blurred, 100, 200)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(gaussImg, kernel, iterations=1)

    cv2.imshow("Img", erosion)
    cv2.waitKey(0)