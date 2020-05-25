import cv2

img = cv2.imread(r'D:\Programming\Python\Machine Learning\Computer Vision\Facial Recognition\Unknown_faces\5c1ab0b7ed56fa5daab76ec7_MichelleObama-inauguration.jpg')
cv2.namedWindow("ye", cv2.WINDOW_NORMAL)
cv2.imshow('ye', img)
cv2.waitKey(0)
cv2.destroyAllWindows()