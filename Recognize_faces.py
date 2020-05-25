import face_recognition
import cv2
import os


class FacialRecognition:
    def __init__(self):
        self.names = []
        self.encoded_images = []

    def label_faces(self, path='.\\Known_faces\\'):
        """
        A function to learn faces of given set of images. Please sure that the image is clear,and
        there's only 1 face in the given image. This function uses the SSL(Single Shot Learning)
        technique to learn and verify faces in one go.
        """
        print("Labelling Faces")
        for filename in os.listdir(path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                name = filename.split('.')[0]
                image = face_recognition.load_image_file(path+filename)
                encoded_face = face_recognition.face_encodings(image)[0]
                self.names.append(name)
                self.encoded_images.append(encoded_face)

    def find_a_match(self, path='.\\Unknown_faces\\'):
        """
        This function will be used to guess the person from it's given face. If there is no match
        found, the face will be labelled as 'Unknown'.
        """
        print("Finding matches")
        name = "Unknown"
        for filename in os.listdir(path):
            image = face_recognition.load_image_file(path+filename)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            img = cv2.imread(path + filename)
            for face_location, face_encoding in zip(face_locations, face_encodings):
                results = face_recognition.compare_faces(self.encoded_images, face_encoding, 0.6)
                if True in results:
                    truth_index = results.index(True)
                    name = self.names[truth_index]
                (top, right, bottom, left) = face_location
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 5)
                cv2.rectangle(img, (left, bottom), (right, bottom+20), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (left+20, bottom+20), font, 1.0, (255, 255, 255), 2)
            cv2.namedWindow(f'{name}', cv2.WINDOW_NORMAL)
            cv2.imshow(f'{name}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


a = FacialRecognition()
a.label_faces()
a.find_a_match()
