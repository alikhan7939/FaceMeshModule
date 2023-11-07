############################################################### Face Mesh Module

############ Importing libraries for face mesh
import mediapipe as mp # mediapipe is good for drawing face mesh
import cv2 
import time # for drame p

############################################ Creating Face Mesh Module
#
########### initializing face mesh class

class FaceMeshDetector():
    def __init__(self,static_image_mode=False, max_num_faces=1, refine_landmarks=False, 
                min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils 
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.static_image_mode,max_num_faces=self.max_num_faces,
                                                  refine_landmarks=self.refine_landmarks,min_detection_confidence=self.min_detection_confidence,
                                                  min_tracking_confidence=self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # mediapipe did not acsept BGR input
        self.result = self.faceMesh.process(self.imgRGB)   # processing input 
        if self.result.multi_face_landmarks:
          for faceLms in self.result.multi_face_landmarks:
            self.mpDraw.draw_landmarks(image=img, landmark_list=faceLms, connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=self.drawSpec, connection_drawing_spec=self.drawSpec) # drawing landmark for each point in face
        return img  

############################################ Main Function to Call the Module
def main():
    cap = cv2.VideoCapture(0) # taking input from webcam
    pTime = 0
    detector = FaceMeshDetector() # face mesh class
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findFaceMesh(img)
        cTime =time.time() # correct time
        fps = 1/(cTime-pTime) # frame per scond
        pTime = cTime # Previous time
        cv2.putText(img, f'FBS:{str(int(fps))}', (15,80), cv2.FONT_HERSHEY_COMPLEX,
                    3, (15,255,10), 1) # add fps to output by cv2.putText
        cv2.imshow("img",img)
        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()   
