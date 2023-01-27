###### Imported Library ###################################################################################################################################################################################
import cv2
import mediapipe as mp
import numpy as np
import datetime

##############################################################################################################################################################################################
mpose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First joint point
    b = np.array(b) # Mid joint point
    c = np.array(c) # End joint point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)   
    return angle 

C = cv2.VideoCapture("input.mp4")
kneeAngle_thresh = 139
time = 5
time_diff = 0
knee_bent = False
rep = False
rep_count = 0
fontScale = 1
with mpose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
    while C.isOpened():
        ret, frame = C.read()
        if np.shape(frame) == ():
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            (hip, knee, ankle) = (23, 25, 27) if (landmarks[mpose.PoseLandmark.LEFT_KNEE.value].z < landmarks[mpose.PoseLandmark.RIGHT_KNEE.value].z) else (24, 26, 28)
            hip_coord = [landmarks[hip].x, landmarks[hip].y]
            knee_coord = [landmarks[knee].x, landmarks[knee].y]
            ankle_coord = [landmarks[ankle].x, landmarks[ankle].y]
            angle_at_knee = calculate_angle(hip_coord, knee_coord, ankle_coord)
            image = cv2.putText(image,"Knee Bend Angle: " + str(angle_at_knee), (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            if angle_at_knee > kneeAngle_thresh:
                knee_bent = False
            if angle_at_knee < kneeAngle_thresh and knee_bent==False:
                knee_bent = True
                rep = False
                now = datetime.datetime.now()
            if knee_bent:
                time_diff = (datetime.datetime.now() - now).total_seconds()
                image = cv2.putText(image,"Time for recent move: " + str(time_diff)+str(" Sec"), (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
            if time_diff > time and rep==False:
                rep_count += 1
                rep = True
        except:
            pass
        
        image = cv2.putText(image,"Rep count:" + str(rep_count), (20, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        image = cv2.putText(image,"Intern Hiring Assignment" ,(300, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    print(rep_count)
    C.release()
    cv2.destroyAllWindows()
