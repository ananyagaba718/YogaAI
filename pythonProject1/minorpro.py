import cv2
import mediapipe as mp
import time
import math

class PoseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

# Existing pose checking functions
def check_crucifix_pose(detector, img, lmList):
    if len(lmList) > 0:
        angle_left_elbow = detector.findAngle(img, 11, 13, 15)  # Left elbow
        angle_right_elbow = detector.findAngle(img, 12, 14, 16)  # Right elbow
        elbow_angle_threshold = (170, 190)
        correct_left_elbow = elbow_angle_threshold[0] <= angle_left_elbow <= elbow_angle_threshold[1]
        correct_right_elbow = elbow_angle_threshold[0] <= angle_right_elbow <= elbow_angle_threshold[1]
        return correct_left_elbow and correct_right_elbow

def check_hands_raised_pose(detector, lmList):
    if len(lmList) > 0:
        left_wrist_y = lmList[15][2]  # Y coordinate of left wrist
        right_wrist_y = lmList[16][2]  # Y coordinate of right wrist
        head_y = lmList[0][2]  # Head position
        return left_wrist_y < head_y and right_wrist_y < head_y
    return False

def check_cat_pose(detector, img, lmList):
    if len(lmList) > 0:
        shoulder_mid = [(lmList[11][1] + lmList[12][1]) // 2, (lmList[11][2] + lmList[12][2]) // 2]
        hip_mid = [(lmList[23][1] + lmList[24][1]) // 2, (lmList[23][2] + lmList[24][2]) // 2]
        angle_back = detector.findAngle(img, 0, 12, 24)
        if angle_back < 160:
            return True
    return False

def check_balasana_pose(detector, img, lmList):
    if len(lmList) > 0:
        # Check the angle at the knees (between the hips and feet)
        knee_angle_left = detector.findAngle(img, 23, 25, 27)  # Left leg
        knee_angle_right = detector.findAngle(img, 24, 26, 28)  # Right leg

        # Check the angle between the torso and legs (hips to shoulders)
        back_angle_left = detector.findAngle(img, 11, 23, 25)  # Left side of the back
        back_angle_right = detector.findAngle(img, 12, 24, 26)  # Right side of the back

        # Display the angles on the screen for debugging
        cv2.putText(img, f'Knee L: {int(knee_angle_left)}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Knee R: {int(knee_angle_right)}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Back L: {int(back_angle_left)}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Back R: {int(back_angle_right)}', (50, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Conditions for Balasana Pose (with more flexible thresholds)
        correct_knee_angles = 20 <= knee_angle_left <= 40 and 20 <= knee_angle_right <= 40
        correct_back_angle = 290 <= back_angle_left < 350 and 290 <= back_angle_right < 350

        # Return whether both the knees are bent correctly and the back is sufficiently lowered
        return correct_knee_angles and correct_back_angle
    return False

def check_dandasana_pose(detector, img, lmList):
    if len(lmList) > 0:
        # Check the angle at the knees (between the hips, knees, and feet)
        knee_angle_left = detector.findAngle(img, 23, 25, 27)  # Left leg
        knee_angle_right = detector.findAngle(img, 24, 26, 28)  # Right leg

        # Check the angle of the back (hips to shoulders)
        back_angle_left = detector.findAngle(img, 11, 23, 25)  # Left side of the back
        back_angle_right = detector.findAngle(img, 12, 24, 26)  # Right side of the back

        # Display the angles on the screen for debugging
        cv2.putText(img, f'Knee L: {int(knee_angle_left)}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Knee R: {int(knee_angle_right)}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Back L: {int(back_angle_left)}', (50, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f'Back R: {int(back_angle_right)}', (50, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Conditions for Dandasana Pose
        correct_knee_angles = 200 <= knee_angle_left <= 250 and 200 <= knee_angle_right <= 250
        correct_back_angle = 60 <= back_angle_left <=100 and 60 <= back_angle_right <= 100

        # Return whether both the knees are straight and the back is upright
        return correct_knee_angles and correct_back_angle
    return False

def main():
    print("Select a pose to check:")
    print("1. Crucifix Pose")
    print("2. Hands Raised Pose")
    print("3. Cat Pose")
    print("4. Balasana Pose -------->")
    print("5. <---------- Dandasana Pose")
    choice = input("Enter your choice (1/2/3/4/5): ")

    cap = cv2.VideoCapture(0)  # Use webcam
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 1280, 720)

    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            if choice == '1':
                pose_correct = check_crucifix_pose(detector, img, lmList)
                pose_status = "Crucifix Pose: Correct" if pose_correct else "Crucifix Pose: Incorrect"
            elif choice == '2':
                pose_correct = check_hands_raised_pose(detector, lmList)
                pose_status = "Hands Raised Pose: Correct" if pose_correct else "Hands Raised Pose: Incorrect"
            elif choice == '3':
                pose_correct = check_cat_pose(detector, img, lmList)
                pose_status = "Cat Pose: Correct" if pose_correct else "Cat Pose: Incorrect"
            elif choice == '4':
                pose_correct = check_balasana_pose(detector, img, lmList)
                pose_status = "Balasana Pose: Correct" if pose_correct else "Balasana Pose: Incorrect"
            elif choice == '5':
                pose_correct = check_dandasana_pose(detector, img, lmList)
                pose_status = "Dandasana Pose: Correct" if pose_correct else "Dandasana Pose: Incorrect"
            else:
                pose_status = "Invalid Choice"
                color = (255, 0, 0)

            color = (0, 255, 0) if pose_correct else (0, 0, 255)
            cv2.putText(img, pose_status, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()