import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
import math

class AIVirtualMouse:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Get screen size
        self.screen_w, self.screen_h = pyautogui.size()

        # Cursor smoothing
        self.prev_x, self.prev_y = 0, 0
        self.alpha = 0.2  # Smoothing factor

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        return lmList

    def moveCursor(self, lmList):
        if len(lmList) == 0:
            return

        # Get fingertip positions
        x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
        x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip
        x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger tip

        # Calculate distances
        distance_index_thumb = math.hypot(x2 - x1, y2 - y1)  # Distance between Index & Thumb
        distance_index_middle = math.hypot(x3 - x1, y3 - y1)  # Distance between Index & Middle

        # Move cursor if all three fingers are close
        if distance_index_thumb < 40 and distance_index_middle < 40:
            screen_x = np.interp(x1, [100, 640 - 100], [0, self.screen_w])
            screen_y = np.interp(y1, [100, 480 - 100], [0, self.screen_h])

            # Apply smoothing
            smooth_x = self.alpha * screen_x + (1 - self.alpha) * self.prev_x
            smooth_y = self.alpha * screen_y + (1 - self.alpha) * self.prev_y

            pyautogui.moveTo(smooth_x, smooth_y)

            self.prev_x, self.prev_y = smooth_x, smooth_y

    def checkClick(self, lmList):
        if len(lmList) < 12:
            return

        # Get fingertip positions
        x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
        x2, y2 = lmList[4][1], lmList[4][2]  # Thumb tip
        x3, y3 = lmList[12][1], lmList[12][2]  # Middle finger tip

        # Calculate distances
        distance_index_thumb = math.hypot(x2 - x1, y2 - y1)  # Distance between Index & Thumb
        distance_index_middle = math.hypot(x3 - x1, y3 - y1)  # Distance between Index & Middle

        # Left Click: If middle finger lifts while index & thumb remain close
        if distance_index_thumb < 40 and distance_index_middle > 50:
            pyautogui.click()
            print("Left Click")

        # Right Click: If thumb lifts while index & middle remain close
        if distance_index_thumb > 50 and distance_index_middle < 40:
            pyautogui.rightClick()
            print("Right Click")

def main():
    cap = cv2.VideoCapture(0)
    detector = AIVirtualMouse()

    if not cap.isOpened():
        print("Error: Camera not detected!")
        return

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        img = cv2.flip(img, 1)  # Mirror cam
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        detector.moveCursor(lmList)
        detector.checkClick(lmList)

        cv2.imshow("AI Virtual Mouse", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
