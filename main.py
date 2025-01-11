import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

class AdvancedFingerCounter:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Skin color range in YCrCb space
        self.skin_ycrcb_min = np.array([0, 135, 85])
        self.skin_ycrcb_max = np.array([255, 180, 135])
        
        # Depth estimation parameters
        self.focal_length = 1000
        self.avg_hand_width = 0.08  # meters
        
    def preprocess_frame(self, frame):
        """Advanced frame preprocessing with multiple color spaces"""
        # Convert to different color spaces
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create masks in different color spaces
        mask_ycrcb = cv2.inRange(ycrcb, self.skin_ycrcb_min, self.skin_ycrcb_max)
        mask_hsv = cv2.inRange(hsv, np.array([0, 30, 70]), np.array([20, 255, 255]))
        
        # Combine masks
        combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask

    def detect_fingers_mediapipe(self, frame):
        """Detect fingers using MediaPipe landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Get fingertip landmarks (indices 4,8,12,16,20)
            fingertips = [4, 8, 12, 16, 20]
            finger_bases = [2, 5, 9, 13, 17]  # Corresponding base points
            
            # Count extended fingers
            count = 0
            for tip_idx, base_idx in zip(fingertips, finger_bases):
                tip = landmarks.landmark[tip_idx]
                base = landmarks.landmark[base_idx]
                
                # Check if finger is extended
                if tip.y < base.y:  # For thumb, use different logic
                    count += 1
                    
            return count
        return 0

    def detect_fingers_contours(self, mask):
        """Detect fingers using contour analysis"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
            
        # Get largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Convex hull analysis
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
        
        if defects is None:
            return 0
            
        # Advanced defect filtering
        finger_count = 1  # Start with 1 for thumb
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            
            # Calculate triangle sides
            a = euclidean(np.array(start), np.array(far))
            b = euclidean(np.array(end), np.array(far))
            c = euclidean(np.array(start), np.array(end))
            
            # Calculate angle
            angle = np.degrees(np.arccos((a**2 + b**2 - c**2)/(2*a*b)))
            
            # Count finger if angle is within range
            if 20 < angle < 90:
                finger_count += 1
                
        return min(finger_count, 5)  # Cap at 5 fingers

    def estimate_depth(self, contour):
        """Estimate depth of hand for better accuracy"""
        if contour is None:
            return None
            
        rect = cv2.minAreaRect(contour)
        width = min(rect[1])
        
        if width > 0:
            distance = (self.avg_hand_width * self.focal_length) / width
            return distance
        return None

    def count_fingers(self, frame):
        """Main method combining multiple detection approaches"""
        # Preprocess frame
        mask = self.preprocess_frame(frame)
        
        # Get counts from different methods
        count_mediapipe = self.detect_fingers_mediapipe(frame)
        count_contours = self.detect_fingers_contours(mask)
        
        # Weight the results based on confidence
        if count_mediapipe > 0:
            # MediaPipe detection is generally more reliable
            final_count = count_mediapipe
        else:
            final_count = count_contours
            
        return final_count, mask

def main():
    cap = cv2.VideoCapture(0)
    finger_counter = AdvancedFingerCounter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Count fingers
        finger_count, mask = finger_counter.count_fingers(frame)
        
        # Display results
        cv2.putText(frame, f'Fingers: {finger_count}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frames
        cv2.imshow('Frame', frame)
        cv2.imshow('Mask', mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()