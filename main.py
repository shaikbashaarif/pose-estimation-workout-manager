import cv2
import mediapipe as mp
import numpy as np
import math
import time


white = (255,255,255)
black = (0,0,0)
green = (0,280, 0)
red = (0, 0, 255)
blue = (245,117,25)


def process_frame(frame, pose):
    #extract dimension of frame
    height, width, _ = frame.shape

    #convert the BGR image to RGB image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    #processing image with mediapipe
    results = pose.process(image)

    #convert the RGB image to BGR image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image,results,height,width
    


def extract_landmarks(results, mp_pose):
    shoulder = None
    elbow = None
    wrist = None

    try:
        landmarks = results.pose_landmarks.landmark  # ✅ Also corrected here

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
    except:
        pass

    return shoulder, elbow, wrist



def calculate_angle(shoulder, elbow, wrist):
    a = np.array(shoulder)
    b = np.array(elbow)
    c = np.array(wrist)


    radians = math.atan2(c[1]- b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])

    angle = np.abs(radians*180/math.pi)

    
    return angle

def calculate_and_display_angle(shoulder, elbow, wrist, stage, counter):
    angle = calculate_angle(shoulder, elbow, wrist)
    if angle >180.0:
        angle = 360 - angle

    if angle>160:
        stage = '  up'
    elif angle<50 and stage == '  up':
        stage = '  down'
        counter = counter + 1

    
    return angle, stage, counter


def render_ui(image, angle, stage, counter, width):
    angle_max =180
    angle_min =25


    cv2.rectangle(image, (int(width/2)-150, 0), (int(width/2)+250,73), blue, -1)
    cv2.putText(image, "Workout Manager", (int(width/2)-100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)

    cv2.rectangle(image, (0, 0), (255,73), blue, -1)
    cv2.putText(image, "REPS ", (15, 25), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)
    cv2.putText(image, str(counter), (15, 60), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)

    cv2.putText(image, " STAGE ", (95, 25), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)
    cv2.putText(image, stage, (95, 60), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)

    progress = ((angle - angle_min)/(angle_max-angle_min))*100
    cv2.rectangle(image,(50,350),(50+int(progress*2),370), green, cv2.FILLED)
    cv2.rectangle(image,(50,350),(250,370), white, 2)
    cv2.putText(image, f"{int(progress)}%", (50, 400), cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)

    return image


def display_time(image, start_time, height, width):
    elapsed_time = time.time() - start_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = int((seconds - int(seconds))*1000)

    elapsed_time = "{:02}:{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds), int(milliseconds))

    position = (int(width/2 + 200), int(height/2))

    cv2.putText(image, elapsed_time, position, cv2.FONT_HERSHEY_COMPLEX, 1, white, 2, cv2.LINE_AA)

    return image

def display_clock_icon(image, icon_path, height, width):
    clock_icon = cv2.imread(icon_path,-1)

    icon_width = 375
    icon_height = 200
    clock_icon = cv2.resize(clock_icon, (icon_width, icon_height))

    x_offset = width - clock_icon.shape[1] - 120
    y_offset = 260

    if clock_icon.shape[2] == 4:
        # Split the icon into its channels and get the alpha channel
        alpha_icon = clock_icon[:, :, 3] / 255.0
        alpha_image = 1.0 - alpha_icon

        for c in range(0, 3):
            # Overlay the clock icon on the image
            image[y_offset:y_offset + clock_icon.shape[0], x_offset:x_offset + clock_icon.shape[1], c] = \
                alpha_icon * clock_icon[:, :, c] + \
                alpha_image * image[y_offset:y_offset + clock_icon.shape[0], x_offset:x_offset + clock_icon.shape[1], c]
    

    return image


def run_pose_detection(mp_drawing, mp_pose, filename):
    cap =cv2.VideoCapture(filename)
    stage = None
    counter = 0
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            if cap.isOpened():
                ret, frame = cap.read()

                if not ret or cv2.waitKey(1) == ord('q'):
                    break

                image,results,height,width = process_frame(frame,pose)

                shoulder, elbow, wrist = extract_landmarks(results,mp_pose)

                if shoulder and elbow and wrist:
                    angle, stage, counter = calculate_and_display_angle(shoulder, elbow, wrist, stage, counter)
                else:
                    angle, stage = 0, None  # or skip drawing angle-dependent UI


                image = render_ui(image, angle, stage, counter, width)

                image = display_clock_icon(image, r"assest\clock.png", height, width)
                image = display_time(image, start_time, height, width)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117,66), thickness = 2, circle_radius =5),
                                          mp_drawing.DrawingSpec(color=(245, 66,230), thickness = 2, circle_radius =5))
                
                cv2.imshow("AI Workout Manager", image)

if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose


    run_pose_detection(mp_drawing, mp_pose, r'D:\project\workout_manager\assest\pushup.mp4')
    # cap = cv2.VideoCapture(r'D:\project\workout_manager\assest\pushup.mp4')

    # while True:
    #     if cap.isOpened():
    #         ret, frame = cap.read()

    #         if not ret or cv2.waitKey(1) == ord('q'):
    #             break
    #         cv2.imshow("Workout Manager", frame)