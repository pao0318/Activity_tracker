import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def calculate_angle_lateral(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
#     if angle >180.0:
#         angle = 360-angle
        
    return angle 


def get_bounding_box(frame, margin = 0.15):
    delta = int(margin * frame.shape[1])
    
    bounding_box_normalised = {
        'x1' : delta/frame.shape[1],
        'y1' : 0,
        'x2' : (frame.shape[1] - delta) / frame.shape[1],
        'y2' : 1
    }
    
    return bounding_box_normalised


def add_bounding_box(frame, flag = 'out', margin = 0.15):
    mp = {
        'out' : (0, 0, 0),
        'in' : (255, 255, 255),
        'good' : (0, 255, 0),
        'bad' : (0, 0, 255)
    }
    
    color = mp[flag]
    
    # Allowing margin % 
    delta = int(margin * frame.shape[1])
    bounding_box = [(delta, 0), (frame.shape[1] - delta, frame.shape[0])]
    
    cv2.rectangle(frame, bounding_box[0], bounding_box[1], color, thickness = 2)
    
    return frame


## Now we need to evaluate when flag turn true or false
## 1) if nose and both left and right hip lies in the bounding box we say 

def evaluate_flag(landmarks, bounding_box):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    visibility_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    visibility_left = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    visibility_nose = landmarks[mp_pose.PoseLandmark.NOSE.value].visibility
    visibility_left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
    visibility_right_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility
    
    flag = 'out'
    
    ## x is the horizontal normalised distance and y is the vertical normalise distance from (0, 0)
    if nose[0] >= bounding_box['x1'] and nose[0] <= bounding_box['x2'] and left_hip[0] >= bounding_box['x1'] and left_hip[0] <= bounding_box['x2'] and right_hip[0] >= bounding_box['x1'] and right_hip[0] <= bounding_box['x2'] and visibility_right >= 0.5 and visibility_left >= 0.5 and visibility_nose >= 0.5: 
        flag = 'in'
        
    return flag


def evaluate_flag_squats(landmarks, bounding_box):
    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    visibility_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    visibility_left = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
    visibility_nose = landmarks[mp_pose.PoseLandmark.NOSE.value].visibility
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    flag = 'out'
    
    ## x is the horizontal normalised distance and y is the vertical normalise distance from (0, 0)
    if nose[0] >= bounding_box['x1'] and nose[0] <= bounding_box['x2'] and left_hip[0] >= bounding_box['x1'] and left_hip[0] <= bounding_box['x2'] and right_hip[0] >= bounding_box['x1'] and right_hip[0] <= bounding_box['x2'] and visibility_right >= 0.5 and visibility_left >= 0.5 and visibility_nose >= 0.5 and left_ankle[0] >= bounding_box['x1'] and left_ankle[0] <= bounding_box['x2'] and right_ankle[0] >= bounding_box['x1'] and right_ankle[0] <= bounding_box['x2']: 
        flag = 'in'
        
    return flag


def add_info(frame, flag, params):
    ## Adding rectangle for Reps
    cv2.rectangle(frame, (0,0), (int(0.15 * 640)-2, 75), (135, 135, 88), -1)
    
    ## Adding rectangle for feedback
    cv2.rectangle(frame, (int(0.15 * 640) + 2, 2), (638 - int(0.15 * 640), 75), (135, 135, 88), -1)
    
    ## Adding one more rectangle
    cv2.rectangle(frame, (642 - int(0.15 * 640), 0), (640, 75), (135, 135, 88), -1)
    
    if flag == 'out':
        cv2.putText(frame, 'Come inside the Box', (130, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
                                )
        
        if params['play_box'] == False:
            playsound('./come_inside_the_box.mp3')
            params['play_box'] = True
            params['box_timer'] = time.time()
        else:
            if time.time() - params['box_timer'] >= 4:
                playsound('./come_inside_the_box.mp3')
                params['box_timer'] = time.time()
        
#     elif flag == 'in':
#         cv2.putText(frame, 'Perform the exercise', (130, 50), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA
#                                 )
        
    return frame


def add_feedback(image, params):
    ## Adding rectangle for Reps
    cv2.rectangle(image, (0,0), (int(0.15 * 640)-2, 75), (135, 135, 88), -1)
    
    ## Adding rectangle for feedback
    cv2.rectangle(image, (int(0.15 * 640) + 2, 2), (638 - int(0.15 * 640), 75), (135, 135, 88), -1)
    
    ## Adding one more rectangle
    cv2.rectangle(image, (642 - int(0.15 * 640), 0), (640, 75), (135, 135, 88), -1)
    
    ## displaying reps
    cv2.putText(image, str(params['counter']), (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    ## displaying Exercise Name
    cv2.putText(image, str(params['name']), (546, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    
    ## Displaying if doing bad
    if time.time() - params['t1'] > 1.25 * params['threshtime'] and params['stage'] == 'down':
        image = add_bounding_box(image, flag = 'bad')
        cv2.putText(image, 'Raise your arms', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        ## Playing sound
        if params['error_flag'] == False:
            playsound("./raise_arms.mp3")
            params['error_timer'] = time.time()
            params['error_flag'] = True
        else:
            if time.time() - params['error_timer'] >= 1.25 * params['threshtime']:
                playsound('./raise_arms.mp3')
                params['error_timer'] = time.time()
        
    elif time.time() - params['t2'] > 1.25 * params['threshtime'] and params['stage'] == 'up':
        image = add_bounding_box(image, flag = 'bad')
        cv2.putText(image, 'Lower your arms', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        ## Playing sound
        if params['error_flag'] == False:
            playsound('./lower_arms.mp3')
            params['error_timer'] = time.time()
            params['error_flag'] = True
        else:
            if time.time() - params['error_timer'] >= 1.25 * params['threshtime']:
                playsound('./lower_arms.mp3')
                params['error_timer'] = time.time()
                
    ### Displaying status after every n reps
    if params['counter'] % params['n_reps'] == 0 and params['counter'] >= params['n_reps']:
        if np.mean(params['times']) - params['threshtime'] > params['threshtime']/4:
            ## Add feedback for doing fast
            image = add_bounding_box(image, flag = 'bad')
            cv2.putText(image, 'Do Faster', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
        elif params['threshtime'] - np.mean(params['times']) > params['threshtime']/4:
            ## Add feedback for doing slow
            image = add_bounding_box(image, flag = 'bad')
            cv2.putText(image, 'Do slower', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        ## 15% tolerance
        elif abs(params['threshtime'] - np.mean(params['times'])) <= 0.22 * params['threshtime']:
            ## Add good feedback
            image = add_bounding_box(image, flag = 'good')
            cv2.putText(image, 'Doing Good', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                      
    return image


def add_feedback_squats(image, params):
    ## Adding rectangle for Reps
    cv2.rectangle(image, (0,0), (int(0.15 * 640)-2, 75), (135, 135, 88), -1)
    
    ## Adding rectangle for feedback
    cv2.rectangle(image, (int(0.15 * 640) + 2, 2), (638 - int(0.15 * 640), 75), (135, 135, 88), -1)
    
    ## Adding one more rectangle
    cv2.rectangle(image, (642 - int(0.15 * 640), 0), (640, 75), (135, 135, 88), -1)
    
    ## displaying reps
    cv2.putText(image, str(params['counter']), (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    ## displaying Exercise Name
    cv2.putText(image, str(params['name']), (546, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    
    ## Displaying if doing bad
    if time.time() - params['t1'] > 1.25 * params['threshtime'] and params['stage'] == 'up':
        image = add_bounding_box(image, flag = 'bad')
        cv2.putText(image, 'Go Down', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        ## Playing sound
        if params['error_flag'] == False:
            playsound("./go_down.mp3")
            params['error_timer'] = time.time()
            params['error_flag'] = True
        else:
            if time.time() - params['error_timer'] >= 1.25 * params['threshtime']:
                playsound('./go_down.mp3')
                params['error_timer'] = time.time()
        
    elif time.time() - params['t2'] > 1.25 * params['threshtime'] and params['stage'] == 'down':
        image = add_bounding_box(image, flag = 'bad')
        cv2.putText(image, 'Go up', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        ## Playing sound
        if params['error_flag'] == False:
            playsound('./go_up.mp3')
            params['error_timer'] = time.time()
            params['error_flag'] = True
        else:
            if time.time() - params['error_timer'] >= 1.25 * params['threshtime']:
                playsound('./go_up.mp3')
                params['error_timer'] = time.time()
                
    ### Displaying status after every n reps
    if params['counter'] % params['n_reps'] == 0 and params['counter'] >= params['n_reps']:
        if np.mean(params['times']) - params['threshtime'] > params['threshtime']/4:
            ## Add feedback for doing fast
            image = add_bounding_box(image, flag = 'bad')
            cv2.putText(image, 'Do Faster', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
        elif params['threshtime'] - np.mean(params['times']) > params['threshtime']/4:
            ## Add feedback for doing slow
            image = add_bounding_box(image, flag = 'bad')
            cv2.putText(image, 'Do slower', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        ## 15% tolerance
        elif abs(params['threshtime'] - np.mean(params['times'])) <= 0.22 * params['threshtime']:
            ## Add good feedback
            image = add_bounding_box(image, flag = 'good')
            cv2.putText(image, 'Doing Good', (130, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                      
    return image



def check_time(params):
    colors = []
    for x in params['rep_time_list']:
        if x >= 1.25 * params['threshtime']:
            colors.append('red')
        elif x <= 0.75 * params['threshtime']:
            colors.append('blue')
        else:
            colors.append('green')
    
    return colors
            

def plot(params):
    ## getting color for different points
    col = check_time(params)

    plt.figure(figsize = (15, 4))
    plt.grid()
    plt.plot(range(1, params['counter']+1), params['rep_time_list'], 'y-.')

    for i in range(len(params['rep_time_list'])):
        plt.scatter(i+1, params['rep_time_list'][i],c = col[i], s = 70,
                    linewidth = 0)

    plt.xticks(list(range(1, params['counter']+1)))
    plt.xlabel('Reps')
    plt.ylabel('Rep time')
    plt.title(params['name'])

    plt.text(params['counter']-4, max(params['rep_time_list'])-0.5,'Good Rep',
             fontsize = 13, bbox = dict(facecolor = 'green', alpha = 0.5))

    plt.text(params['counter']-2.5, max(params['rep_time_list'])-0.5,'Slow Rep',
             fontsize = 13, bbox = dict(facecolor = 'red', alpha = 0.5))

    plt.text(params['counter']-1.1, max(params['rep_time_list'])-0.5,'Fast Rep',
             fontsize = 13, bbox = dict(facecolor = 'blue', alpha = 0.5))
    
    try:
        plt.text(params['counter']-4, max(params['rep_time_list'])-1, 'Max Angle: ' + str(int(params['max_angle'])),
             fontsize = 13, bbox = dict(facecolor = 'yellow', alpha = 0.5))
        
        plt.text(params['counter']-2.5, max(params['rep_time_list'])-1, 'Min Angle: ' + str(int(params['min_angle'])),
             fontsize = 13, bbox = dict(facecolor = 'yellow', alpha = 0.5))
    except:
        pass
    
    plt.show()
