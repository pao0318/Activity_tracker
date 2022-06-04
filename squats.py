def squats(image, landmarks, params):
    params['name'] = 'Squats'
    
    # Get coordinates
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    # Calculate angle
    angle_left = calculate_angle(left_hip, left_knee, left_ankle)
    angle_right = calculate_angle(right_hip, right_knee, right_ankle)
    angle = np.round((angle_left + angle_right)/2)
    
    # displaying angle in the frame
    cv2.putText(image, str(int(angle_left)), 
                           tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
    cv2.putText(image, str(int(angle_right)), 
                           tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
    
    ## Counter logic
    
    if angle > 160 and (params['stage'] is None or params['stage'] == 'down'):
        if params['stage'] == 'down':
            params['times'][(params['counter']-1) % params['n_reps']] = abs(time.time() - params['t1'])
            params['rep_time'] = abs(time.time() - params['t1'])
            params['rep_time_list'].append(params['rep_time'])
            
        params['t1'] = time.time()
        
    if angle > 160:
        params['stage'] = 'up'
        
    if angle < 70 and params['stage'] == 'up':
        params['stage'] = 'down'
        params['counter'] += 1
        params['t2'] = time.time()
        
        ## Error Calculation
    
    
    return image, params