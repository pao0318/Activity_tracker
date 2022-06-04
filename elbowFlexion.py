def elbowFlexion(image, landmarks, params):
    params['name'] = 'Elbow Flexion'
    
    # Get coordinates
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
    # Calculate angle
    angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)
    angle = np.round((angle_left + angle_right)/2)
    
    # displaying angle in the frame
    cv2.putText(image, str(int(angle_left)), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
    cv2.putText(image, str(int(angle_right)), 
                           tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
    
    ## Counter logic
    
    if angle > 160 and (params['stage'] is None or params['stage'] == 'up'):
        if params['stage'] == 'up':
            params['times'][(params['counter']-1) % params['n_reps']] = abs(time.time() - params['t1'])
            params['rep_time'] = abs(time.time() - params['t1'])
            params['rep_time_list'].append(params['rep_time'])
            
        params['t1'] = time.time()
        
    if angle > 160:
        params['stage'] = 'down'
        params['max_angle'] = max(params['max_angle'], angle)
        
    if angle < 38 and params['stage'] == 'down':
        params['stage'] = 'up'
        params['counter'] += 1
        params['t2'] = time.time()
        
    if angle < 38:
        params['min_angle'] = min(params['min_angle'], angle)
        
        ## Error Calculation
    
    
    return image, params