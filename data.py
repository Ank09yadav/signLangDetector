from function import *
import cv2

# create directories for each actions and store frames
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)
        except:
            pass
# initialize mediapipe hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    ) as hands:

# collect data for each action
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                # Calculate the flat image index
                img_index = sequence * sequence_length + frame_num
                frame = cv2.imread(f'image/{action}/{img_index}.png')

                if frame is None:
                    print(f'Frame {img_index} is None (image/{action}/{img_index}.png)')  
                    continue

                image, results = mediapipe_detection(frame, hands)
                if results.multi_hand_landmarks:
                    draw_styles_landmarks(image, results)
                else:
                    print(f'No hand landmarks detected in frame {frame_num}')
                    # Still save zero keypoints or handle missing data? 
                    # Current logic skips saving if continue is called, which might be bad for sequence length.
                    # But the view_file showed 'continue' in a weird place in original code? 
                    # Let's stick to fixing syntax/path first.
                    pass

                draw_styles_landmarks(image, results)
                
                # display collection status
                message = f'Collecting data for {action} - Sequence {sequence} - Frame {frame_num}'
                cv2.putText(image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.imshow('Data Collection', image)

                # extract and save keypoints 
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints) 

                if cv2.waitKey(10) & 0xFF == ord('q'):
                   break
    
    cv2.destroyAllWindows()