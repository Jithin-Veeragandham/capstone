import streamlit as st
import cv2
import torch

import pandas as pd
model = torch.hub.load('C:\\Users\\jithi\\OneDrive\\Desktop\\VsCode\\yolov7-main\\yolov7-main', 'custom', "C:\\Users\\jithi\\Downloads\\yolov7.pt",force_reload=True, source='local',trust_repo=True)
def main():
    st.title("Object Detection")
    video_path = st.text_input("Enter the path to the video file:", "")

    is_person = st.checkbox("Person")
    is_car = st.checkbox("Car")
    is_truck = st.checkbox("Truck")
    is_dog = st.checkbox("Dog")

    selected_classes = []
    if is_person: selected_classes.append('person')
    if is_car: selected_classes.append('car')
    if is_truck: selected_classes.append('truck')
    if is_dog: selected_classes.append('dog')

    if st.button("Submit"):
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform inference using your model (replace this with your actual model)
                results = model(frame)
                
                # Convert results to pandas DataFrame
                df = results.pandas().xyxy[0]

                for index, row in df.iterrows():
                    class_name = row['name']
                    if class_name in selected_classes:
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Video with Bounding Boxes', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            st.warning("Please enter a valid video path.")

if __name__ == "__main__":
    main()
