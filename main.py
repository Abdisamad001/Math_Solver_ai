import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

##
st.set_page_config(layout="wide")
st.image('solver.png')

col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize generative AI model
genai.configure(api_key="AIzaSyADcXEWrBonf51xuZvUetC_CXBJ-Opi6Y0")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)  # Try using device index 0 for default webcam
cap.set(3, 1280)  # Set width
cap.set(4, 720)  # Set height

# Check if the webcam opened successfully
if not cap.isOpened():
    st.error("Failed to access the webcam. Please check if it's being used by another app.")
    st.stop()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up (drawing mode)
        current_pos = lmList[8][0:2]  # Get the index finger tip position
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (clear screen)
        canvas = np.zeros_like(canvas)
    return current_pos, canvas


def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Specific hand gesture to trigger AI model
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return ""


prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Main loop to capture video and process frames
while run:
    success, img = cap.read()

    # Check if webcam is capturing the image successfully
    if not success or img is None:
        st.error("Failed to capture image from webcam. Please ensure the webcam is working properly.")
        print("Camera capture failed. Success:", success)
        break  # Exit loop if camera capture fails

    img = cv2.flip(img, 1)  # Flip image horizontally for a mirror view

    if canvas is None:
        canvas = np.zeros_like(img)  # Initialize the drawing canvas

    info = getHandInfo(img)  # Get hand info (fingers up)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Draw on canvas based on hand position
        output_text = sendToAI(model, canvas, fingers)  # Send to AI model if correct gesture

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)  # Overlay the drawing on the webcam image
    FRAME_WINDOW.image(image_combined, channels="BGR")  # Display image in Streamlit

    if output_text:
        output_text_area.text(output_text)  # Display AI response

    cv2.waitKey(1)  # Small delay to allow frame updates

# Release the webcam when the app is closed
#cap.release()
#cv2.destroyAllWindows()
