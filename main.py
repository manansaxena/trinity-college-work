# import os
# os.system("git clone https://github.com/hukkelas/DSFD-Pytorch-Inference.git")
# os.system("python3 DSFD-Pytorch-Inference/setup.py install")

import streamlit as st 
import cv2
import face_detection
import numpy as np
import io
from PIL import Image
import zipfile


def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")

	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]

			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	# return the pixelated blurred image
	return image

def run_pixelate(image,method):
    detector = face_detection.build_detector(method, confidence_threshold=.5, nms_iou_threshold=.3)
    
    detections = detector.detect(image)
    
    for face in detections:
        if face[4] > 0.5:
            roi = image[int(face[1]):int(face[3]),int(face[0]):int(face[2])]
            roi = anonymize_face_pixelate(roi,blocks=4)
            image[int(face[1]):int(face[3]),int(face[0]):int(face[2])] = roi
    return image

def run():

    st.set_page_config(
        page_title="FacePixelate",
        page_icon="😶‍🌫️",
    )

    st.write("# Pixelate those faces on your photos")
    st.write('### Upload the photos')
    uploaded_files = st.file_uploader("# **Upload the photos**", accept_multiple_files=True, label_visibility="hidden")

    st.write("### Choose a face detection method")
    method = st.selectbox("Choose a face detection method",('DSFDDetector', 'RetinaNetResNet50', 'RetinaNetMobileNetV1'),index=None,
                           placeholder="Choose a face detection method...", label_visibility="hidden")
    st.write('You selected:', method)

    pixelated_images = []
    captions = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if uploaded_files and method:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for uploaded_file in uploaded_files:
                    photo = Image.open(uploaded_file).convert('RGB')
                    photo = np.array(photo)
                    # Assuming you have a function to process the photo
                    photo = run_pixelate(photo,method)
                    pixelated_images.append(photo)
                    photo = Image.fromarray(photo)
                    captions.append(uploaded_file.name)


                    buffered = io.BytesIO()
                    photo.save(buffered, format="JPEG")
                    buffered.seek(0)
                    zip_file.writestr(f"pixelated_{uploaded_file.name}", buffered.read())

            st.image(pixelated_images, caption=captions, width=250)

            zip_buffer.seek(0)
            st.download_button(
                label="Download All Pixelated Images",
                data=zip_buffer.getvalue(),
                file_name="all_pixelated_images.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    run()