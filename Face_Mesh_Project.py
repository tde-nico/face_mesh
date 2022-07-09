import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 1 ?

mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime = 0
cTime = 0

while 1:
	success, img = cap.read()
	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = face.process(imgRGB)

	lms = results.multi_face_landmarks
	if lms:
		for faceLms in lms:
			mpDraw.draw_landmarks(img, faceLms, mpFace.FACEMESH_CONTOURS,
				drawSpec, drawSpec)
			try:
				for id, lm in enumerate(faceLms.landmarks):
					h, w, c = img.shape
					x, y = int(lm.x * w), int(lm.y * h)
			except:
				pass

	cTime = time.time()
	fps = 1/(cTime - pTime)
	pTime = cTime

	cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

	cv2.imshow("Image", img)
	cv2.waitKey(1)
