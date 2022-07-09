import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
	def __init__(self, static_mode=False, max_faces=2, refine_lms=True,
			min_detection_con=0.5, min_track_con=0.5):
		self.static_mode = static_mode
		self.max_faces = max_faces
		self.refine_lms = refine_lms
		self.min_detection_con = min_detection_con
		self.min_track_con = min_track_con
		self.mpFace = mp.solutions.face_mesh
		self.face = self.mpFace.FaceMesh(static_mode, max_faces,
			refine_lms, min_detection_con, min_track_con)
		self.mpDraw = mp.solutions.drawing_utils
		self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


	def find_face(self, img, draw=True):
		lm_list = []
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		results = self.face.process(imgRGB)
		lms = results.multi_face_landmarks
		if lms:
			for faceLms in lms:
				face = []
				if draw:
					self.mpDraw.draw_landmarks(img, faceLms, 
						self.mpFace.FACEMESH_CONTOURS,
						self.drawSpec, self.drawSpec)
				try:
					for id, lm in enumerate(faceLms.landmarks):
						h, w, c = img.shape
						x, y = int(lm.x * w), int(lm.y * h)
						face.append([id, x, y])
					lm_list.append(face)
				except:
					pass
		return img, lm_list


def main():
	cap = cv2.VideoCapture(0) # 1 ?
	detector = FaceMeshDetector()
	pTime = 0
	cTime = 0

	while 1:
		success, img = cap.read()
		img, lm_list = detector.find_face(img)

		cTime = time.time()
		fps = 1/(cTime - pTime)
		pTime = cTime

		cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == '__main__':
	main()
