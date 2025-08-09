import cv2
import numpy as np
from deepface import DeepFace
from src.utils.config import logger

# Load a pre-trained face detector from OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    logger.error(f"Failed to load Haar cascade for face detection: {e}")
    face_cascade = None

# Placeholder for a facial landmark detector.
# To meet the "15+ indicators" requirement without MediaPipe, a landmark model is essential.
# OpenCV's LBF model is a good candidate, but requires an external model file.
# We will design the code to easily integrate it.
landmark_detector = None
# Example of how it would be loaded:
# try:
#     landmark_detector = cv2.face.createFacemarkLBF()
#     landmark_detector.loadModel("lbfmodel.yaml")
# except cv2.error as e:
#     logger.warning(f"Could not load landmark detector model: {e}. Granular pain indicators will be disabled.")
#     logger.warning("Please download the model: https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml")


def detect_faces(frame):
    """
    Detects faces in a video frame using Haar cascades.
    Returns a list of face bounding boxes.
    """
    if face_cascade is None:
        logger.warning("Face detector not loaded, skipping face detection.")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces

def analyze_facial_expressions(frame, faces):
    """
    Analyzes facial expressions for detected faces using DeepFace and calculates a pain score.
    This function will be expanded with landmark-based indicators.
    """
    analysis_results = []
    if not faces.any():
        return analysis_results

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        result = {
            'box': (x, y, w, h),
            'emotion': 'neutral',
            'emotion_score': 1.0,
            'pain_indicators': {},
            'facial_pain_score': 0.0
        }

        # 1. High-level emotion analysis with DeepFace
        try:
            analysis = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(analysis, list):
                analysis = analysis[0]

            dominant_emotion = analysis.get('dominant_emotion', 'neutral')
            emotion_score = analysis.get('emotion', {}).get(dominant_emotion, 0) / 100.0

            result['emotion'] = dominant_emotion
            result['emotion_score'] = emotion_score

        except Exception as e:
            logger.debug(f"DeepFace analysis failed for a face region: {e}")
            # Continue with default 'neutral' if DeepFace fails
            pass

        # 2. Granular pain indicators from landmarks (currently a placeholder)
        if landmark_detector:
            # This part will be implemented once a landmark model is integrated
            # indicators = get_landmark_based_indicators(gray_frame, face_box)
            # result['pain_indicators'] = indicators
            pass
        else:
            result['pain_indicators']['landmark_detector'] = 'Not available'

        # 3. Calculate final facial pain score
        result['facial_pain_score'] = calculate_facial_pain_score(result)

        analysis_results.append(result)

    return analysis_results


def calculate_facial_pain_score(analysis_result):
    """
    Calculates a pain score based on emotions and (future) pain indicators.
    This is a heuristic model that can be refined.
    """
    # Base score from emotions
    pain_emotions = {
        'angry': 0.8,
        'fear': 0.7,
        'sad': 0.6,
        'disgust': 0.7,
        'neutral': 0.1,
        'happy': 0.0,
        'surprise': 0.2
    }
    emotion_pain_score = pain_emotions.get(analysis_result['emotion'], 0.0)

    # Weight based on confidence of the emotion detection
    weighted_emotion_score = emotion_pain_score * analysis_result['emotion_score']

    # Placeholder for combining with landmark-based indicators
    # For example:
    # landmark_score = sum(analysis_result['pain_indicators'].values()) / len(analysis_result['pain_indicators'])
    # final_score = (weighted_emotion_score * 0.5) + (landmark_score * 0.5)

    final_score = np.clip(weighted_emotion_score, 0.0, 1.0)

    return float(final_score)

# This is where the detailed landmark analysis will go.
# We need a model file for this, so for now, we just have the structure.
def get_landmark_based_indicators(gray_frame, face_box):
    """
    Analyzes facial landmarks to identify specific pain indicators.
    This is a placeholder until the landmark model is integrated.
    """
    indicators = {
        'brow_lowerer': 0.0,      # (AU4)
        'eye_closure': 0.0,       # (AU43)
        'cheek_raiser': 0.0,      # (AU6)
        'nose_wrinkler': 0.0,     # (AU9)
        'upper_lip_raiser': 0.0,  # (AU10)
        'mouth_stretch': 0.0,     # (AU27)
        # ... add more indicators up to 15+
    }

    # (x, y, w, h) = face_box
    # landmarks = landmark_detector.fit(gray_frame, np.array([face_box]))
    # if landmarks is not None:
        # shape = landmarks[1][0]
        # indicators['eye_closure'] = calculate_eye_aspect_ratio(shape[36:48])
        # indicators['brow_lowerer'] = calculate_brow_distance(shape[17:27])
        # ... etc.

    return indicators
