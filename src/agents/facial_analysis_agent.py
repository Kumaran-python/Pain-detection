from src.utils.config import logger
from src.analysis.facial_features import detect_faces, analyze_facial_expressions

def facial_analysis_node(state):
    """
    A LangGraph node that performs facial analysis on a video frame.

    This node detects faces, analyzes their expressions for emotions and pain
    indicators, and updates the state with the results.

    Args:
        state (dict): The current state of the graph. It must contain a 'frame'.

    Returns:
        dict: A dictionary with the key 'facial_analysis_results' containing
              a list of analyses for each detected face.
    """
    logger.info("Executing facial analysis node...")
    frame = state.get("frame")

    if frame is None:
        logger.warning("No frame available in the state for facial analysis.")
        return {"facial_analysis_results": []}

    try:
        # Detect all faces in the frame
        detected_faces = detect_faces(frame)

        if not detected_faces.any():
            logger.info("No faces detected in the frame.")
            return {"facial_analysis_results": []}

        # Analyze the expressions of the detected faces
        analysis_results = analyze_facial_expressions(frame, detected_faces)
        logger.info(f"Facial analysis complete. Found {len(analysis_results)} face(s).")

        # Log the dominant emotion of the first detected face for debugging
        if analysis_results:
            logger.debug(f"  - Dominant emotion (face 0): {analysis_results[0]['emotion']}")
            logger.debug(f"  - Facial pain score (face 0): {analysis_results[0]['facial_pain_score']:.2f}")

        return {"facial_analysis_results": analysis_results}

    except Exception as e:
        logger.error(f"An error occurred during facial analysis: {e}", exc_info=True)
        # Return an empty list to allow the workflow to continue gracefully
        return {"facial_analysis_results": []}
