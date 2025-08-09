import cv2
import time
from src.utils import config
from src.utils.config import logger
from src.workflow import create_pain_monitoring_workflow, PainAnalysisState
from src.analysis.movement_detection import MovementDetector
import numpy as np

def draw_results_on_frame(frame: np.ndarray, state: PainAnalysisState):
    """
    Annotates the video frame with the results from the analysis workflow.

    Args:
        frame: The original video frame to draw on.
        state: The final state from the LangGraph workflow containing all analysis data.

    Returns:
        The annotated frame.
    """
    # Draw movement contours first, so they are in the background
    movement_contours = state.get("movement_contours", [])
    if movement_contours:
        cv2.drawContours(frame, movement_contours, -1, (0, 255, 0), 2)

    # Draw facial analysis results
    facial_results = state.get("facial_analysis_results", [])
    for result in facial_results:
        x, y, w, h = result['box']
        # Draw bounding box for the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)  # Orange box

        # Prepare text for emotion and facial score
        emotion = result.get('emotion', 'N/A')
        facial_score = result.get('facial_pain_score', 0.0)
        info_text = f"{emotion.capitalize()} ({facial_score:.2f})"

        # Put text above the bounding box
        cv2.putText(frame, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # Display the final aggregated pain score
    final_score = state.get("final_pain_score", 0.0)
    score_text = f"OVERALL PAIN SCORE: {final_score:.2f}"

    # Determine text color based on severity
    if final_score >= config.PAIN_THRESHOLD:
        text_color = (0, 0, 255)  # Red for high alert
    elif final_score >= config.PAIN_THRESHOLD / 2:
        text_color = (0, 255, 255)  # Yellow for caution
    else:
        text_color = (0, 255, 0)  # Green for normal

    # Add a semi-transparent background for the score text for better visibility
    (w, h), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)
    cv2.rectangle(frame, (10, 10), (10 + w + 10, 10 + h + 10), (0,0,0), -1)
    cv2.putText(frame, score_text, (20, 20 + h), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 3)

    return frame


def main():
    """
    The main entry point for the Pain Monitoring System.
    """
    logger.info("--- Starting Pain Monitoring System ---")

    # 1. Initialize core components
    app = create_pain_monitoring_workflow()
    movement_detector = MovementDetector()

    logger.info(f"Opening webcam at index: {config.WEBCAM_INDEX}")
    cap = cv2.VideoCapture(config.WEBCAM_INDEX)
    if not cap.isOpened():
        logger.error(f"Fatal: Could not open webcam at index {config.WEBCAM_INDEX}.")
        return

    # 2. Setup the initial state for the workflow graph
    # This state object will be updated and passed through the workflow on each frame.
    initial_state = PainAnalysisState(
        movement_detector=movement_detector,
        last_alert_time=0.0  # Initialize to 0 to allow an immediate first alert
    )

    logger.info("Entering main processing loop. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame from webcam. End of stream?")
            break

        # The frame needs to be flipped as webcams are often mirrored
        frame = cv2.flip(frame, 1)

        # 3. Execute the workflow for the current frame
        # The input to invoke is the current state, including the new frame.
        current_input_state = {**initial_state, "frame": frame}
        final_state = app.invoke(current_input_state)

        # Persist the last alert time for the next iteration.
        if 'last_alert_time' in final_state:
            initial_state['last_alert_time'] = final_state['last_alert_time']

        # 4. Visualize the results
        annotated_frame = draw_results_on_frame(frame, final_state)
        cv2.imshow('Pain Monitoring System', annotated_frame)

        # 5. Check for exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("'q' key pressed. Shutting down.")
            break

    # 6. Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    logger.info("--- System shut down successfully ---")

if __name__ == "__main__":
    main()
