from src.utils.config import logger
from src.analysis.movement_detection import MovementDetector

def movement_analysis_node(state):
    """
    A LangGraph node that performs movement analysis on a video frame.

    This node uses a stateful MovementDetector instance passed in the state
    to detect significant movement and updates the state with its findings.

    Args:
        state (dict): The current state of the graph. It is expected to contain:
                      - 'frame': The current video frame.
                      - 'movement_detector': An instance of the MovementDetector class.

    Returns:
        dict: A dictionary with the results of the analysis:
              - 'movement_score': A float score (0.0-1.0) of detected movement.
              - 'movement_mask': The binary mask showing moving regions.
              - 'movement_contours': The contours of moving objects.
    """
    logger.info("Executing movement analysis node...")
    frame = state.get("frame")
    movement_detector = state.get("movement_detector")

    if frame is None:
        logger.warning("No frame provided for movement analysis. Skipping.")
        return {}

    if not isinstance(movement_detector, MovementDetector):
        logger.error("MovementDetector instance not found or invalid in state. Skipping analysis.")
        # Return default values to prevent the graph from crashing
        return {"movement_score": 0.0, "movement_mask": None, "movement_contours": []}

    try:
        score, mask, contours = movement_detector.detect(frame)

        if score > 0.05:  # Log only if movement is non-trivial
            logger.info(f"Movement detected with score: {score:.2f}")

        return {
            "movement_score": score,
            "movement_mask": mask,
            "movement_contours": contours
        }

    except Exception as e:
        logger.error(f"An error occurred in movement analysis node: {e}", exc_info=True)
        return {"movement_score": 0.0, "movement_mask": None, "movement_contours": []}
