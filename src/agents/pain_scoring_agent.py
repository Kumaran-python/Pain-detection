from src.utils.config import logger
import numpy as np

# Define weights for combining the different analysis scores.
# These could be externalized to the config file for easier tuning.
FACIAL_SCORE_WEIGHT = 0.70
MOVEMENT_SCORE_WEIGHT = 0.30

def pain_scoring_node(state):
    """
    A LangGraph node that calculates a comprehensive pain score.

    This node aggregates the outputs from the facial and movement analysis
    nodes, weights them, and produces a final, unified pain score.

    Args:
        state (dict): The current state of the graph. It should contain:
                      - 'facial_analysis_results': Output from the facial node.
                      - 'movement_score': Output from the movement node.

    Returns:
        dict: A dictionary containing the 'final_pain_score'.
    """
    logger.info("Executing pain scoring node...")

    facial_results = state.get("facial_analysis_results", [])
    movement_score = state.get("movement_score", 0.0)

    # --- Calculate the contribution from facial analysis ---
    # If multiple faces are detected, we take the highest pain score among them.
    # This assumes that if one person is in pain, the system should react.
    max_facial_pain_score = 0.0
    if facial_results:
        # Create a list of all facial pain scores from the results
        all_facial_scores = [res.get("facial_pain_score", 0.0) for res in facial_results]
        if all_facial_scores:
            max_facial_pain_score = max(all_facial_scores)

    logger.debug(f"  - Max facial pain score: {max_facial_pain_score:.2f}")
    logger.debug(f"  - Movement score: {movement_score:.2f}")

    # --- Combine scores using the defined weights ---
    final_score = (
        (max_facial_pain_score * FACIAL_SCORE_WEIGHT) +
        (movement_score * MOVEMENT_SCORE_WEIGHT)
    )

    # Clip the final score to ensure it remains within the [0.0, 1.0] range.
    final_score_clipped = np.clip(final_score, 0.0, 1.0)

    logger.info(f"Pain scoring complete. Final score: {final_score_clipped:.2f}")

    return {"final_pain_score": float(final_score_clipped)}
