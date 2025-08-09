from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from src.agents.facial_analysis_agent import facial_analysis_node
from src.agents.movement_analysis_agent import movement_analysis_node
from src.agents.pain_scoring_agent import pain_scoring_node
from src.agents.alerting_agent import alerting_node
from src.analysis.movement_detection import MovementDetector
from src.utils.config import logger

class PainAnalysisState(TypedDict, total=False):
    """
    Defines the state for the pain analysis workflow.
    `total=False` means keys are optional, which is useful in LangGraph
    as nodes progressively populate the state.

    Attributes:
        frame: The current video frame (np.ndarray).
        movement_detector: The stateful MovementDetector instance.
        last_alert_time: The timestamp of the last alert sent.
        facial_analysis_results: A list of dicts from the facial analysis node.
        movement_score: A float score from the movement analysis node.
        movement_mask: The binary mask from the movement analysis node.
        movement_contours: Contours of moving objects.
        final_pain_score: The final aggregated pain score.
    """
    frame: Any
    movement_detector: MovementDetector
    last_alert_time: float
    facial_analysis_results: List[dict]
    movement_score: float
    movement_mask: Any
    movement_contours: List[Any]
    final_pain_score: float


def create_pain_monitoring_workflow():
    """
    Builds and compiles the LangGraph workflow for pain monitoring.

    The workflow follows a sequential process for clarity and robustness:
    1. Facial Analysis
    2. Movement Analysis
    3. Pain Scoring (Aggregation)
    4. Alerting (Conditional)

    Returns:
        A compiled LangGraph runnable.
    """
    workflow = StateGraph(PainAnalysisState)

    # Add the four main nodes to the graph
    logger.info("Adding nodes to the workflow graph...")
    workflow.add_node("facial_analysis", facial_analysis_node)
    workflow.add_node("movement_analysis", movement_analysis_node)
    workflow.add_node("pain_scoring", pain_scoring_node)
    workflow.add_node("alerting", alerting_node)

    # Define the execution flow (edges)
    # The flow is sequential for simplicity and predictable execution.
    logger.info("Defining workflow edges...")
    workflow.set_entry_point("facial_analysis")
    workflow.add_edge("facial_analysis", "movement_analysis")
    workflow.add_edge("movement_analysis", "pain_scoring")
    workflow.add_edge("pain_scoring", "alerting")

    # After the alerting node, the process for the current frame ends.
    workflow.add_edge("alerting", END)

    logger.info("Compiling the pain monitoring workflow...")
    # The compiled graph is a runnable that can process state objects.
    app = workflow.compile()
    logger.info("Workflow compiled successfully.")

    return app
