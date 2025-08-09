import cv2
import numpy as np
from src.utils.config import logger

class MovementDetector:
    """
    Detects movement in a video stream using background subtraction (MOG2).

    This class maintains a running model of the background and identifies
    foreground objects (i.e., movement). It's designed to be stateful,
    processing one frame at a time.
    """

    def __init__(self, history=150, threshold=40, min_contour_area=500, scaling_factor=10):
        """
        Initializes the MovementDetector.

        Args:
            history (int): The number of last frames that affect the background model.
            threshold (int): Threshold on the squared Mahalanobis distance to decide
                             if a pixel is part of the background. A lower value detects
                             more subtle changes.
            min_contour_area (int): The minimum pixel area of a contour to be
                                    considered significant movement.
            scaling_factor (int): A factor to amplify the movement score, since movement
                                  might only cover a small portion of the frame.
        """
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=True  # We'll filter out shadows manually
        )
        self.min_contour_area = min_contour_area
        self.scaling_factor = scaling_factor
        # Kernel for morphological operations to clean up noise from the mask
        self.kernel = np.ones((5, 5), np.uint8)
        logger.info("MovementDetector initialized.")

    def detect(self, frame):
        """
        Processes a single frame to detect movement.

        Args:
            frame: The input video frame (in BGR format).

        Returns:
            A tuple containing:
            - movement_score (float): A score from 0.0 to 1.0 indicating the
                                      amount of significant movement.
            - mask (np.array): The binary foreground mask for visualization.
            - contours (list): A list of contours for the detected moving objects.
        """
        # 1. Apply the background subtractor to get the foreground mask.
        # This mask includes shadows (marked with value 127).
        fg_mask = self.bg_subtractor.apply(frame)

        # 2. Clean the mask.
        # We only want definite foreground, so we threshold to remove shadows.
        _, binary_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

        # Use morphological opening (erosion followed by dilation) to remove noise.
        binary_mask = cv2.erode(binary_mask, self.kernel, iterations=1)
        binary_mask = cv2.dilate(binary_mask, self.kernel, iterations=2)

        # 3. Find contours of the moving objects.
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Calculate the movement score based on the area of significant contours.
        total_movement_area = 0
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                total_movement_area += area
                significant_contours.append(contour)

        # 5. Normalize the score.
        frame_area = frame.shape[0] * frame.shape[1]
        if frame_area == 0:
            return 0.0, binary_mask, []

        # The raw score is the ratio of moving area to total area.
        # We scale it up to make it more sensitive to smaller but important movements.
        movement_score = (total_movement_area / frame_area) * self.scaling_factor

        # Clip the score to ensure it stays within the 0.0 to 1.0 range.
        movement_score = np.clip(movement_score, 0.0, 1.0)

        return float(movement_score), binary_mask, significant_contours
