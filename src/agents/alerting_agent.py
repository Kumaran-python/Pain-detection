import time
from src.utils import config
from src.utils.config import logger

# We conditionally import twilio to allow the rest of the system to run
# without it for testing purposes.
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    logger.warning("Twilio library not found. SMS alerting will be disabled.")
    TWILIO_AVAILABLE = False
    Client = None
    TwilioRestException = None


def alerting_node(state):
    """
    A LangGraph node that sends an SMS alert if the pain score is high.

    This node checks the final pain score against a configurable threshold.
    If the score is high, it checks a cooldown timer. If the cooldown has
    passed, it sends an alert via Twilio and resets the timer.

    Args:
        state (dict): The current state of the graph. It should contain:
                      - 'final_pain_score': The aggregated pain score.
                      - 'last_alert_time': Timestamp of the last alert sent.

    Returns:
        dict: A dictionary containing the updated 'last_alert_time' if an
              alert was sent. Otherwise, an empty dictionary.
    """
    logger.info("Executing alerting node...")

    final_pain_score = state.get("final_pain_score", 0.0)
    last_alert_time = state.get("last_alert_time", 0)

    pain_threshold = config.PAIN_THRESHOLD
    cooldown_seconds = config.ALERT_COOLDOWN_SECONDS

    # 1. Check if the pain score is above the threshold
    is_above_threshold = final_pain_score >= pain_threshold

    # 2. Check if the cooldown period has elapsed
    time_since_last_alert = time.time() - last_alert_time
    is_cooldown_over = time_since_last_alert > cooldown_seconds

    if is_above_threshold and is_cooldown_over:
        logger.warning(f"Pain score {final_pain_score:.2f} exceeds threshold of {pain_threshold}. Sending alert.")

        if not TWILIO_AVAILABLE:
            logger.error("Cannot send alert: Twilio library is not installed.")
            return {}

        try:
            # Validate config before trying to send
            config.validate_config()

            client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
            message_body = f"Pain Monitoring Alert: A high pain score of {final_pain_score:.2f} has been detected for the patient."

            message = client.messages.create(
                body=message_body,
                from_=config.TWILIO_PHONE_NUMBER,
                to=config.TO_PHONE_NUMBER
            )

            logger.info(f"SMS alert sent successfully! SID: {message.sid}")

            # Return the new alert time to update the state
            return {"last_alert_time": time.time()}

        except (TwilioRestException, ValueError) as e:
            logger.error(f"Failed to send Twilio SMS alert: {e}", exc_info=True)
            # We don't update the alert time, so the system can try again on the next cycle.
            return {}

    elif is_above_threshold and not is_cooldown_over:
        remaining_cooldown = cooldown_seconds - time_since_last_alert
        logger.info(f"Pain score ({final_pain_score:.2f}) is high, but in cooldown. "
                    f"{remaining_cooldown:.0f}s remaining. No alert sent.")

    # If no alert is sent, no state change is needed.
    return {}
