import os
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Twilio Configuration ---
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TO_PHONE_NUMBER = os.getenv("TO_PHONE_NUMBER")

# --- Application Configuration ---
# Default to 0.7 if not set or invalid
try:
    PAIN_THRESHOLD = float(os.getenv("PAIN_THRESHOLD", "0.7"))
except (ValueError, TypeError):
    PAIN_THRESHOLD = 0.7

# Default to 300 seconds (5 minutes) if not set or invalid
try:
    ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))
except (ValueError, TypeError):
    ALERT_COOLDOWN_SECONDS = 300

# Default to webcam 0 if not set or invalid
try:
    WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "0"))
except (ValueError, TypeError):
    WEBCAM_INDEX = 0

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def validate_config():
    """
    Validates that essential configurations are set and not using placeholder values.
    Raises ValueError if configuration is missing or invalid.
    """
    # Check for placeholder values in Twilio credentials
    if not TWILIO_ACCOUNT_SID or "ACx" in TWILIO_ACCOUNT_SID:
        raise ValueError("TWILIO_ACCOUNT_SID is not configured. Please set it in your .env file.")

    if not TWILIO_AUTH_TOKEN or "your_auth_token" in TWILIO_AUTH_TOKEN:
        raise ValueError("TWILIO_AUTH_TOKEN is not configured. Please set it in your .env file.")

    if not TWILIO_PHONE_NUMBER:
        raise ValueError("TWILIO_PHONE_NUMBER is not configured. Please set it in your .env file.")

    if not TO_PHONE_NUMBER:
        raise ValueError("TO_PHONE_NUMBER is not configured. Please set it in your .env file.")

    logger.info("Twilio configuration appears to be valid.")

# The application can call this function at startup to ensure all critical env vars are set.
# Example:
# if __name__ == '__main__':
#     try:
#         validate_config()
#         logger.info("Configuration loaded and validated successfully.")
#     except ValueError as e:
#         logger.error(f"Configuration Error: {e}")
