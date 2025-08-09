# Real-Time Pain Monitoring System

This project is a sophisticated pain monitoring system designed to detect pain in patients, particularly those who may be unable to communicate verbally, such as paralyzed individuals. It uses computer vision to analyze facial expressions and body movements in real-time via a webcam feed.

The system employs a multi-agent architecture built with LangGraph, where specialized agents collaborate to analyze different aspects of the video feed, calculate a comprehensive pain score, and send SMS alerts via Twilio when the score exceeds a set threshold.

## Key Features

- **Advanced Facial Analysis:** Utilizes DeepFace for emotion detection and is architected to support over 15 granular facial pain indicators (e.g., brow furrow, eye closure) through an extensible facial landmark model.
- **Movement Detection:** Employs OpenCV-based background subtraction to detect significant hand and body movements, contributing to the overall pain assessment.
- **Multi-Agent Workflow:** A robust, modular system where different agents handle facial analysis, movement detection, pain scoring, and alerting, orchestrated by a LangGraph workflow.
- **Comprehensive Pain Scoring:** Aggregates data from facial and movement analysis to generate a normalized pain score from 0.0 to 1.0.
- **SMS Alerts:** Automatically sends alerts via Twilio when the pain score surpasses a configurable threshold.
- **Alert Cooldown System:** Prevents alert fatigue by enforcing a cooldown period between SMS notifications.
- **Real-time Webcam Processing:** Captures and analyzes video from a webcam in real-time, providing immediate feedback.
- **Environment-based Configuration:** All sensitive data and parameters (API keys, thresholds) are managed securely through a `.env` file.

## Tech Stack

- **Python 3.13+**
- **OpenCV:** For all core computer vision tasks (video capture, image processing).
- **DeepFace:** For high-level facial emotion analysis.
- **LangGraph:** For building the multi-agent workflow.
- **Twilio:** For sending SMS notifications.
- **NumPy:** For numerical operations.

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository**
```bash
git clone <repository-url>
cd pain-detection
```

**2. Create a Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies**
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables**
The system requires API keys for Twilio.
- Copy the example environment file:
  ```bash
  cp .env.example .env
  ```
- Open the `.env` file and replace the placeholder values with your actual credentials and desired settings.

**5. (Optional) Download Facial Landmark Model**
For the most advanced facial pain indicators (e.g., eye closure, brow furrowing), the system is designed to use a facial landmark model. The current implementation uses emotion as the primary driver, but you can enable more detailed analysis by downloading the LBF model.
- Download the model from [here](https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml).
- Place it in the root directory of the project.
- Uncomment the landmark detector loading code in `src/analysis/facial_features.py`.

## Running the Application

To start the pain monitoring system, run the `main.py` script from the root directory:

```bash
python -m src.main
```

- A window will appear showing your webcam feed, annotated with real-time analysis.
- Press the **'q'** key on the keyboard to stop the application.

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

```
├── src/
│   ├── agents/             # Contains individual nodes for the LangGraph workflow
│   │   ├── facial_analysis_agent.py
│   │   ├── movement_analysis_agent.py
│   │   ├── pain_scoring_agent.py
│   │   └── alerting_agent.py
│   ├── analysis/           # Core modules for computer vision analysis
│   │   ├── facial_features.py
│   │   └── movement_detection.py
│   ├── utils/              # Utility scripts, like configuration management
│   │   └── config.py
│   ├── main.py             # Main entry point for the application
│   └── workflow.py         # Defines the LangGraph state and graph
├── .env.example            # Example environment file
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Configuration Details

The `.env` file controls the application's behavior. Here are the key variables:

- `TWILIO_ACCOUNT_SID`: Your Twilio Account SID.
- `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token.
- `TWILIO_PHONE_NUMBER`: The Twilio phone number that will send the SMS.
- `TO_PHONE_NUMBER`: The phone number that will receive the alert.
- `PAIN_THRESHOLD`: The score (0.0-1.0) above which an alert is sent. Default is `0.7`.
- `ALERT_COOLDOWN_SECONDS`: The minimum time in seconds between alerts. Default is `300`.
- `WEBCAM_INDEX`: The index of the webcam to use. Default is `0`.
- `LOG_LEVEL`: The logging level (e.g., INFO, DEBUG). Default is `INFO`.
