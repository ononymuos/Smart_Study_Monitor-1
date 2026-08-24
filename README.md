# Smart Study Monitor

An AI-powered desktop application to help you stay focused during study sessions. It detects sleepiness, phone usage, and absence from the desk using your webcam, and sounds an alarm when you get distracted.

### Features
- **Micro-Sleep Detection:** Tracks eye-aspect ratio. Now improved with 3D Head-Pose Pitch Estimation so it doesn't trigger when you look down to read or write notes.
- **Phone Detection:** Uses YOLOv8 to detect if you pick up your phone while studying.
- **Absence Detection:** Alerts you if you leave the desk or cover your face for too long.
- **Instant Audio Cutoffs:** Alarms stop immediately once the distraction condition is resolved.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Smart_Study_Monitor.git
   cd Smart_Study_Monitor
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   # We recommend using uv or standard venv
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

## Customizing Alarms

To change the alarm sounds, simply replace the following MP3 files in the main folder with your own audio files (you must keep the exact same filenames):
- `alarm.mp3` — Plays when sleep is detected.
- `faudio.mp3` — Plays when your face is hidden or away from the camera.
- `paudio.mp3` — Plays when a phone is detected.

## Acknowledgements
Based on the original prototype by [@SwastikBiswas26](https://github.com/SwastikBiswas26/Smart_Study_Monitor). Enhanced with smart pitch-estimation and rapid-response audio controls.

## License
This project is licensed under the MIT License.
