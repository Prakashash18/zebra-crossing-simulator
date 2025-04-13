# Zebra Crossing Simulator for Kids

An interactive web-based simulator that uses pose detection to teach children how to safely cross roads.

## Features

- Real-time pose detection using a pre-trained PyTorch model
- Interactive step-by-step guidance
- Gamification elements (points, achievements, progress tracking)
- Voice instructions and sound effects
- Animated traffic light and character visualizations

## Supported Poses

The simulator can detect six key poses:
1. Looking Left
2. Looking Right
3. Looking Straight
4. Looking Down
5. Raise Right Hand
6. Raise Left Hand

## Requirements

- Python 3.7+
- Webcam
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/zebra-crossing-simulator.git
cd zebra-crossing-simulator
```

2. Create a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Grant camera permissions when prompted
4. Follow the on-screen instructions to practice safe road crossing

## Audio Files

For a better experience, you can add sound effects and voice prompts. Place audio files in `app/static/audio/` with the following names:
- `correct.mp3` - Sound played when a correct pose is detected
- `incorrect.mp3` - Sound played when an incorrect pose is detected
- `complete.mp3` - Sound played when the crossing is complete
- `traffic.mp3` - Background traffic ambient sound
- `crossing_beep.mp3` - Pedestrian crossing beeping sound

## Images

Add a zebra crossing image to `app/static/images/zebra-crossing.png` for the home page.

## License

MIT 