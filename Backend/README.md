# Stutter Detection

This project is an AI-powered tool designed to analyze audio files and detect speech patterns related to stuttering. It leverages advanced audio processing and transcription techniques to provide accurate and insightful analysis.

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Audio Processing**: Extracts features from audio files for analysis.
- **Stutter Detection**: Identifies speech patterns associated with stuttering.
- **Visualizations**: Displays results in a user-friendly manner.
- **Extensible Architecture**: Modular codebase for easy updates and enhancements.

## Directory Structure

```
.
├── analysis_results_YYYYMMDD_HHMMSS/   # Stores analysis outputs
├── src/                               # Source code
│   ├── audio/                         # Audio processing modules
│   │   ├── audio_config.py
│   │   ├── audio_recorder.py
│   │   ├── feature_extractor.py
│   │   ├── processor.py
│   │   ├── stutter_detector.py
│   │   └── transcription_analyzer.py
│   ├── utils/                         # Utility scripts
│   │   ├── audio_utils.py
│   └── visualization/                 # Visualization scripts
│       ├── speech_visualizer.py
├── tests/                             # Test audio files
│   └── test.mp3
├── venv/                              # Virtual environment (ignored by Git)
├── .gitignore                         # Specifies untracked files
├── main.py                            # Entry point for the application
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stutter-detection.git
   cd stutter-detection
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```bash
   python main.py --file_path tests/test.mp3
   ```

2. View the analysis results in the `analysis_results_YYYYMMDD_HHMMSS` folder.

## Contributing

Contributions are welcome!
