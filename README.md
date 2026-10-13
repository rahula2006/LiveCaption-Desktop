
# LiveCaption Desktop 🎙️

A lightweight Python desktop application that captures system audio and transcribes spoken content into text using local Whisper speech recognition.

## 💡 Motivation

While watching recorded lectures and educational videos, I wanted a convenient way to generate transcripts without manually extracting audio from video files.

LiveCaption Desktop explores a solution using system-audio loopback capture, a floating graphical interface, and locally executed speech recognition.

## ✨ Features

- **Floating interface:** An always-on-top desktop window.
- **System-audio capture:** Captures audio playing through the selected output device.
- **AI transcription:** Uses faster-whisper to convert speech into text.
- **Live text display:** Displays recognized speech in the application.
- **Text export:** Saves transcripts as plain-text files.
- **Silence-based stopping:** Optional stopping after a period of silence.

## 🛠️ Technology Stack

- Python
- Tkinter
- SoundCard
- NumPy
- faster-whisper

## ⚙️ Installation

### Prerequisites

- Python 3.10 or newer recommended.
- A compatible audio output device.
- An internet connection for the initial Whisper model download.

### Install dependencies

Clone or download this repository, open a terminal in the project directory, and run:

```bash
python -m pip install -r requirements.txt
```

### Run the application

```bash
python floating_transcriber.py
```

## 🚀 Usage

1. Launch the application.
2. Click **START**.
3. Play a video or recorded lecture on your computer.
4. View the recognized speech in the floating window.
5. Click **STOP & SAVE** when finished.
6. Find the automatically saved transcript in the `transcripts` folder.

You can also use **Save As** to save the displayed text to a chosen location.

## 🔒 Privacy

The application uses local Whisper inference for speech recognition. After the required model has been downloaded, transcription can run offline.

Audio-device compatibility depends on the operating system, audio drivers, and selected output device.

## ⚠️ Current Limitations

This project is an early prototype.

- Transcription may lag behind playback on slower computers.
- Speech recognition accuracy depends on audio quality, language, and background noise.
- Timestamps are currently relative to individual audio chunks rather than the complete recording.
- Silence detection may stop the session during long pauses.
- System-audio capture has not been verified across all devices and operating systems.

## 🗺️ Future Improvements

- Accurate cumulative timestamps.
- SRT subtitle export.
- Pause and resume controls.
- Language selection.
- Improved audio buffering and error handling.
- Better compatibility with different audio devices.
- Windows executable packaging.

## 📄 License

This project is intended as an open-source learning and development project. A license can be added when the repository's licensing terms are finalized.
