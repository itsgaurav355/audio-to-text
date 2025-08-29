# Audio Transcriber with Question Extraction

This Python script transcribes audio files using OpenAI's Whisper model with GPU acceleration and extracts only the questions from the transcription.

## Features

- 🎵 **Audio Transcription**: Supports various audio formats (MP3, WAV, M4A, etc.)
- 🚀 **GPU Acceleration**: Automatically detects and uses CUDA GPU for faster processing
- 🔍 **Question Extraction**: Intelligently identifies and extracts questions from transcribed text
- 💾 **Output Options**: Saves questions to text files with customizable naming
- 🎛️ **Flexible Models**: Choose from different Whisper model sizes (tiny, base, small, medium, large)

## Prerequisites

### System Requirements
- Python 3.7 or higher
- CUDA-compatible GPU (optional, for acceleration)
- FFmpeg (for audio processing)

### FFmpeg Installation

**Windows:**
1. Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to a folder (e.g., `C:\ffmpeg`)
3. Add `C:\ffmpeg\bin` to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python audio_transcriber.py --help
```

## ⚠️ **CRITICAL: Environment Activation**

**For GPU detection to work properly, you MUST activate your virtual environment first:**

```bash
# Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# Windows (Command Prompt)
.\myenv\Scripts\activate.bat

# Then run the script
python audio_transcriber.py audio.mp3
```

**Alternative: Use the provided batch file**
```bash
# This automatically uses myenv Python
run_with_myenv.bat audio.mp3
```

**Why this matters:**
- The script needs CUDA-enabled PyTorch to detect your GPU
- Global Python installations often don't have CUDA support
- Your `myenv` environment has the correct CUDA-enabled PyTorch

## Usage

### Basic Usage

```bash
# Transcribe audio with default settings
python audio_transcriber.py audio.mp3

# Transcribe with specific model
python audio_transcriber.py audio.wav --model large

# Force CPU usage
python audio_transcriber.py audio.m4a --no-gpu
```

### Advanced Usage

```bash
# Specify output file
python audio_transcriber.py audio.mp3 --output my_questions.txt

# Use large model for better accuracy
python audio_transcriber.py audio.wav --model large --output detailed_questions.txt

# Show full transcription (verbose mode)
python audio_transcriber.py audio.mp3 --verbose

# Combine options
python audio_transcriber.py audio.wav --model medium --output questions.txt --verbose
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio_file` | Path to audio file (required) | - |
| `--model, -m` | Whisper model size: tiny, base, small, medium, large | base |
| `--output, -o` | Output file path | questions_<audio_name>.txt |
| `--no-gpu` | Force CPU usage | False |
| `--verbose, -v` | Show full transcription | False |

## Model Sizes

| Model | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|-------|------------|--------------|--------------|---------------|----------------|
| tiny | 39 M | ✓ | ✓ | ~1 GB | ~32x |
| base | 74 M | ✓ | ✓ | ~1 GB | ~16x |
| small | 244 M | ✓ | ✓ | ~2 GB | ~6x |
| medium | 769 M | ✓ | ✓ | ~5 GB | ~2x |
| large | 1550 M | ✓ | ✓ | ~10 GB | 1x |

## Question Extraction Logic

The script identifies questions using two methods:

1. **Question Mark Detection**: Sentences ending with "?"
2. **Interrogative Word Detection**: Sentences starting with question words like:
   - What, When, Where, Who, Why, How
   - Can, Could, Would, Will, Should
   - Do, Does, Did, Is, Are, Was, Were

## Supported Audio Formats

- MP3, MP4, M4A
- WAV, FLAC, OGG
- AVI, MOV (video files with audio)
- And many more (thanks to FFmpeg)

## Performance Tips

1. **Use GPU**: Ensure CUDA is properly installed for significant speed improvements
2. **Model Selection**: 
   - Use `tiny` or `base` for quick processing
   - Use `large` for maximum accuracy
3. **Audio Quality**: Higher quality audio files produce better transcriptions

## Troubleshooting

### Common Issues

**"CUDA out of memory" error:**
- Use a smaller model (tiny, base, or small)
- Close other GPU-intensive applications
- Use `--no-gpu` flag to force CPU usage

**"FFmpeg not found" error:**
- Install FFmpeg and add it to your system PATH
- Restart your terminal/command prompt

**Poor transcription quality:**
- Use a larger model (medium or large)
- Ensure audio file is clear and has minimal background noise
- Check if audio file is corrupted

**Slow processing:**
- Verify GPU is being used (check output messages)
- Use a smaller model for faster processing
- Ensure sufficient RAM and storage space

## Examples

### Example 1: Quick Question Extraction
```bash
python audio_transcriber.py interview.mp3
```
Output: `questions_interview.txt`

### Example 2: High-Quality Transcription
```bash
python audio_transcriber.py lecture.wav --model large --verbose
```
Output: Full transcription + `questions_lecture.txt`

### Example 3: Custom Output
```bash
python audio_transcriber.py meeting.m4a --output meeting_questions.txt
```

## License

This script is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the script.

## Acknowledgments

- OpenAI for the Whisper model
- The open-source community for various Python packages
- FFmpeg for audio processing capabilities
