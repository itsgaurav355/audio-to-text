# Audio Transcriber with Question Extraction

A Python script that transcribes audio files using OpenAI's Whisper model with GPU acceleration and intelligently extracts questions from the transcription.

## Features

- 🎵 **Audio Transcription**: Supports various audio formats (MP3, WAV, M4A, FLAC, OGG, AAC, M4B, WMA)
- 🚀 **GPU Acceleration**: Automatically detects and uses CUDA GPU for faster processing
- 🔍 **Smart Question Extraction**: Uses multiple patterns to identify questions (question marks, interrogative words, rhetorical patterns)
- 📁 **Batch Processing**: Process entire folders of audio files at once
- 💾 **Dual Output**: Saves both full transcription and extracted questions
- 🎛️ **Flexible Models**: Choose from different Whisper model sizes (tiny, base, small, medium, large)
- 🖥️ **Cross-Platform**: Works on Windows, macOS, and Linux

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
4. Restart your terminal/command prompt

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## Complete Setup Instructions

### Step 1: Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Navigate to your project directory
cd "G:\Coding\AIML\AudioToText"

# Create virtual environment
python -m venv myenv

# Activate virtual environment
.\myenv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
# Navigate to your project directory
cd "G:\Coding\AIML\AudioToText"

# Create virtual environment
python -m venv myenv

# Activate virtual environment
.\myenv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
# Navigate to your project directory
cd /path/to/AudioToText

# Create virtual environment
python3 -m venv myenv

# Activate virtual environment
source myenv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Make sure your virtual environment is activated (you should see (myenv) in your prompt)
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Check if Whisper is installed
python -c "import whisper; print('Whisper installed successfully')"

# Check if PyTorch with CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test the script
python audio_transcriber.py --help
```

## Usage

### Basic Usage

```bash
# Transcribe a single audio file
python audio_transcriber.py audio.mp3

# Transcribe with specific model size
python audio_transcriber.py audio.wav --model large

# Force CPU usage (useful if GPU has memory issues)
python audio_transcriber.py audio.m4a --no-gpu
```

### Advanced Usage

```bash
# Specify output directory for both transcription and questions
python audio_transcriber.py audio.mp3 --output ./output_folder/

# Process entire folder of audio files
python audio_transcriber.py ./audios/ --output ./questions/

# Use large model for maximum accuracy
python audio_transcriber.py audio.wav --model large --output ./detailed_output/
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio_input` | Path to audio file or folder (required) | - |
| `--model, -m` | Whisper model size: tiny, base, small, medium, large | base |
| `--output, -o` | Output directory for files | Current directory |
| `--no-gpu` | Force CPU usage even if GPU is available | False |

## Model Sizes and Performance

| Model | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|-------|------------|--------------|--------------|---------------|----------------|
| tiny | 39 M | ✓ | ✓ | ~1 GB | ~32x |
| base | 74 M | ✓ | ✓ | ~1 GB | ~16x |
| small | 244 M | ✓ | ✓ | ~2 GB | ~6x |
| medium | 769 M | ✓ | ✓ | ~5 GB | ~2x |
| large | 1550 M | ✓ | ✓ | ~10 GB | 1x |

**Recommendation**: Start with `base` for testing, use `large` for production quality.

## Question Extraction Logic

The script uses multiple intelligent patterns to identify questions:

1. **Question Mark Detection**: Sentences ending with "?"
2. **Interrogative Word Detection**: Sentences starting with question words:
   - What, When, Where, Who, Whom, Whose, Why, How, Which
   - Can, Could, Would, Will, Should, Do, Does, Did
   - Is, Are, Was, Were, Have, Has, Had
   - May, Might, Must, Shall, Ought, Need, Dare
3. **Rhetorical Patterns**: "right?", "correct?", "you know?", "isn't it?"
4. **Contextual Questions**: Sentences with words like "think", "believe", "suppose", "wonder"

## Output Files

For each audio file, the script creates two files:

1. **`{audio_name}_transcription.txt`**: Complete transcription
2. **`{audio_name}_questions.txt`**: Extracted questions only

**Example output structure:**
```
questions/
├── interview_transcription.txt
├── interview_questions.txt
├── lecture_transcription.txt
└── lecture_questions.txt
```

## Supported Audio Formats

- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, M4B, WMA
- **Video with Audio**: AVI, MOV, MP4 (audio track will be extracted)

## Complete Workflow Examples

### Example 1: Single File Processing
```bash
# Activate environment
.\myenv\Scripts\Activate.ps1

# Transcribe single file
python audio_transcriber.py "audios\interview.mp3" --model large

# Output: interview_transcription.txt and interview_questions.txt in current directory
```

### Example 2: Batch Processing
```bash
# Activate environment
.\myenv\Scripts\Activate.ps1

# Process all audio files in audios folder
python audio_transcriber.py "audios\" --output "questions\"

# Output: All transcriptions and questions saved to questions/ folder
```

### Example 3: High-Quality Processing
```bash
# Activate environment
.\myenv\Scripts\Activate.ps1

# Use large model for maximum accuracy
python audio_transcriber.py "audios\lecture.wav" --model large --output "detailed_output\"
```

## Performance Tips

1. **GPU Usage**: The script automatically detects and uses CUDA GPU if available (if does not recognize than install torch higher version)
2. **Model Selection**: 
   - Use `tiny` or `base` for quick processing and testing
   - Use `large` for maximum accuracy and production use
3. **Audio Quality**: Higher quality audio files produce better transcriptions
4. **Batch Processing**: Process multiple files at once for efficiency

## Troubleshooting

### Common Issues

**"CUDA out of memory" error:**
```bash
# Use smaller model
python audio_transcriber.py audio.mp3 --model base

# Or force CPU usage
python audio_transcriber.py audio.mp3 --no-gpu
```

**"FFmpeg not found" error:**
- Install FFmpeg and add to system PATH
- Restart terminal/command prompt
- Verify with: `ffmpeg -version`

**Poor transcription quality:**
- Use larger model: `--model large`
- Ensure audio is clear with minimal background noise
- Check if audio file is corrupted

**Virtual environment not working:**
```bash
# Windows PowerShell
.\myenv\Scripts\Activate.ps1

# Windows Command Prompt
.\myenv\Scripts\activate.bat

# macOS/Linux
source myenv/bin/activate
```

**GPU not detected:**
- Ensure virtual environment is activated
- Verify PyTorch with CUDA is installed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- Check CUDA installation: `nvidia-smi`

## File Structure

```
AudioToText/
├── audio_transcriber.py          # Main script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── myenv/                       # Virtual environment
├── audios/                      # Input audio files
└── questions/                   # Output files (created after processing)
```

## License

This script is provided as-is for educational and personal use.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the script.

## Acknowledgments

- OpenAI for the Whisper model
- The open-source community for various Python packages
- FFmpeg for audio processing capabilities
