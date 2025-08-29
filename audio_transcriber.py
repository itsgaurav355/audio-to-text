#!/usr/bin/env python3
"""
Audio Transcriber with Question Extraction
This script transcribes audio files using Whisper (with GPU support) and extracts only questions.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Optional
import torch

try:
    import whisper
except ImportError:
    print("Error: whisper package not found. Please install it using:")
    print("pip install openai-whisper")
    sys.exit(1)

def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available for faster processing."""
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠️  GPU not available. Using CPU (will be slower).")
        return False

def transcribe_audio(audio_path: str, model_size: str = "base", use_gpu: bool = True) -> str:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        use_gpu: Whether to use GPU if available
    
    Returns:
        Transcribed text
    """
    print(f"🎵 Loading Whisper model: {model_size}")
    
    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    try:
        # Load model
        model = whisper.load_model(model_size, device=device)
        print(f"✅ Model loaded successfully on {device.upper()}")
        
        # Transcribe audio
        print(f"🎤 Transcribing audio: {audio_path}")
        result = model.transcribe(audio_path)
        
        # Check if transcription is empty or very short
        if not result["text"] or len(result["text"].strip()) < 10:
            print("⚠️  Warning: Transcription seems very short or empty")
            print(f"   Raw transcription: '{result['text']}'")
        
        return result["text"]
        
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        
        # Provide more specific error information
        if "Failed to load audio" in str(e):
            print("💡 This usually means the audio file is corrupted or in an unsupported format")
            print("   Try converting the audio to MP3 or WAV format")
        elif "CUDA out of memory" in str(e):
            print("💡 GPU memory insufficient. Try using a smaller model or --no-gpu flag")
        
        return ""

def extract_questions(text: str) -> List[str]:
    """
    Extract questions from transcribed text.
    
    Args:
        text: Transcribed text
        
    Returns:
        List of questions found in the text
    """
    if not text:
        return []
    
    # Clean the text
    text = text.strip()
    
    # Split into sentences and identify questions
    questions = []
    
    # Pattern 1: Sentences ending with question marks
    question_pattern = r'[^.!?]*\?'
    question_matches = re.findall(question_pattern, text)
    questions.extend([q.strip() for q in question_matches if q.strip()])
    
    # Pattern 2: Sentences starting with question words (interrogative sentences)
    question_words = [
        'what', 'when', 'where', 'who', 'whom', 'whose', 'why', 'how',
        'which', 'can', 'could', 'would', 'will', 'should', 'do', 'does',
        'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had',
        'may', 'might', 'must', 'shall', 'ought', 'need', 'dare'
    ]
    
    # Pattern 3: Rhetorical questions and statements that are questions
    rhetorical_patterns = [
        r'\b(?:right|correct|true|yes|no|okay|ok)\?',  # "right?", "correct?"
        r'\b(?:you\s+know|you\s+see|you\s+understand)\?',  # "you know?", "you see?"
        r'\b(?:isn\'t\s+it|aren\'t\s+they|don\'t\s+you|doesn\'t\s+it)\?',  # "isn't it?"
        r'\b(?:can\'t\s+you|won\'t\s+you|wouldn\'t\s+you)\?',  # "can't you?"
    ]
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Check if sentence starts with question word (case insensitive)
            sentence_lower = sentence.lower()
            for word in question_words:
                if sentence_lower.startswith(word + ' '):
                    # Make sure it's not already captured as a question mark sentence
                    # and doesn't end with a question mark (to avoid duplicates)
                    if not sentence.endswith('?'):
                        questions.append(sentence)
                    break
            
            # Check for rhetorical patterns
            for pattern in rhetorical_patterns:
                if re.search(pattern, sentence_lower):
                    if not sentence.endswith('?'):
                        questions.append(sentence)
                    break
    
    # Pattern 4: Look for sentences with question-like intonation patterns
    # (sentences that might be questions but don't have obvious markers)
    potential_questions = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence.split()) >= 3:  # At least 3 words
            sentence_lower = sentence.lower()
            
            # Check for sentences that might be questions based on context
            if any(word in sentence_lower for word in ['think', 'believe', 'suppose', 'guess', 'wonder']):
                if not sentence.endswith('?') and sentence not in questions:
                    potential_questions.append(sentence)
    
    # Add potential questions (but mark them as uncertain)
    questions.extend(potential_questions)
    
    # Remove duplicates while preserving order
    unique_questions = []
    seen = set()
    for q in questions:
        q_clean = q.lower().strip()
        if q_clean not in seen:
            unique_questions.append(q)
            seen.add(q_clean)
    
    return unique_questions

def save_questions(questions: List[str], output_path: str):
    """Save extracted questions to a text file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("QUESTIONS EXTRACTED FROM AUDIO TRANSCRIPTION\n")
            f.write("=" * 50 + "\n\n")
            
            if questions:
                for i, question in enumerate(questions, 1):
                    f.write(f"{i}. {question}\n")
                f.write(f"\nTotal questions found: {len(questions)}")
            else:
                f.write("No questions found in the transcription.")
        
        print(f"✅ Questions saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving questions: {str(e)}")

def save_transcription_and_questions(transcription: str, questions: List[str], audio_name: str, output_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """
    Save both transcription and extracted questions to text files.
    
    Args:
        transcription: The transcribed text.
        questions: List of extracted questions.
        audio_name: The name of the audio file (without extension).
        output_dir: Optional output directory for both files.
    
    Returns:
        A tuple of (Path to transcription file, Path to questions file).
    """
    if output_dir is None:
        output_dir = Path(".") # Save in current directory
    
    transcription_file = output_dir / f"{audio_name}_transcription.txt"
    questions_file = output_dir / f"{audio_name}_questions.txt"
    
    try:
        # Save transcription
        with open(transcription_file, 'w', encoding='utf-8') as f:
            f.write("AUDIO TRANSCRIPTION\n")
            f.write("=" * 50 + "\n\n")
            f.write(transcription)
        print(f"✅ Transcription saved to: {transcription_file}")
        
        # Save questions
        save_questions(questions, str(questions_file))
        
    except Exception as e:
        print(f"❌ Error saving transcription or questions: {str(e)}")
    
    return transcription_file, questions_file

def process_audio_folder(folder_path: str, model_size: str = "large", use_gpu: bool = True, output_dir: str = None):
    """
    Process all audio files in a folder.
    
    Args:
        folder_path: Path to folder containing audio files
        model_size: Whisper model size
        use_gpu: Whether to use GPU
        output_dir: Output directory for questions (default: same as folder)
    """
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"❌ Error: '{folder_path}' is not a valid directory.")
        return
    
    # Supported audio formats
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.m4b', '.wma'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(folder_path.glob(f"*{ext}"))
        audio_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        print(f"❌ No audio files found in '{folder_path}'")
        print(f"   Supported formats: {', '.join(audio_extensions)}")
        return
    
    print(f"🎵 Found {len(audio_files)} audio files in '{folder_path}'")
    
    # Set output directory
    if output_dir is None:
        output_dir = folder_path / "transcriptions_and_questions"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process each audio file
    successful = 0
    failed = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(audio_files)}: {audio_file.name}")
        print(f"{'='*60}")
        
        try:
            # Transcribe audio
            transcription = transcribe_audio(str(audio_file), model_size, use_gpu)
            
            if not transcription:
                print(f"❌ Failed to transcribe {audio_file.name}")
                failed += 1
                continue
            
            # Extract questions
            questions = extract_questions(transcription)
            
            # Save both transcription and questions
            transcription_file, questions_file = save_transcription_and_questions(transcription, questions, audio_file.stem, output_dir)
            
            # Show summary
            if questions:
                print(f"✅ Found {len(questions)} questions in {audio_file.name}")
            else:
                print(f"ℹ️  No questions found in {audio_file.name}")
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {str(e)}")
            failed += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   - Total files: {len(audio_files)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    print(f"   - Files saved to: {output_dir}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed. Check the error messages above.")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio and extract questions using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_transcriber.py audio.mp3
  python audio_transcriber.py audio.wav --output questions.txt
  python audio_transcriber.py audio.m4a --no-gpu
  python audio_transcriber.py ./audios/ --output ./questions/
        """
    )
    
    parser.add_argument(
        "audio_input",
        help="Path to audio file or folder containing audio files"
    )
    
    # Model argument is kept for compatibility, but default is now 'large'
    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for questions (default: questions_<audio_name>.txt) or output directory for batch processing"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    # --verbose is kept for compatibility, but full transcription is always shown
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="(Deprecated) Show detailed transcription output (now always shown)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.audio_input):
        print(f"❌ Error: Input path '{args.audio_input}' not found.")
        sys.exit(1)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    # Use GPU by default unless --no-gpu is specified and GPU is available
    use_gpu = gpu_available and not args.no_gpu
    
    # Determine if input is a file or folder
    if os.path.isfile(args.audio_input):
        # Process single audio file
        print(f"\n🚀 Starting audio transcription...")
        print(f"📁 Audio file: {args.audio_input}")
        print(f"🤖 Model: {args.model}")
        print(f"⚡ Device: {'GPU' if use_gpu else 'CPU'}")
        print("-" * 50)
        
        # Transcribe audio
        transcription = transcribe_audio(args.audio_input, args.model, use_gpu)
        
        if not transcription:
            print("❌ Transcription failed. Exiting.")
            sys.exit(1)
        
        # Always show full transcription by default
        print(f"\n📝 Full Transcription:")
        print("-" * 30)
        print(transcription)
        print("-" * 30)
        
        # Extract questions
        print(f"\n🔍 Extracting questions...")
        questions = extract_questions(transcription)
        
        if questions:
            print(f"✅ Found {len(questions)} questions:")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
        else:
            print("ℹ️  No questions found in the transcription.")
        
        # Save both transcription and questions
        if args.output:
            # If output is specified, treat it as a directory for both files
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            audio_name = Path(args.audio_input).stem
            transcription_file, questions_file = save_transcription_and_questions(transcription, questions, audio_name, output_dir)
        else:
            # Save in current directory
            audio_name = Path(args.audio_input).stem
            transcription_file, questions_file = save_transcription_and_questions(transcription, questions, audio_name)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Audio transcribed: {len(transcription)} characters")
        print(f"   - Questions extracted: {len(questions)}")
        print(f"   - Transcription saved to: {transcription_file}")
        print(f"   - Questions saved to: {questions_file}")
        
    elif os.path.isdir(args.audio_input):
        # Process folder of audio files
        print(f"\n🚀 Starting batch audio transcription...")
        print(f"📁 Input folder: {args.audio_input}")
        print(f"🤖 Model: {args.model}")
        print(f"⚡ Device: {'GPU' if use_gpu else 'CPU'}")
        print("-" * 50)
        
        process_audio_folder(args.audio_input, args.model, use_gpu, args.output)
        
    else:
        print(f"❌ Error: '{args.audio_input}' is neither a valid file nor directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
