import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
from nltk.tokenize import sent_tokenize
import tempfile
import os
import io
import nltk
import time
import threading
import queue

# Ensure punkt is downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(page_title="Live Question Extractor", layout="wide")
st.title("🎧 Live Question Extractor (Continuous Listening)")

# --- Session State Initialization ---
if "listening" not in st.session_state:
    st.session_state.listening = False
if "questions_log" not in st.session_state:
    st.session_state.questions_log = []
if "latest_status" not in st.session_state:
    st.session_state.latest_status = "Click '▶️ Start Listening' to begin."
if "audio_queue" not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if "processing_thread" not in st.session_state:
    st.session_state.processing_thread = None

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🎛️ Settings")
    duration = st.slider("Chunk Duration (seconds)", 3, 10, 5)
    model_size = st.selectbox("Model size", ["base", "small", "medium", "large-v2"])

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Listening")
    with col2:
        stop_button = st.button("⏹️ Stop Listening")

# --- Load Whisper model (cached for efficiency) ---
@st.cache_resource
def load_model(size):
    """
    Loads the Whisper model. This function is cached to prevent reloading
    the model on every Streamlit rerun, improving performance.
    """
    with st.spinner(f"Loading Whisper model: {size}. This may take a moment..."):
        return WhisperModel(size, compute_type="int8", device="cpu")

model = load_model(model_size)

# --- Question Detection Logic ---
def is_question(sentence):
    """
    Determines if a given sentence is likely a question based on
    punctuation and common question words.
    """
    question_words = (
        'what', 'when', 'where', 'who', 'whom', 'which',
        'why', 'how', 'do', 'does', 'did', 'is', 'are', 'can',
        'could', 'would', 'should', 'will', 'shall', 'may', 'might'
    )
    sentence_lower = sentence.lower().strip()
    # Check for question mark or if it starts with a common question word
    return (sentence_lower.endswith('?') or 
            any(sentence_lower.startswith(qw + ' ') for qw in question_words))

# --- Audio Processing Function ---
def process_audio_chunk(audio_data, fs, model):
    """Process a single audio chunk and return detected questions."""
    tmpfile_path = None
    try:
        # Create a temporary file with delete=False to handle manually
        tmpfile = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmpfile_path = tmpfile.name
        tmpfile.close()  # Close the file handle immediately
        
        # Write audio data to the temporary file
        wav.write(tmpfile_path, fs, audio_data)
        
        # Transcribe the audio using the Whisper model
        segments, _ = model.transcribe(tmpfile_path)
        full_text = " ".join([seg.text for seg in segments])
        
        if full_text.strip():
            # Extract sentences and identify questions
            sentences = sent_tokenize(full_text)
            questions = [s.strip() for s in sentences if is_question(s.strip())]
            return full_text, questions
        else:
            return "", []
            
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return "", []
    finally:
        # Clean up the temporary file with retry logic for Windows
        if tmpfile_path and os.path.exists(tmpfile_path):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.unlink(tmpfile_path)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Wait a bit before retrying
                    else:
                        # If we can't delete after retries, log the issue but don't crash
                        print(f"Warning: Could not delete temporary file {tmpfile_path}")
                except Exception:
                    break  # File might already be deleted

# --- Continuous Audio Recording Function ---
def continuous_recording(duration, fs, audio_queue, stop_flag):
    """Record audio continuously in chunks."""
    while not stop_flag[0]:
        try:
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            if not stop_flag[0]:  # Check if we should stop before adding to queue
                audio_queue.put(recording)
        except Exception as e:
            print(f"Recording error: {e}")
            break

# Handle button clicks
if start_button and not st.session_state.listening:
    st.session_state.listening = True
    st.session_state.latest_status = "Starting to listen..."
    st.session_state.audio_queue = queue.Queue()
    
    # Start continuous recording in a separate thread
    stop_flag = [False]
    st.session_state.stop_flag = stop_flag
    recording_thread = threading.Thread(
        target=continuous_recording,
        args=(duration, 16000, st.session_state.audio_queue, stop_flag)
    )
    recording_thread.daemon = True
    recording_thread.start()
    st.session_state.recording_thread = recording_thread

if stop_button and st.session_state.listening:
    st.session_state.listening = False
    st.session_state.latest_status = "Stopped listening."
    if hasattr(st.session_state, 'stop_flag'):
        st.session_state.stop_flag[0] = True

# --- Main Content Area ---
st.write(f"**Status:** {st.session_state.latest_status}")

# Create containers for dynamic updates
status_container = st.container()
transcription_container = st.container()
questions_container = st.container()

# --- Process Audio Queue ---
if st.session_state.listening and not st.session_state.audio_queue.empty():
    try:
        # Get the latest audio chunk
        audio_data = st.session_state.audio_queue.get_nowait()
        
        with status_container:
            with st.spinner("Processing audio..."):
                transcribed_text, new_questions = process_audio_chunk(audio_data, 16000, model)
        
        # Show transcribed text
        if transcribed_text:
            with transcription_container:
                st.caption(f"**Latest transcription:** \"{transcribed_text}\"")
        
        # Add new questions to log
        if new_questions:
            st.session_state.questions_log.extend(new_questions)
            st.session_state.latest_status = f"Detected {len(new_questions)} question(s)! Continuing to listen..."
        else:
            st.session_state.latest_status = "No questions detected. Continuing to listen..."
            
    except queue.Empty:
        pass
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.session_state.listening = False
        st.session_state.latest_status = "Error occurred. Listening stopped."

# --- Display Questions ---
with questions_container:
    if st.session_state.questions_log:
        st.markdown("### ❓ Detected Questions:")
        
        # Add a clear button
        if st.button("🗑️ Clear Questions"):
            st.session_state.questions_log = []
            st.rerun()
        
        # Display questions with timestamps (newest first)
        for i, question in enumerate(reversed(st.session_state.questions_log)):
            st.markdown(f"**{len(st.session_state.questions_log) - i}.** {question}")
    else:
        st.info("No questions detected yet. Start listening to begin capturing questions.")

# Auto-refresh when listening
if st.session_state.listening:
    time.sleep(0.5)  # Small delay to prevent overwhelming
    st.rerun()

# --- Cleanup on app termination ---
if not st.session_state.listening and hasattr(st.session_state, 'stop_flag'):
    st.session_state.stop_flag[0] = True