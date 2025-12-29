# Testing Guide - Live Recording Fix

## What Was Fixed

1. **Audio Format Conversion**: Added WebM to WAV conversion using pydub
2. **Better Error Handling**: Added comprehensive error messages and logging
3. **Debug Logging**: Added console.log statements to track the recording process
4. **Empty Audio Check**: Added validation to ensure audio data is collected

## How to Test

### Step 1: Ensure Dependencies Are Installed

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # PowerShell
# OR
venv\Scripts\activate  # Command Prompt

# Install/Update dependencies
pip install -r requirements.txt

# Install ffmpeg (REQUIRED for WebM conversion)
# Option 1: Using Chocolatey (if installed)
choco install ffmpeg

# Option 2: Manual installation
# Download from https://ffmpeg.org/download.html
# Extract and add to PATH
```

### Step 2: Verify ffmpeg Installation

```bash
ffmpeg -version
```

If this command works, ffmpeg is installed correctly.

### Step 3: Start the Flask App

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### Step 4: Test in Browser

1. Open browser: `http://127.0.0.1:5000`
2. Open Developer Console (F12) to see debug messages
3. Click "Start Recording"
4. Allow microphone access when prompted
5. Speak for 2-3 seconds
6. Click "Stop Recording"
7. Check the console for messages:
   - "Recording stopped. Audio chunks: X"
   - "Recording duration: X seconds"
   - "Created blob, size: X bytes"
   - "Sending audio to server..."
   - "Server response status: 200"

### Step 5: Check Flask Terminal

You should see:
- "POST request received. Audio file: ..."
- "Saving audio to: uploads/recording_..."
- "Converting .webm file to WAV..."
- "Successfully converted to WAV: ..."
- "Processing audio file: ..."
- "Extracted traits: ..."
- "Prediction: ..., Confidence: X%"

## Troubleshooting

### Issue: "No output after recording"

**Check Browser Console (F12):**
- Look for JavaScript errors
- Check if "Recording stopped" message appears
- Check if "Sending audio to server..." appears
- Check server response status

**Check Flask Terminal:**
- Look for error messages
- Check if file is being saved
- Check if conversion is happening
- Look for audio processing errors

### Issue: "ffmpeg not found" or "WebM conversion failed"

**Solution:** Install ffmpeg
- Windows: Download from https://ffmpeg.org/download.html
- Extract to a folder (e.g., `C:\ffmpeg`)
- Add to PATH environment variable
- Restart terminal and Flask app

### Issue: "No audio chunks collected"

**Possible causes:**
- Recording duration too short (less than 100ms)
- Microphone not working
- Browser permissions denied

**Solution:**
- Record for at least 2 seconds
- Check microphone permissions in browser settings
- Try a different browser (Chrome/Edge recommended)

### Issue: "Audio processing error"

**Check Flask terminal for detailed error:**
- If it's a format error: Install ffmpeg
- If it's a librosa error: Check if audio file is valid
- If it's a feature extraction error: Check audio quality

### Issue: "Page reloads but no results"

**Possible causes:**
- Backend error occurred (check Flask terminal)
- Template variables not being passed correctly
- Error occurred but not displayed

**Solution:**
- Check Flask terminal for errors
- Check browser console for JavaScript errors
- Look for error messages in the UI

## Expected Behavior

1. **Recording Start:**
   - Record button becomes disabled
   - Stop button becomes enabled
   - Recording indicator appears with timer
   - Waveform visualization starts

2. **Recording Stop:**
   - Recording UI disappears immediately
   - Loading indicator appears
   - Page reloads after processing

3. **Results Display:**
   - Predicted emotion appears
   - Confidence score appears
   - Explanation text appears

## Debug Information

All debug messages are logged to:
- **Browser Console**: JavaScript debug messages
- **Flask Terminal**: Python backend debug messages

If something doesn't work, check both for error messages.

