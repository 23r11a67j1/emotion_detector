# SER Project Review: "Unveiling Hidden Factors: Explainable AI for Feature Boosting in Speech Emotion Recognition"

## Executive Summary

**Overall Status**: ⚠️ **PARTIALLY WORKING** with significant limitations

The project demonstrates a functional architecture but has critical issues that prevent it from working correctly end-to-end. The inference logic uses an unconventional prototype-based approach that may not accurately predict emotions from real audio inputs.

---

## 1. Environment & Dependencies ❌

### Status: FAILED

**Findings:**
- **requirements.txt is INCOMPLETE**: Missing critical dependencies:
  - Flask
  - scikit-learn
  - pandas
  - numpy
  - joblib
  - soundfile (required by librosa)

**Current requirements.txt contains:**
- pycaret==2.3.6
- praat-parselmouth
- librosa
- pca
- statsmodels
- shap
- numba==0.57

**Impact**: Project cannot be installed and run from scratch using requirements.txt.

**Recommendation**: Update requirements.txt with all dependencies.

---

## 2. Project Structure Validation ✅

### Status: PASSED

**All required files present:**
- ✅ app.py (Flask inference + XAI logic)
- ✅ train_model.py (model training)
- ✅ model.joblib (trained RandomForest model)
- ✅ all_handcrafted_data_tess.csv (handcrafted feature dataset - 2800 samples, 44 columns)
- ✅ templates/index.html (frontend UI)
- ✅ static/style.css (styling)
- ✅ uploads/ directory (audio inputs)

**Additional files found:**
- predict_audio.py (standalone prediction script - not used by app)

---

## 3. Model Training Verification ⚠️

### Status: PARTIALLY VERIFIED

**Verified components:**
- ✅ Loads CSV dataset correctly (2800 samples, 44 columns)
- ✅ Drops non-feature columns (path, source)
- ✅ Uses RandomForestClassifier (n_estimators=300, max_depth=20)
- ✅ Applies 80/20 train-test split with stratification
- ✅ Model saved as model.joblib
- ✅ Model expects 41 features (matches dataset after dropping path, source, class)

**Issues:**
- ⚠️ **Unicode encoding error**: train_model.py uses emoji characters in print statements that cause failures on Windows (cp1252 encoding)
- ⚠️ **Version mismatch warning**: Model was trained with scikit-learn 1.7.2 but loaded with 1.8.0 (may cause inconsistencies)

**Cannot verify accuracy** due to encoding errors preventing script execution, but model.joblib exists and loads successfully.

---

## 4. Inference Logic Validation ⚠️

### Status: FUNCTIONAL BUT PROBLEMATIC

**Current Approach (Prototype-Based):**
1. Extracts only **3 traits** from audio: energy (RMS), pitch (librosa.yin), zero-crossing rate
2. Scales these 3 traits using StandardScaler
3. Finds 15 nearest neighbors in the dataset based on these 3 traits
4. Uses full 41-feature vectors from those neighbors to predict
5. Majority vote on 15 predictions

**Verified:**
- ✅ Model loads correctly (expects 41 features)
- ✅ Dataset loads correctly (41 feature columns after dropping metadata)
- ✅ Feature columns match model expectations
- ✅ Trait extraction function exists
- ✅ Nearest neighbor search implemented
- ✅ Majority voting implemented
- ✅ Predictions work when tested with dataset samples

**Critical Issues:**
- ⚠️ **Feature mismatch**: Model trained on 41 features, but inference uses only 3 traits for similarity search
- ⚠️ **Unconventional approach**: The prototype-based method may not capture full audio characteristics
- ⚠️ **No direct prediction**: System doesn't extract full feature set from audio and predict directly
- ⚠️ **Limited discriminative power**: 3 traits may not be sufficient to distinguish between similar emotions

**Test Results:**
- When tested with dataset samples, predictions match true labels (3/3 matches in test)
- However, this doesn't validate predictions on real, unseen audio files

---

## 5. Prediction Correctness Check ❌

### Status: CANNOT VERIFY (Critical)

**Problem**: Cannot test with actual audio files because:
- librosa is not installed in current environment
- No way to verify if predictions vary meaningfully with different audio characteristics

**Inference Logic Analysis:**
- The system extracts only 3 simple traits (RMS energy, pitch, ZCR)
- These traits are used to find "similar" samples in the training dataset
- The actual prediction comes from the labels of those similar samples

**Risk Assessment:**
- **HIGH RISK** that the system produces similar predictions for different emotions
- **HIGH RISK** that audio with similar energy/pitch/ZCR profiles always predict the same emotion
- The 3-trait similarity search may collapse diverse audio into the same nearest neighbors

**What Should Be Tested (But Cannot Due to Missing Dependencies):**
1. Calm, low-energy speech → Should predict Neutral or Sad
2. High-energy happy speech → Should predict Happy
3. Loud, sharp angry speech → Should predict Angry
4. Fearful or trembling speech → Should predict Fear
5. Sad or slow speech → Should predict Sad

**Verdict**: ❌ **CANNOT CONFIRM** that predictions are correct or meaningful for real audio inputs.

---

## 6. Explainable AI (XAI) Validation ❌

### Status: FAILED (Not True XAI)

**Current Implementation:**
- `generate_explanation()` function uses **hardcoded thresholds**:
  - Energy > 0.05 → "high vocal energy"
  - Pitch > 200 → "elevated pitch"
  - ZCR > 0.1 → "rapid speech articulation"

**Critical Issues:**
- ❌ **SHAP explainer created but NEVER USED**: Line 16 creates `explainer = shap.TreeExplainer(model)` but it's never called
- ❌ **Not model-based**: Explanations are rule-based, not derived from the model's decision
- ❌ **Not feature-based**: Doesn't explain which of the 41 features contributed to the prediction
- ❌ **Generic thresholds**: Fixed thresholds (0.05, 200, 0.1) may not be appropriate for all emotions
- ⚠️ **Contradictory**: Explanations are based on 3 traits, but prediction uses 41 features from nearest neighbors

**What Real XAI Should Do:**
- Use SHAP values to identify which features contributed most to the prediction
- Explain why specific audio characteristics led to the emotion classification
- Show feature importance specific to each prediction

**Verdict**: ❌ **NOT EXPLAINABLE AI** - Just rule-based text generation, not true XAI.

---

## 7. Frontend & UX Check ✅

### Status: FUNCTIONAL

**Verified:**
- ✅ HTML template exists (index.html)
- ✅ CSS styling exists (style.css)
- ✅ Supports audio file upload (input type="file" accept=".wav")
- ✅ Supports live audio recording (MediaRecorder API implemented)
- ✅ Displays predicted emotion
- ✅ Displays confidence score
- ✅ Displays explanation text
- ✅ Form submission implemented correctly

**Issues:**
- ⚠️ No error messages displayed to user if prediction fails
- ⚠️ No loading indicator during processing
- ⚠️ Recording functionality may have browser compatibility issues

**Note**: Cannot test Flask rendering without Flask installed, but structure appears correct.

---

## 8. Stability & Error Handling ❌

### Status: POOR

**Missing Error Handling:**
- ❌ No try/except blocks in `app.py` inference route
- ❌ No error handling in `extract_audio_traits()` - will crash if:
  - Audio file is corrupted
  - Audio format is unsupported
  - File doesn't exist
  - librosa.load() fails
- ❌ No error handling for model prediction failures
- ❌ No error handling for file upload issues
- ❌ No validation of audio file format/size before processing

**Potential Runtime Errors:**
- `librosa.load()` may fail with invalid audio files
- `scaler.transform()` may fail if traits are NaN or inf
- `model.predict()` may fail if feature columns don't match
- Flask route may crash without user feedback

**Stability Risks:**
- Repeated uploads of invalid files will crash the app
- No graceful degradation
- No user-facing error messages

---

## 9. Code Quality Issues

### Additional Findings:

1. **Unused Code:**
   - `predict_audio.py` exists but is not used by the Flask app
   - SHAP explainer created but never used

2. **Encoding Issues:**
   - train_model.py uses emoji characters that cause failures on Windows
   - Prevents script execution in standard Windows environments

3. **Inconsistencies:**
   - `predict_audio.py` extracts different features than `app.py`
   - Two different inference approaches in the codebase

4. **Documentation:**
   - README mentions accuracy of 0.9875 but this cannot be verified
   - No documentation of the prototype-based inference approach

---

## 10. Final Verdict

### ⚠️ **PARTIALLY WORKING** - Significant Limitations

**What Works:**
- ✅ Project structure is complete
- ✅ Model training script structure is correct
- ✅ Model loads and makes predictions on dataset samples
- ✅ Frontend structure is functional
- ✅ Inference pipeline executes without syntax errors (on dataset samples)

**What Doesn't Work:**
- ❌ **Cannot verify predictions on real audio** (missing librosa, cannot test)
- ❌ **XAI is not true explainable AI** (hardcoded rules, SHAP unused)
- ❌ **Incomplete dependencies** (requirements.txt missing critical packages)
- ❌ **Poor error handling** (app will crash on invalid inputs)
- ❌ **Unconventional inference** (3-trait similarity may not work correctly)
- ❌ **Encoding issues** (train_model.py fails on Windows)

**Critical Concerns:**
1. **Prediction Accuracy Unknown**: Cannot confirm if the system actually distinguishes emotions correctly from real audio
2. **XAI False Promise**: System claims explainable AI but uses simple rule-based explanations
3. **Production Readiness**: No error handling means the app will crash in real-world use

---

## Recommendations

### Critical Fixes (Must Do):

1. **Fix requirements.txt**: Add all missing dependencies
   ```
   flask
   scikit-learn
   pandas
   numpy
   joblib
   librosa
   soundfile
   shap
   ```

2. **Fix encoding issues**: Remove emoji characters from train_model.py print statements

3. **Add error handling**: Wrap all inference code in try/except blocks

4. **Fix XAI implementation**: Either use SHAP properly or remove it (don't create unused objects)

### Important Improvements:

5. **Rethink inference approach**: Consider extracting full feature set from audio and predicting directly, or document why prototype-based approach is used

6. **Test with diverse audio**: Verify that different emotions produce different predictions

7. **Improve explanations**: Use SHAP to generate actual feature-based explanations

8. **Add input validation**: Validate audio files before processing

### Nice to Have:

9. **Add logging**: Track predictions and errors

10. **Add unit tests**: Test each component independently

11. **Document inference approach**: Explain why prototype-based method was chosen

---

## Conclusion

The project demonstrates a working architecture but has critical gaps that prevent it from being a reliable, production-ready Speech Emotion Recognition system. The most significant concern is the inability to verify that predictions are correct and meaningful for real audio inputs. The explainable AI claim is not substantiated by the implementation.

**Recommendation**: Address critical fixes before claiming the system works end-to-end.

