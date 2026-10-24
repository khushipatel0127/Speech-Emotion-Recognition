# SpeechSense Project Demonstration Guide

## Before the evaluation

1. Start SpeechSense and check `/api/health` before the presentation.
2. Keep two short supported audio files ready, including one browser-recorded or
   phone-recorded example.
3. Confirm the History page contains either intentional demo records or is empty.
4. Keep the dataset-validation and model-evaluation reports open for questions.
5. Do not promise a particular emotion result for a live recording.

## Suggested five-minute walkthrough

1. Introduce the problem: emotion-related acoustic classification is difficult
   because speakers, acting style, microphones, volume, and datasets differ.
2. Show the Overview page and explain that its counts come from SQLite.
3. Upload a recording on Analyze Speech, play it back, and run the real backend
   model. Explain that the bars are model-produced scores, not certainty about a
   person's internal state.
4. Open the result details, play the stored converted audio, and download the
   report.
5. Show History search and emotion filtering.
6. Present Phase 2 methodology: six common labels, 115 speakers, exact-duplicate
   detection, speaker-independent splitting, grouped cross-validation, a sealed
   final test set, separate dataset results, and training-only augmentation.
7. Compare the 47.45% handcrafted validation macro F1 with WavLM-Large's 73.39%.
8. Present the one-time speaker-independent final test: 77.54% accuracy and
   77.78% macro F1 across 1,318 recordings from 18 speakers. State clearly that
   the result is below 90% and does not measure natural conversation.
9. Show the resumable Colab shards, saved classifier, and historical rollback.

The current natural-recording pilot contains three intentionally Happy phone
recordings. The historical deployed model predicted Angry for all three. Present
this as evidence motivating Phase 2, not as a complete natural-speech benchmark.

## Likely questions

**Why six emotions instead of eight?**  
They are the labels shared by RAVDESS and CREMA-D. Calm and surprised are not
merged because that would change their original meaning.

**How did you prevent leakage?**  
Every speaker belongs to exactly one partition. Feature preprocessing and SVM
calibration are fitted only within training groups. Exact PCM duplicates are
identified, and none crosses a partition.

**Did the project reach 90% accuracy?**
No. The honest speaker-independent final-test accuracy is 77.54%. A random
recording split could produce an inflated number through speaker leakage, so it
was deliberately avoided.

**Are the displayed probabilities true confidence?**  
They are calibrated model estimates. They are useful for ranking classes but do
not guarantee that the predicted emotion is correct.

**Why not fine-tune a deep model locally?**  
The laptop has 8 GB RAM and no CUDA GPU. Frozen WavLM embeddings preserve a
pretrained-speech approach while keeping classifier training lightweight; the
expensive extraction step is prepared for a free Colab GPU.

## Claims to avoid

- Do not call the application a lie detector, diagnostic tool, or psychological
  assessment.
- Do not report 90% accuracy. The measured final-test accuracy is 77.54%.
- Do not describe public acted-speech performance as natural-speech accuracy.
- Do not tune further using the final-test results.
