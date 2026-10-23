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
7. Present the measured limitation honestly: the best handcrafted validation
   macro F1 is 47.45%, cross-dataset transfer is weak, and gain sensitivity is
   substantial. The augmented model improves 3×-gain macro F1 from 31.80% to
   46.05% but slightly lowers normal validation performance, so explain why it
   was retained as an experiment rather than prematurely deployed.
8. Show the resumable WavLM Colab workflow as the next evidence-based experiment.

## Likely questions

**Why six emotions instead of eight?**  
They are the labels shared by RAVDESS and CREMA-D. Calm and surprised are not
merged because that would change their original meaning.

**How did you prevent leakage?**  
Every speaker belongs to exactly one partition. Feature preprocessing and SVM
calibration are fitted only within training groups. Exact PCM duplicates are
identified, and none crosses a partition.

**Why is accuracy below 70%?**  
Speaker-independent evaluation is harder and more honest than random recording
splits. The cross-dataset experiment directly demonstrates domain shift.

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
- Do not report the 70% target as achieved.
- Do not describe public acted-speech performance as natural-speech accuracy.
- Do not describe the reserved final test as evaluated until finalization has
  actually occurred.
