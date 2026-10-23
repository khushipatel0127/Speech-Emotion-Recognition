# SpeechSense Phase 2 experiments

Phase 2 trains a six-emotion model using the labels shared by RAVDESS and
CREMA-D: angry, disgust, fearful, happy, neutral, and sad. RAVDESS calm and
surprised recordings remain available to the historical eight-class model but
are deliberately excluded here rather than merged into another class.

Run dataset validation from the repository root:

```powershell
python experiments/phase2/dataset_validation.py
```

The command reads the original datasets without modifying them and writes:

- `reports/phase2/dataset_manifest.csv`
- `reports/phase2/split_assignments.json`
- `reports/phase2/dataset_validation.json`
- `reports/phase2/DATASET_VALIDATION.md`

The frozen split is dataset-aware and speaker-independent. The Phase 2 final
test partition must not be read during model selection. Grouped cross-validation
will be performed only on the train and validation development speakers.

Run the manifest regression tests with:

```powershell
python -m unittest experiments.phase2.test_dataset_validation -v
```

After installing the Phase 2 ML dependencies, extract the reproducible 84-value
handcrafted baseline features. The cache is append-only and resumes safely:

```powershell
python experiments/phase2/extract_features.py
```

The baseline comparison uses five-fold stratified grouped cross-validation on
training speakers. Its calibrated SVM also uses speaker-grouped inner folds.
It then scores candidates on the frozen validation speakers and intentionally
does not inspect final-test outcomes:

```powershell
python experiments/phase2/train_baselines.py
```

Validation-only transfer and gain robustness checks leave the reserved final
test speakers untouched:

```powershell
python experiments/phase2/cross_dataset_evaluation.py
python experiments/phase2/robustness_tests.py
```

The controlled Phase 2 volume experiment creates one deterministic augmented
copy for each training recording, never validation or test audio:

```powershell
python experiments/phase2/extract_augmented_features.py --jobs 2
python experiments/phase2/train_volume_augmented.py
```

WavLM is implemented as a frozen feature extractor with resumable 64-recording
shards. Because local CPU extraction is not practical on the target laptop, use
`SpeechSense_WavLM_Colab.ipynb` with a free GPU runtime. The notebook contains a
single configurable project path and does not embed credentials or personal
Drive locations. It runs:

```powershell
python experiments/phase2/extract_wavlm_embeddings.py --device auto
python experiments/phase2/train_embedding_classifier.py
```

Only after both validation reports exist, run the guarded one-time finalization:

```powershell
python experiments/phase2/finalize_model.py
```

The finalizer selects the higher validation macro F1 across approaches, refits on
train plus validation speakers, evaluates the reserved test once, and saves a
versioned, non-deployed model. It refuses to overwrite an existing final report.

For natural-speech evaluation, copy `real_world_manifest.template.csv` to
`reports/phase2/real_world_manifest.csv`, enter human-supplied labels and file
paths, then run:

```powershell
python experiments/phase2/evaluate_real_world.py
```

The workflow converts every listed file through the deployed FFmpeg path and
records predictions, uncertainty flags, errors, accuracy, macro F1, and a
confusion matrix. An empty manifest produces no claimed score.
