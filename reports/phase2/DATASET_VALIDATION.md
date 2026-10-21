# SpeechSense Phase 2 Dataset Validation

Generated: 2026-10-21T11:12:39.257862+00:00

Status: **PASSED**

## Dataset inventory

| Dataset | Discovered recordings | Model-eligible recordings | Speakers |
|---|---|---|---|
| CREMA-D | 7442 | 7436 | 91 |
| RAVDESS | 1056 | 1055 | 24 |

Total discovered six-class recordings: **8498**  
Total model-eligible recordings: **8491**  
Total dataset-prefixed speakers: **115**

## Emotion distribution

| Dataset | Emotion | Recordings |
|---|---|---|
| RAVDESS | angry | 192 |
| RAVDESS | disgust | 192 |
| RAVDESS | fearful | 192 |
| RAVDESS | happy | 191 |
| RAVDESS | neutral | 96 |
| RAVDESS | sad | 192 |
| CREMA-D | angry | 1270 |
| CREMA-D | disgust | 1270 |
| CREMA-D | fearful | 1270 |
| CREMA-D | happy | 1270 |
| CREMA-D | neutral | 1086 |
| CREMA-D | sad | 1270 |

## Frozen speaker-independent split

| Split | RAVDESS | CREMA-D | Total recordings | Speakers |
|---|---|---|---|---|
| train | 703 | 5229 | 5932 | 80 |
| validation | 176 | 1065 | 1241 | 17 |
| test | 176 | 1142 | 1318 | 18 |

Random seed: `42`. Speaker assignments are saved in `split_assignments.json`.
All recordings from a speaker are assigned to exactly one split.

## Audio properties

- Sample rates: `{"16000": 8498}`
- Channel counts: `{"1": 8498}`
- Duration range: 1.268–5.272 seconds
- Mean duration: 2.689 seconds
- Median duration: 2.569 seconds
- Total duration: 6.35 hours

## Integrity checks

- Duplicate manifest path groups: **0**
- Exact duplicate audio-content groups: **4**
- Recordings excluded because of exact duplicates: **7**
- Duplicate-content groups crossing splits: **0**
- Speaker overlap between splits: **0**

Validation errors:

None.

## Method notes and limitations

- RAVDESS calm and surprised are excluded; no label merging is performed.
- Previously examined RAVDESS actors are prohibited from the new final test split.
- The test split remains a public-dataset holdout and is not equivalent to natural-speech evaluation.
- Original sample rates are recorded here; model feature extraction will resample to 16 kHz mono.
- SHA-256 hashes cover decoded PCM frame bytes and identify exact audio-content duplicates.
- Conflicting-label duplicate groups are excluded in full; one redundant same-label copy is excluded.
