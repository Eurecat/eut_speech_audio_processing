# Utterance benchmark: WER and speaker attribution

This benchmark plays an audio file from `src/audio_stream_manager/recordings/aimara-suite/` through `Docker/docker-compose_mp3.yaml` and records every `/speech_result`. It then compares each ground-truth (GT) utterance with what the pipeline said, and with the speaker it was assigned to.

## Files

| path | what |
|---|---|
| `gt/<stem>.json` | GT utterances: `start`, `end`, `speaker` (A, B, ...), `text`, `overlap`, `verified` |
| `rttm/<stem>.rttm` | CallHome speaker turns copied from `~/aimara-bench/benchmarks/data/suite/` |
| `hyp/<stem>.json` | Captured pipeline output: transcripts plus `/speech_activity_detection` events, in seconds of audio |
| `results/<stem>.json` | Per-utterance scores and a word diff |
| `results/REPORT.md` | Summary table, then one table per file |
| `results/report.html` | Same data as a page. Open it in a browser (no server needed). |

## The GT text is silver. Review it.

No text transcripts exist for these files. The AIMARA suite has only speaker turns (RTTM), and aimara-bench scored DER only. `make_silver_gt.py` cuts the audio on each GT turn and transcribes each piece offline with faster-whisper large-v3-turbo, which is a different model from the Parakeet streaming pipeline.

To review the GT, edit `text` in `gt/<stem>.json` and set `"verified": true` on each utterance you check. Do not change `text_silver`. It keeps the machine version. Until you review the text, the WER column measures how far streaming Parakeet is from offline Whisper, not true WER.

The speaker labels are real GT:

- CallHome: from the RTTM.
- Weather file: VAD segments with the known order `A,B,B,C,C,B,A,C,B,A`.

Utterances flagged `overlap` have another speaker talking at the same time, so their silver text is the least reliable.

## Run

```bash
cd benchmarks/utterance_eval
./run_one.sh wer_es__es_es_weather_wer                 # 35 s
./run_one.sh callhome_spa_snr20__spa_0019_4spk_snr20   # 120 s, 4 speakers
./run_one.sh callhome_spa_snr20__spa_0018_2spk_snr20   # 120 s, 2 speakers
```

`run_one.sh` uses `ROS_DOMAIN_ID=77` and compose project `utterance_eval`. It refuses to start if `speech_recognition`, `mongodb` or `mp3_audio_source` is already running.

After editing a GT file, rescore without replaying the audio:

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/eval" --entrypoint python3 eut_audio_arm:jazzy /eval/score.py
```

To add a new file, generate its silver GT inside the image with `--rttm` or `--speaker-order` (see `make_silver_gt.py --help`), then run `run_one.sh`.

## How it scores

- **WER**: all GT words (in time order) are aligned against all published words with one alignment. Each GT utterance gets the hypothesis words that landed on it, so the ASR can split sentences differently from the GT without being penalised. Text is lowercased, and punctuation and accents are removed. Digits become words (`17` = `diecisiete`).
- **Speaker**: pipeline ids (`speaker1`, ...) are mapped one-to-one onto GT labels by Hungarian matching on aligned word counts. Each utterance gets one status:
  - `ok`: its majority mapped speaker is the GT speaker.
  - `wrong`: the pipeline assigned it to another speaker.
  - `unknown`: the pipeline published it as `unknown`.
  - `missed`: no word of it was transcribed.
  
  Accuracy excludes `missed` utterances.
- **DER** (pyannote.metrics, 0.25 s collar) is computed two ways. *activity* uses the diarization `/speech_activity_detection` labels. *ASR utt* uses the published utterances, placed in time using `audio_ms`, `proc_ms` and the 0.25 s silence gate.
