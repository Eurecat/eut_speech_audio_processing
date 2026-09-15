# REDI voice identity — status, history and work plan

Status: living document. Rewritten 2026-09-15 when the project switched from a diart-coupled
REDI backend to an independent one.

Companion document: `plan_REDI_diart.md` (the architecture spec). Read that first.

**Scope rule:** the `diart` backend must keep working exactly as the user knows it unless a change
is explicitly switched on. Every diart-side change in this document is behind a flag that
defaults to off, and `diarization_engine.py` is byte-identical to `HEAD`.

---

## 1. Current state

| Piece | File | State |
|---|---|---|
| Shared identity layer | `speech_recognition/voice_identity_manager.py` | **Done, unit tested** |
| REDI backend (independent, no diart) | `speech_recognition/redi_voice_engine.py` | **Done, unit tested, 5 mp3 runs — see §5 Steps 2-4** |
| diart backend + shared identity layer | `speech_recognition/diart_identity_engine.py` | **Done, A/B'd, behind `diart_use_voice_identity_manager: False`** |
| Legacy diart backend | `speech_recognition/diarization_engine.py` | **Unchanged — identical to HEAD** |
| Old diart-coupled REDI | `redi_diarization_engine.py`, `redi_speaker_identity.py`, `test/test_diarization.py` | **Deleted** (recoverable from git history) |

### Backend selection

| `diarization_backend` | `diart_use_voice_identity_manager` | Engine |
|---|---|---|
| `diart` | `False` (default) | `DiarizationEngine` — legacy, untouched |
| `diart` | `True` | `DiartManagedIdentityEngine` |
| `redimnet2` | n/a | `RediVoiceEngine` |

The log line `Selected diarization backend: <backend> (engine=<class>)` now names the engine
class. **Check it before trusting any result.** For a long stretch this project evaluated
"REDI" results that were really the diart backend: `Docker/docker-compose_mp3.yaml` passes
`diarization_backend:=${DIARIZATION_BACKEND:-diart}` explicitly, and an explicit launch argument
overrides the launch file's `default_value`. The live-mic `Docker/docker-compose.yaml` passes no
backend argument, so it uses the launch file default instead.

---

## 2. Unit tests — run these before any docker test

```bash
cd src/speech_recognition
python3 -m pytest test/test_voice_identity_manager.py test/test_redi_turn_segmenter.py \
    -q -p no:cacheprovider --noconftest
```

No ROS, no GPU, no model; under a second. Current result: **49 passed, 1 xfailed** (plus `test_asr_chunk_boundaries.py`).

This step exists because almost every regression in §4 would have been caught for free by a unit
test, and instead each cost a 2-3 minute docker run to discover.

### What the synthetic tests model

A speaker is a random unit vector in 192 dimensions; an utterance is that vector plus Gaussian
noise. Measured similarity per noise level:

| noise | same-speaker pair | utterance vs true speaker | different speakers |
|---|---|---|---|
| 0.05 | 0.68 | 0.82 | ~0.0 |
| 0.08 | 0.46 | 0.67 | ~0.0 |
| 0.11 | 0.29 | 0.54 | ~0.0 |

Real ReDimNet2 same-speaker matches on the test mp3 measured 0.60-0.77 (an upward-biased sample,
§4), so **0.05 is the realistic regime**; 0.11 is a stress case marked `xfail`.

**Known gap in the synthetic model:** real different-speaker embeddings are not orthogonal. Speech
embeddings share structure, so different people plausibly sit 0.2-0.4 apart rather than 0.0,
which makes the synthetic separation easier than reality. The top1-top2 margin is what should
protect against that. Check with the rejection logs from a real run (§5).

### Design bugs the tests found in the manager, before any docker run

1. **Young identities never grew.** A new identity matched at the young threshold but was only
   *updated* at the full threshold, so it kept being matched, never gained samples, never
   converged and never confirmed. Fixed: an identity learns at the bar it matches at.
2. **Confirmed identities were starved.** `update_threshold` 0.62 sat above the expected score
   against a 3-sample mean (~0.56), so updates were refused and the mean never improved. Fixed:
   `update_threshold` defaults to `similarity_threshold`.
3. **Strays were permanent.** A noisy utterance that missed its speaker spawned a one-sample
   identity, and merging only considered identities with ≥4 samples, so strays were never
   removed. Fixed: `_absorb_stray_identities` re-checks young identities against speakers that
   have since matured, using the normal threshold and margin.

After these, at noise 0.08 the system ends with exactly the right speakers: 9 identities created,
6 strays absorbed, 3 remaining.

### A known, accepted cost: transient stray labels

A stray's label is published when it is created. It is absorbed into the right speaker later,
but the transcript already published under the stray id is not revised. At noise 0.08 the worst
speaker's live label purity is 0.81; at the realistic 0.05 it is 1.00. Removing this entirely
requires delaying speaker labels, which costs latency. That is a product decision, not a tuning
one, and has not been made.

---

## 3. Fixed and must be kept

### 3.1 `redi_repository` pinned to a commit

`PalabraAI/redimnet2:v1.0.0` (2026-03-04) predates the `b6`/`vb2+vox2+cnc2_v0` checkpoint, which
was added upstream in commit `e8bfa167` (2026-04-23) and uploaded to the same v1.0.0 release
without a new tag. Its stored `model_config` carries `agg_gnorm`, which the tag's
`ReDimNet2Wrap` rejects:
`TypeError: ReDimNet2Wrap.__init__() got an unexpected keyword argument 'agg_gnorm'`.
Pinned to `c5bbe0b76e37df698c403f8844e41304ceab6307`. It reproduced on every cold torch-hub
cache, which means every `--force-recreate`, because that cache is not a volume.

### 3.2 `asr_engine.py` lost-utterance race

A short pause followed by more speech lost the second utterance entirely and speech activity
never went false. The silence timer thread was only spawned if the previous one had finished,
and a late reset from a slow Whisper call zeroed the next segment's start time. Fixed with a
segment-dispatched flag, an unconditional timer thread and a guarded reset. Confirmed by the
user.

---

## 4. Abandoned — the diart-coupled REDI backend

Kept so nobody rebuilds it.

The old REDI backend ran diart's segmentation **and** diart's `OnlineSpeakerClustering`, swapping
only the embedding model. Every attempt failed, for structural reasons:

- `delta_new` is a cosine **distance** threshold. A new cluster is created only when an
  embedding is further than `delta_new` from every centroid. At the shipped 0.90 that is
  effectively unreachable for speech, so diart built one cluster for the whole conversation.
- diart's centroids are **unnormalised running sums** (`self.centers[g] += emb`), and assignment
  is an argmin over them. An established centroid wins every assignment, grows, and wins again.
  A centroid that stops being assigned freezes and scores exactly 1.000 against itself forever.
- So no threshold fixes labelling: `delta_new` only gates *creating* clusters, never *which*
  existing cluster wins.

| Attempt | Result |
|---|---|
| Overlap sentinel id | Reverted by user — much worse |
| Filter embeddings to window-active tracks | Worse: starved the second speaker's track |
| `delta_new` 0.75 / 0.80 / 0.85 / 0.90 | 7 / 3 / 2 / 1 ids; none fixed the merged A/B dialogue |
| `rho_update` 0.1 | Unfroze a dead centroid; did not fix labelling |
| Bypass diart clustering, per-slot embeddings | 19 ids |
| Same, thresholds recalibrated | 21 ids — calibrated on a biased sample |

**Measurement lesson that still applies:** the old code logged `score=1.000` when it *created* an
identity, so failed match scores were never visible and every calibration was fitted to
successes only. `VoiceIdentityManager` logs the nearest identity, its score, the required score
and the margin on every creation. Calibrate from those lines.

**Second measurement lesson:** the pipeline runs in real time against wall-clock audio, so the
same input gives noticeably different output between runs. One run is not evidence for a small
difference.

---

## 5. Work plan

### Step 1 — A/B the diart identity layer — DONE

Same mp3, back to back in one session, `diart_use_voice_identity_manager` False vs True.

- Both ran clean with the correct engine class.
- 13 of 14 transcript lines identical; 7 ids each. The one difference moved `"Alright, fine."`
  from one wrong id to another wrong id.
- **No regression, no improvement on this clip.** Expected: on the diart path the identity layer
  can only fix *track → identity* mistakes, and every error on this clip is diart mislabelling
  frames underneath it (B spread across three diart tracks). The layer's value — no permanent
  track pinning when diart reuses a track index, merging, persistence — is structural and shows
  on long live sessions rather than a 43s clip.

Flag stays `False` by default. Enabling it is the user's decision.

### Step 2 — real runs of the REDI engine — DONE, 3 runs

All three: clean, no errors, `engine=RediVoiceEngine`, no diart loaded. For reference the legacy
diart backend produced **7** ids on this clip; the abandoned diart-coupled REDI produced 19-21.

| Run | Change | ids | Notes |
|---|---|---|---|
| 1 | first run | 4 | GT lines 2-5 (B,B,A,B) one turn, one label. `speaker2` learned from that blend, so B's next clean turn scored **0.214** against it and split off |
| 2 | stickiness floor, 2s window, mixed turns not learned | 6 | Label switched **inside** the unpaused turn and ASR split it — line 4 separated from lines 2-3 for the first time. But B fragmented: same speaker scored 0.325-0.375 vs young identities, under the 0.40 bar |
| 3 | `redi_identity_young_threshold` 0.30 | 5 | Best so far. Lines 2-3 one clean `speaker2` transcript; line 4 separated from them (still merged with line 5); B's line 7 returned as a new id |

**Unbiased real-audio calibration** (from `New voice identity` rejection logs):

| Comparison | Score |
|---|---|
| Different speakers | 0.05-0.21 |
| Same speaker vs a *clean, matured* identity | 0.65-0.92 |
| Same speaker vs a *young* identity (seeded from 0.8s) | 0.27-0.38 |

Separation against matured identities is large. Against young identities it **overlaps** the
different-speaker range.

#### Bugs found by the real runs, all fixed with a unit test

1. **Stickiness had no absolute floor.** When the speaker changed inside one track and the
   newcomer was unknown, the previous identity was also the best candidate, so
   `previous >= best - margin` pinned the track regardless of score. Fixed: the previous identity
   must still score ≥ `young_identity_threshold`.
2. **Mixed turns poisoned identities.** A turn whose label changed while running was still
   learned. Fixed: such a turn is labelled but never learned (`mixed, not learned` in the log).
3. **The embedding window was too long.** 6s blended a whole A/B exchange. Now 2s, refreshed
   every 0.5s, so the label follows whoever is talking now and ASR splits on the change.

#### Threshold tuning has reached its limit — do not keep lowering it

The rejected scores sit just under **whatever** threshold is set: 0.325-0.375 against 0.40 in run 2,
then 0.274-0.295 against 0.30 in run 3. That is a selection effect — lowering the bar moves the
marginal cases down with it. Run 3 already rejected a **0.207** with margin 0.023, i.e. the
marginal cases are entering the different-speaker range. Lowering further will start merging
different people.

### Step 3 — seed quality, continuous learning, ASR repeated text — DONE (run 4)

Reported by the user from a real run: repeated text across a speaker split, and B returning under a
new id (`speaker4 created: nearest=speaker3 score=0.295`). The nearest identity was **A**, so
B's own identity scored below 0.21 against B — a bad seed, not a threshold miss.

Fixes, each with a unit test (including a direct reproduction of the reported sequence):

1. **`redi_min_create_seconds: 1.5`** — a window may match an existing speaker at any length but
   may only create one from ≥1.5s. Identities were being seeded from 0.8s windows.
2. **Stable-window learning** — identities learn whenever two consecutive windows of a turn agree
   on the speaker, at most once per `max_embed_seconds` of speech. Previously only whole
   unmixed turns were learned, and conversational turns are almost always mixed, so B's identity
   never grew past its 0.8s seed. Windows straddling a speaker change are never learned
   (`learn_if_assigned_to`).
3. **ASR repeated text** (`asr_engine.py`, shared by both backends — a bug on diart too) — after a
   mid-speech split the next chunk re-prepended `pre_buffer_duration` (0.5s) of audio already
   published. Pre-buffer now applies only at a genuine speech onset (`_segment_has_onset`). The
   forced long-chunk split had the same bug.
4. `redi_identity_young_threshold` restored to 0.40 — 0.30 only compensated for fix 1's cause.

Run 4 result: **3 identities for 3 speakers**; B's return matched B's original id; B confirmed
(`required=0.550`); no repeated words at splits.

Still wrong in run 4:
- A's line 4 is cut ~1.5s late at the A/B boundary, and B's lines 5-6 carry A's label.
- A's line 1 (~1s) shares the intro speaker's id: too short to create a speaker, so it inherits
  the previous label — the accepted cost of fix 1.

### Step 4 — speaker-change lag at boundaries — DONE (run 5)

The label flipped only once a 2s window was dominated by the new speaker: 1.5s late at B→A, and
A→B inside one turn was never detected (the window straddling it even got learned into A).
`diarization_offset: -1.0` in `asr_params.yaml` is shared with diart and was not touched.

**Measured first, offline.** The mp3 was replayed through `RediVoiceEngine` with the real
ReDimNet2 model, cached Silero VAD and Whisper word timings (deterministic, ~1s per config instead
of a 2-3 minute docker run; it reproduced run 4's scores exactly). On real embeddings:

| window | same speaker vs matured identity | different speaker |
|---|---|---|
| 0.8s | 0.40-0.75 | ≤0.39 |
| 1.0s | 0.42-0.82 | ≤0.36 |
| 2.0s | 0.59-0.94 | ≤0.36 |

A 1.0s probe vs its speaker's identity dropped to 0.14-0.24 about 1s after each real change.

Fixes, each with unit tests:

1. **Change probe** (`redi_probe_seconds: 1.0`, `redi_change_threshold: 0.35`) — each refresh also
   embeds the last 1.0s of speech. Below 0.35 against the current label, the turn is **split** at
   the probe start (`TurnSegmenter.split`): the rest of the turn is a new segment (`turn7.2`) with
   its own track id, so stickiness cannot pin it and its windows never contain the old speaker.
   The probe labels the new segment immediately; ~1s after the change, which the ASR offset of
   -1.0 lines up. Observations queued before the split are dropped.
2. **Probe may create a speaker, then reseed it** — the change itself is the evidence of a new
   voice. The seed is replaced by the segment's longer windows (`reseed_identity`) until it is a
   full 2s window, so a 1s seed never becomes a permanent weak reference (the Step 3 failure).
3. **Short-window match bar** (`redi_identity_short_window_seconds: 1.5`,
   `..._threshold: 0.45`) — windows under 1.5s match a known speaker at 0.45 instead of 0.55, per the
   table above (later lowered to 0.40, see Step 4b). Diart's managed identity path leaves it
   disabled (default `0.0`).

Offline, whole 12-minute mp3: one extra identity compared with no probe (a change scoring 0.078 at
392s); no false splits in single-speaker speech. `probe_seconds: 0.8` switched earlier but split
speakers into extra ids.

Run 5 (live docker) vs ground truth:

| # | Speaker | Run 4 | Run 5 |
|---|---|---|---|
| 0 | C | s1 | s1 |
| 1 | A | s1 ✗ | s1 ✗ |
| 2-3 | B | s2 | s2 |
| 4 | A | cut mid-sentence, first half B ✗ | **whole line s3** |
| 5 | B | s3 ✗ | **s2** |
| 6 | B | s3 ✗ | **s2** |
| 7 | B | s2 | s2 |
| 8 | A (overlap) | s2 | **s3** |

3 identities for 3 speakers, no repeated text.

**Line 1 cannot be fixed in a streaming label.** Silero marks only 0.32s of "You're a jerk, Tom."
as speech (probabilities 0.0-0.3 through most of it), B follows 0.13s later with no pause, and A
has not been heard yet. Even the full 0.85s of raw audio, which does score 0.515 against A's final
identity, arrives before A's identity exists. Labelling it A needs hindsight: re-attributing an
already published transcript once A is known, which the ASR output contract does not support.

Also still visible: "Look, Celia," is missing from the transcript in runs 4 and 5. B's first label
arrives ~0.6s into B's speech (B is new, so it waits for `redi_min_create_seconds`), and the ASR
flush at that label cuts through those words. An ASR chunk-edge issue, not a wrong label.

### Step 4b — MongoDB persistence across restarts — DONE

Same idea as EutHRIFaces: speakers survive a restart and keep their `EUT_speakerN`.

- **Switch:** `redi_use_database: True` in `diarization_params.yaml`, separate from `use_database`
  (legacy diart, still `False`). `docker-compose_mp3.yaml` forces it off through
  `redi_use_database:=${REDI_USE_DATABASE:-false}`, so ground-truth runs start from zero; run with
  `REDI_USE_DATABASE=true` to test persistence. The launch argument is empty by default (yaml wins).
- **Where:** database `speaker_recognition`, collection `voice_identities`, filtered by
  `model_key=redimnet2:b6:lm:vb2+vox2+cnc2_v0`. Legacy diart uses collection `speakers`; embeddings
  of different models are never mixed. Reset REDI speakers with
  `db.getSiblingDB("speaker_recognition").voice_identities.deleteMany({model_key: "redimnet2:b6:lm:vb2+vox2+cnc2_v0"})`.
- **What is saved:** confirmed speakers, and unconfirmed ones with ≥2 embeddings from ≥3.0s of
  speech (`min_persist_seconds`). A single short seed is never saved. The `confirmed` flag is
  stored and restored, so a thin speaker reloads with the young match bar. Persisted speakers are
  exempt from inactive cleanup (a reloaded one always looks stale).
- **When:** first time a speaker becomes persistable, every 5 updates, **at every turn end**, and at
  shutdown. Faces only save at shutdown; here the mp3 runs showed the shutdown flush cannot be
  relied on when the container is stopped, so turn-end saves carry it. A database error is logged
  and never stops diarization.

Found while testing: once A was confirmed (the second session), the overlapped 1s probe of
"Whatever, Tom" scored 0.41-0.43 against A, under the 0.45 short-window bar, and created a duplicate
id. That would also happen in any long single session. `redi_identity_short_window_threshold`
lowered to **0.40** (1.0s windows: different speakers ≤0.36). Offline over the whole mp3: 6 ids
instead of 7, conversation labels unchanged.

Runs (mp3, `REDI_USE_DATABASE=true`, database wiped first):

| Run | Loaded | Result |
|---|---|---|
| e | none | B=s2, A=s3 created and persisted (2 embeddings, 4.0s each) |
| f, g (bar 0.45) | s2, s3 | A and B matched their saved ids; "Whatever, Tom" created s5 |
| h (bar 0.40) | s2, s3 (now confirmed) | every A and B line on its saved id, **no new id created** |

The test speakers were deleted from the database afterwards.

### Step 3b — old diart-coupled REDI code — DONE

Deleted `redi_diarization_engine.py`, `redi_speaker_identity.py`, `test/test_diarization.py`;
README tree updated. The one behaviour of the old tests not already covered (loading persisted
identities) is `test_persisted_identities_are_loaded_and_matched_on_startup`. Session edits to the
deleted engine are backed up in the scratchpad as `redi_diarization_engine_session_edits.patch`.

### Step 5 — later, only if needed

- Retroactive re-attribution of short lines once their speaker is known (line 1 above). Needs an
  ASR-side contract change.
- Delayed label publication, if transient stray labels (§2) prove unacceptable.
- PLDA/PSDA scoring instead of cosine.

---

## 6. Test protocol

### 6.1 Reading a run

```bash
grep -n "Selected diarization backend" run.log      # backend AND engine class
grep -niE "error|exception|traceback" run.log | grep -v mongodb
grep -n "Transcript:\|Active eut_speaker_id\|New voice identity\|REDI turn\|Merged\|Absorbed" run.log
```

`HF_TOKEN not set` is expected noise on the diart path: the pyannote weights are already cached
in the bind-mounted `weights_pyannote` directory. The REDI engine does not use pyannote at all.

### 6.2 Ground truth — `src/audio_stream_manager/four_speakers_GT`, first 43s

| # | Line | Speaker |
|---|---|---|
| 0 | "We have main engine start. Four, three, two, one." | C |
| 1 | "You're a jerk, Tom." | A |
| 2 | "Look, Celia, we have to follow our passions." | B |
| 3 | "You have your robotics and I just want to be awesome in space." | B |
| 4 | "why don't you just admit that you're freaked out by my robot hand?" | **A** |
| 5 | "I'm not freaked out, but it's…" | B |
| 6 | "All right, fine." | B |
| 7 | "I'm freaked out. I'm having nightmares..." | B |
| 8 | "Oh, whatever TOM" | A — overlaps line 7, acceptable to lose |

Line 4 is the discriminating case. Score by consistency, not absolute ids: A's lines (1, 4) share
one id, B's lines share a different one, and C differs from both.

REDI splits turns on **pauses** and, inside a turn, on the change probe (Step 4). A line under ~1s
by a speaker not yet heard, followed with no pause by someone else (line 1), keeps the previous
label. Check the audio before counting that as a regression.
