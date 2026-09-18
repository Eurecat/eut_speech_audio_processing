# Voice identity lifecycle: what the code actually does

Written after the AIMARA Sprint 4 benchmark reported "11 speaker ids for a 2-speaker
recording". Source of truth is the code at `2256de1` plus the diarization node logs of a
real run over the Android TCP bridge, not the README or earlier plans.

**Headline:** the eleven ids are not eleven speakers. Two of them own the two real people;
nine are provisional hypotheses, six of which the system consolidates by itself during the
same session. None of the provisional ones is ever written to MongoDB.

---

## 1. Where a published `speaker_id` comes from

`diarization.py:372`

```python
msg.speaker_id = eut_speaker_id.replace("EUT_", "")
```

So `speaker1` on ROS is `EUT_speaker1` internally. There is one namespace, not two. The
benchmark's `speaker1..speaker10` were `EUT_speaker1..EUT_speaker10`, plus `unknown` from
the ASR fallback at `asr_engine.py:750` when no speaker was active over the interval.

- `/speech_activity_detection` — published by the diarization node on VAD state changes
  (`_publish_speech_activity`), carrying the currently active `EUT_speakerN` and
  `speaker_id_confidence = engine.speaker_confidence`.
- `/speech_result` — published by the ASR node. The speaker comes from
  `asr_engine._resolve_speaker_for_interval(start, end)`: the speaker most active over the
  ASR interval, exactly the intended semantics.

## 2. Answers to the lifecycle questions

| Question | Answer (code reference) |
|---|---|
| When is a new id created? | `VoiceIdentityManager._create_identity`, only when `_assign_batch` returns no match and the caller passed `allow_create=True` |
| How much voice evidence is required? | Normal path: `redi_min_create_seconds = 1.5 s` (`redi_voice_engine._handle`). **Speaker-change probe: no minimum** — `_handle_change` does not pass `allow_create`, so a 1.0 s probe can create |
| Similarity threshold | `0.55` confirmed, `0.40` while young, `0.40` for windows under `short_window_seconds = 1.5`. Plus a top1-top2 margin of `0.06` |
| What if the score is ambiguous? | A near-tie fails the margin test, `_assign_batch` returns no match, and the caller **creates a new identity** |
| Is a MongoDB identity created immediately? | **No.** `_create_identity` never persists. Only `_add_embedding` calls `_save`, and `_persistable` requires `confirmed` **or** ≥2 embeddings and ≥`min_persist_seconds` (3.0 s) of clean speech |
| Are identities merged? | Yes, twice over: `_merge_similar_identities` (both sides ≥4 embeddings, similarity ≥0.80) and `_absorb_stray_identities` (young folded into a matured speaker at the full threshold plus margin) |
| Are low-confidence observations published as new speakers? | Yes — that is the behaviour that produced the extra ids |
| How stable is a confirmed id? | Stable. `stickiness_margin = 0.25` keeps a track on its speaker unless another beats it clearly, and the merge keeps the lower speaker number so ids do not churn |
| Does the stored profile get updated? | Yes. `_add_embedding` appends, recomputes the mean, and re-saves every `persist_every = 5` updates |
| Re-identification after silence? | Yes, within a session: `cleanup_inactive_identities` only drops identities that are stale **and** thin **and** unconfirmed **and** unpersisted |
| Re-identification across restart? | By construction yes — `MongoVoiceIdentityStore.load()` restores persisted identities and continues the numbering. **Not verified in the benchmark**, which ran with `REDI_USE_DATABASE=false` |

## 3. Why eleven ids appeared

From the node log of `spa_0018_2spk_clean` (CALLHOME Spanish, 2 speakers, 120 s):

- 10 identities created. **9 of them by the speaker-change probe**, one by the normal path.
- **6 of the 9 were near-ties**: the probe cleared the score bar against a known speaker
  (0.446–0.654, bar 0.400) and failed only the margin (0.003–0.060, minimum 0.06).
- **6 identities were absorbed during the session**: `6, 8 → 2` and `1, 3, 4, 7 → 9`, at
  similarities 0.556–0.794. The system converges on the two real people on its own.
- Scored against the reference: **11 ids published, 2 dominant, 9 transient**, attribution
  accuracy 70.9 %.

The probe exists to label a newcomer roughly a second after they start instead of waiting
for a full window. Fast turn-taking, as in a phone conversation, fires it constantly, and
each firing is a chance to create an id from one second of speech.

## 4. The obvious fix was implemented, measured, and rejected

Hypothesis: creating a speaker from an ambiguous match is never right — if it sounds like
one of two known voices, the answer is one of those two.

Implemented as `allow_create_when_ambiguous` on the manager plus
`redi_create_on_ambiguous_probe` on the node. Measured on the same file, same bytes:

| | published ids | dominant | transient | attribution | DER fair |
|---|---|---|---|---|---|
| current behaviour | 11 | 2 | 9 | **70.9 %** | **51.1 %** |
| guard enabled | 12 | 2 | 10 | 66.4 % | 56.5 % |

The guard fires six times and then the **same ambiguity reappears on the next full window**,
which creates the identity through the normal path on the same track. The ambiguity is a
property of short ReDimNet2 windows under fast turn-taking, not of the probe path.

**Left at the current behaviour by default.** The parameter and its unit tests
(`test/test_voice_identity_ambiguity.py`) stay so the hypothesis can be retested if
embedding windows or thresholds change.

## 5. The current bias is a deliberate, correct design choice

Creating on ambiguity **fragments** one person across several ids — repairable, and
`_absorb_stray_identities` does repair it (6 of 10 here).

Matching on ambiguity would **merge** two people into one id — **irreparable**: there is no
operation that splits a contaminated identity.

The code comment in `_assign_batch` states this outright: *"A near-tie means we cannot tell
the speakers apart; treating that as a match is how two people end up sharing an identity."*
Any future change must preserve that asymmetry.

## 6. Message API: no change needed

`SpeechResult` and `SpeechActivityDetection` already carry `speaker_id_confidence`, so an
"unresolved vs stable" distinction needs no new field. Two observations:

- `asr.py:242` hardcodes `msg.speaker_id_confidence = 0.0`. Populating it with the identity
  manager's match score would let `EutPersonManager.link_voice` weigh the link — it already
  accepts a `confidence` argument. Not done here: the score is not currently threaded
  through `asr_engine._resolve_speaker_for_interval`, so it is more than a minimal change.
- `asr.py:240` and `:243` overload `transcript_confidence` and `locale` as metric carriers
  for the Android bridge. That is already tracked as a P0 contract fix on the Android side
  and should be resolved there, not by adding fields here.

`EutPersonManager` is already conservative: `_attempt_vsad_match` links a voice only after
enough VSAD frames and a mean visual-speaking confidence above threshold, and refuses when
the face already has a voice. A transient id that appears briefly will usually not gather
enough frames to be linked. **The residual risk** is that a transient id wins that race
before the real identity consolidates, after which the correct id is dismissed for that face
("Face already has a voice … dismissing speaker permanently"). That is an argument for
reducing provisional publications in EutSpeech, not for changing EutPersonManager.

## 7. Recommended direction, in order of cost

1. **Delay publishing a probe's id** until the following full window confirms it. Costs one
   embed interval (~0.5 s) of speaker-activity latency for a newcomer; removes most
   provisional ids from ROS. The identity would still be created internally.
2. **Decide `SpeechResult`'s speaker from the whole utterance.** When ASR closes an
   utterance, embed the accumulated clean speech of that interval in one go and match that,
   instead of inheriting the label from whichever chunk was most active. A 3-second
   utterance embedding is far more reliable than a 1-second probe, and `SpeechResult` is
   what `EutPersonManager` consumes. This fits the pipeline naturally because the audio is
   already buffered, and it adds no latency beyond the existing end-of-utterance point.
3. Expose the match score on `speaker_id_confidence` (§6) so downstream consumers can weigh
   a link rather than trusting every id equally.

None of these were implemented in this sprint: real-world behaviour of the deployed system
is reported as good, and the measured evidence did not justify changing it blind.
