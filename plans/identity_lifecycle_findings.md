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

## 5 bis. Is the speaker already decided from the complete utterance?

Half of it is, and the half that is missing is the one that matters.

**What already happens.** `asr_engine._resolve_speaker_for_interval(start, end)` takes the
whole ASR utterance interval and picks the speaker with the largest overlap in
`speaker_timeline`, falling back to the last speaker seen before the interval and then to
the nearest one. So the *vote* is already taken over the complete utterance, exactly as the
`/speech_result` semantics require.

**What does not happen.** The entries being voted over are labels produced from
`redi_max_embed_seconds = 2.0` windows and 1.0 s change probes. Aggregating them inherits
their errors: if the chunks were ambiguous, the majority vote over ambiguous chunks is still
ambiguous. **Nothing ever embeds the complete utterance audio in one go and re-matches it**
against the identity database, and a 3-second utterance embedding is a far better ReDimNet2
input than any 1-second probe.

**Latency cost: none.** This is worth stating because it was the objection. The idea does not
delay anything:

- `/speech_activity_detection` keeps streaming per chunk exactly as today, in real time;
- the extra embedding happens at end-of-utterance, a point the pipeline already reaches and
  already does work at (that is when ASR runs);
- only `/speech_result`'s `speaker_id` changes, and it is published after ASR anyway.

**The real cost is architectural**, which is why it is not implemented here: ASR and
diarization are separate ROS 2 nodes. The ASR node holds the utterance audio but not the
ReDimNet2 model or the identity manager; the diarization node holds both but does not know
where ASR decided the utterance boundaries are. Closing that needs either a
`ResolveSpeakerForInterval` service on the diarization node (cleanest, keeps one identity
owner) or moving utterance-level embedding into the diarization node driven by an
ASR-published interval. Either is a real design change and should be measured, not assumed.

## 6. Message API: no change needed — and `speaker_id_confidence` is now populated

`SpeechResult` and `SpeechActivityDetection` already carry `speaker_id_confidence`, so an
"unresolved vs stable" distinction needs no new field. Two observations:

- `asr.py` used to hardcode `msg.speaker_id_confidence = 0.0`, which downstream cannot tell
  apart from "identified with zero confidence". **Now implemented.** The diarization node
  already publishes its match score on `SpeechActivityDetection`; the ASR node was dropping
  it. It is now carried into `speaker_timeline` and combined at publish time as

  ```text
  speaker_id_confidence = coverage_of_utterance x mean_identity_match_score
  ```

  Both factors have to hold for the label to be worth anything: a speaker identified with
  certainty who held a third of the utterance is a weak label, and so is one who held all of
  it but was barely recognised. `-1.0` is emitted when the backend reported no score, and
  must be read as *unavailable*, never as *low*. `EutPersonManager.link_voice` already takes
  a `confidence` argument and can now be given a real one. Unit tests:
  `test/test_speaker_confidence.py`.
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

1. ~~**Delay publishing a probe's id**~~ — **rejected**: it costs about half a second of
   speaker-activity latency for a newcomer, and keeping the activity signal real time is a
   product requirement. The provisional ids stay visible on
   `/speech_activity_detection`; the fix belongs on `/speech_result` instead, which is what
   `EutPersonManager` consumes.
2. **Decide `SpeechResult`'s speaker from the whole utterance** — the recommended next
   step, and the one with no latency cost (§5 bis). It needs a service boundary between the
   ASR and diarization nodes, so it is a design change rather than a patch.
3. ~~Expose the match score on `speaker_id_confidence`~~ — **done**, see §6.

Item 3 is implemented. Item 1 is rejected on latency grounds. Item 2 is the recommended
next step and needs a design decision about the node boundary, so it is deliberately left
unimplemented rather than rushed.
