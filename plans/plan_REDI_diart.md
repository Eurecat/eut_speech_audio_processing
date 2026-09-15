# REDI voice identity — architecture

Status: design spec. Rewritten 2026-09-15 after the diart-coupled approach was abandoned.

Companion document: `plan_REDI_fixes.md` (status, history, work plan, test protocol).

---

## 0. What changed and why

The first REDI implementation kept diart in the loop: diart did segmentation **and** online
clustering, ReDimNet2 only replaced diart's embedding model, and our identity layer sat on top of
diart's local tracks. That was a mistake. It was never the intent of this project, and it does
not work — see `plan_REDI_fixes.md` §4/§6 for the measured evidence. Summary of the failure:

- diart's `OnlineSpeakerClustering` decides who is who **before** our code sees anything.
- Its `delta_new` is a cosine **distance** threshold; the shipped 0.90 is effectively unreachable
  for speech embeddings, so diart built **one cluster for an entire conversation**.
- Its centroids are **unnormalised running sums** (`self.centers[g] += emb`), so an established
  centroid wins every subsequent assignment and starves the others — a lock-in that no threshold
  value fixes. Sweeping 0.75 / 0.80 / 0.85 / 0.90 never produced correct labels.
- Result: everything merges into one speaker id, which is exactly the complaint this project
  started from.

**New approach: REDI does not use diart at all.** It mirrors the face recognition package, which
already solves this same problem well for faces.

---

## 1. The model we are copying

`EutHRIFaces/face_recognition/face_recognition/identity_manager.py`.

Its shape, which we replicate for voices:

```
detector + tracker  ->  {track_id: face_crop}
                              |
                     embedding model per crop
                              |
              {track_id: embedding}   (a batch, per frame)
                              |
        IdentityManager.process_new_embedding_batch()
                              |
              {track_id: (U1, confidence)}
```

The properties worth copying, all of which our current `SpeakerIdentityManager` lacks:

| Property | Why it matters for voices |
|---|---|
| **Batch, exclusive 1:1 assignment** | Two speakers in one window can never both be assigned the same identity. Our per-track `assign()` calls have no such guarantee |
| **Stickiness margin** (`track_identity_stickiness_margin`) | Keeps a track on its previous identity unless another beats it by a margin. Prevents flip-flop without hard-pinning |
| **Identity merging** (`_check_and_perform_merges_batch`) | Recovers from fragmentation: if one person spawned two identities early, they get merged once both are well populated. Nothing in the current voice code can ever undo a split |
| **`all_embeddings` + `mean_embedding` (+ optional EWMA)** | A real population per identity, not one drifting vector. Matching can use mean / recent / top-confidence combined |
| **Inactive cleanup** (identities *and* track mappings) | Stops a stale `track_id -> identity` mapping being inherited by a new, different speaker |
| **Mongo persistence with load-on-start** | Same lifecycle the faces already have |

---

## 2. Target architecture

```
                      ┌──────────────────────────────────────┐
                      │       VoiceIdentityManager           │
                      │  backend-agnostic, face-style        │
                      │  - VoiceIdentityCluster              │
                      │  - batch exclusive assignment        │
                      │  - stickiness margin                 │
                      │  - merging / cleanup                 │
                      │  - MongoDB persistence               │
                      └───────────────▲──────────────────────┘
                                      │
                {track_id: embedding} │ {track_id: (EUT_speakerN, confidence)}
                                      │
              ┌───────────────────────┴────────────────────────┐
              │                                                │
   ┌──────────────────────┐                      ┌─────────────────────────────┐
   │  diart frontend      │                      │  redi frontend              │
   │  (legacy, unchanged  │                      │  (new, independent)         │
   │   segmentation +     │                      │  VAD turn segmentation      │
   │   clustering)        │                      │  + ReDimNet2 per turn       │
   │  -> local track ids  │                      │  -> turn ids                │
   └──────────────────────┘                      └─────────────────────────────┘
```

**Both backends share the identity layer.** That is what makes "if I go back to the diart backend
I have the same things" true: the same `EUT_speakerN` semantics, the same merging, the same
persistence, the same stickiness. The backends differ only in *where embeddings come from*.

### 2.1 The REDI frontend, concretely

No segmentation model, no clustering. The turn is the unit:

```
/vad  ->  speech / not speech
            |
   turn = contiguous speech between silences
            |
   accumulate that turn's audio (already buffered for ASR)
            |
   ReDimNet2 embedding of the turn's clean speech
            |
   VoiceIdentityManager.process_new_embedding_batch({turn_id: emb})
            |
   EUT_speakerN  ->  /speech_activity_detection
```

A turn is the voice analogue of a tracked face: a bounded observation of one person that we embed
once and hand to the identity manager.

**Known limitation, accepted for v1:** if two people speak inside one VAD turn with no pause
between them, that turn yields one embedding and therefore one identity. Resolving *who spoke
when inside a turn* is what a segmentation model is for, and it is explicitly out of scope for
v1. It can be added later as a mid-turn split (§4) without changing the identity layer.

This limitation is acceptable because it fails **safe**: it under-segments (merges) rather than
inventing speakers, and the merge is visible and explainable rather than the current situation
where diart silently assigns everything to one cluster for reasons we cannot control.

---

## 3. Design rules carried over from the original brief

These were right in the original plan and survive unchanged:

1. **MongoDB is not the hot loop.** Load identities once at startup into RAM, match in RAM with
   a matrix multiply, write back only on create / significant update / merge / shutdown.
2. **Do not represent a person with one centroid.** Keep a population of embeddings plus a mean;
   voice varies with mic distance, room, noise, emotion, whisper vs shout.
3. **Matching is not `if cosine > X`.** Use absolute score **and** the top1-top2 margin. `0.74 vs
   0.73` is a fundamentally different situation from `0.74 vs 0.31` even though top-1 is equal.
4. **Provisional before confirmed.** Do not persist a permanent identity from one noisy 400ms
   sample. Accumulate clean speech first.
5. **Hysteresis.** Continuing an identity should need less evidence than switching to a different
   one.
6. **Do not update an identity from overlapped or low-quality audio.** One bad assignment
   poisons an identity permanently. Gate updates on duration, quality and single-speaker.

Rule 6 is the one place the voice problem is genuinely harder than the face problem: two faces
never blend into one image region, but two voices do sum in one waveform.

---

## 4. Out of scope for v1, listed so nobody re-derives them

- Mid-turn speaker splitting (segmentation model to cut a turn where the speaker changes).
- Overlap-aware separation of simultaneous speakers.
- PLDA/PSDA scoring instead of cosine (the original brief's "v2" — worth revisiting once the
  simple manager is proven).
- Replacing VAD.

---

## 5. Original context

This project began from the observation that diart's diarization performed badly, and that
ReDimNet2-B6 (192-dim) is a much stronger speaker embedding model. The intent was always to use
those embeddings with **our own** clustering and **our own** speaker ids, exactly as the face
package does with face embeddings — not to bolt ReDimNet2 into diart's clustering. This document
returns the project to that intent.
