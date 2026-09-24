#!/usr/bin/env bash
# Test: play the 3 reviewed files through docker-compose_mp3.yaml and write
# TEST_compose_mp3.md (WER all, WER long, per-utterance speaker check).
#
#   ./run_test_mp3.sh            # replay all 3, then score
#   ./run_test_mp3.sh --score    # score the existing hyp/ captures only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STEMS=(
  wer_es__es_es_weather_wer
  callhome_spa_snr20__spa_0019_4spk_snr20
  callhome_spa_snr20__spa_0018_2spk_snr20
)
IMAGE="$(grep -E '^BUILT_IMAGE=' "$HERE/../../Docker/.env" | tail -1 | cut -d= -f2)"

if [ "${1:-}" != "--score" ]; then
  for stem in "${STEMS[@]}"; do
    "$HERE/run_one.sh" "$stem" 2>&1 | grep -E "playing|wrote|WER" || true
    [ -f "$HERE/hyp/$stem.json" ] || { echo "capture failed for $stem"; exit 1; }
  done
fi

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HERE:/eval" --entrypoint python3 "$IMAGE" \
  /eval/score.py --test-md /eval/TEST_compose_mp3.md \
  --title "Test: docker-compose_mp3 pipeline ($(date +%F))" "${STEMS[@]}"
