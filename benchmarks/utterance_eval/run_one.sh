#!/usr/bin/env bash
# Play one aimara-suite file through docker-compose_mp3.yaml, capture the output
# into hyp/<stem>.json, tear the stack down, then score.
#
#   ./run_one.sh wer_es__es_es_weather_wer
#
# Uses ROS_DOMAIN_ID 77 and compose project "utterance_eval" so it does not mix
# with a live microphone stack (see Docker/.env, domain 12/13).
set -euo pipefail

STEM="${1:?usage: run_one.sh <stem of recordings/aimara-suite/<stem>.wav>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DOCKER_DIR="$REPO/Docker"
DOMAIN="${ROS_DOMAIN_ID_EVAL:-77}"
PROJECT=utterance_eval
AUDIO_IN_CONTAINER="/workspace/src/audio_stream_manager/recordings/aimara-suite/$STEM.wav"

[ -f "$REPO/src/audio_stream_manager/recordings/aimara-suite/$STEM.wav" ] || { echo "no such audio: $STEM"; exit 1; }
[ -f "$HERE/gt/$STEM.json" ] || echo "warning: no gt/$STEM.json, capture only"

set -a; source "$DOCKER_DIR/.env"; set +a
export ROS_DOMAIN_ID="$DOMAIN" AUDIO_FILE="$AUDIO_IN_CONTAINER"

# Exited containers from an earlier run keep the fixed container_names busy.
for c in mongodb speech_recognition mp3_audio_source utterance_eval_capture; do
  state=$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null || true)
  if [ "$state" = "running" ] && [ "$c" != "utterance_eval_capture" ]; then
    echo "container $c is running (live stack?). Stop it first."; exit 1
  fi
  [ -n "$state" ] && docker rm -f "$c" >/dev/null
done

cleanup() { docker compose -p "$PROJECT" -f "$DOCKER_DIR/docker-compose_mp3.yaml" --env-file "$DOCKER_DIR/.env" down >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Root on purpose: as another uid FastDDS discovers topics but gets no data over shared memory.
docker run -d --rm --name utterance_eval_capture --network host --ipc host \
  -e ROS_DOMAIN_ID="$DOMAIN" -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" \
  -e FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}" \
  -v "$HERE:/eval" --entrypoint bash "$BUILT_IMAGE" \
  -c "source /workspace/install/setup.bash && python3 /eval/capture_hyp.py --out /eval/hyp/$STEM.json.tmp; chown -R $(id -u):$(id -g) /eval/hyp" >/dev/null

docker compose -p "$PROJECT" -f "$DOCKER_DIR/docker-compose_mp3.yaml" --env-file "$DOCKER_DIR/.env" up -d
echo "playing $STEM on ROS_DOMAIN_ID=$DOMAIN ..."
docker logs -f utterance_eval_capture 2>&1 | grep --line-buffered -E "audio started|\] .*:|wrote" || true
docker wait utterance_eval_capture >/dev/null 2>&1 || true

mv "$HERE/hyp/$STEM.json.tmp" "$HERE/hyp/$STEM.json"
cleanup
trap - EXIT

if [ -f "$HERE/gt/$STEM.json" ]; then
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$HERE:/eval" --entrypoint python3 "$BUILT_IMAGE" /eval/score.py "$STEM"
fi
