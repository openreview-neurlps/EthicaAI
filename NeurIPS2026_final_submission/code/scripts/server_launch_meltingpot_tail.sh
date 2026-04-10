#!/usr/bin/env bash
set -euo pipefail

# YSH-Server 전용 Melting Pot tail shard launcher.
# - 기존 JSON 결과를 읽고 완료된 pair는 자동으로 skip한다.
# - 컨테이너가 이미 running이면 아무것도 하지 않는다.
# - 컨테이너가 exited 상태면 제거 후 같은 shard를 다시 올린다.
# - @reboot cron에서 다시 호출해도 안전한 idempotent wrapper로 설계한다.

ROOT_DIR="${ROOT_DIR:-/home/ysh/neurips2026/EthicaAI/NeurIPS2026_final_submission/code}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/meltingpot}"
OUTPUT_FILE="${OUTPUT_FILE:-meltingpot_final_results_server_tail.json}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/server_tail.log}"
IMAGE="${IMAGE:-meltingpot:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-meltingpot_final}"
SEED_INDICES="${SEED_INDICES:-12-24}"
FLOORS="${FLOORS:-0.0,0.2}"

# 서버는 16코어/16GiB이므로 Sora/기타 서비스 헤드룸을 남기고 tail shard에는 12코어/12GiB만 할당한다.
CPU_LIMIT="${CPU_LIMIT:-12}"
MEM_LIMIT="${MEM_LIMIT:-12g}"
THREADS="${THREADS:-12}"

mkdir -p "${OUTPUT_DIR}"
touch "${LOG_PATH}"

docker_cmd() {
  if [[ -n "${SUDO_PASSWORD:-}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | sudo -S -p '' docker "$@"
  else
    docker "$@"
  fi
}

count_targets() {
  python3 - "${SEED_INDICES}" "${FLOORS}" <<'PY'
import sys

seed_spec = sys.argv[1]
floors = [token for token in sys.argv[2].split(",") if token.strip()]
indices = set()
for chunk in seed_spec.split(","):
    chunk = chunk.strip()
    if not chunk:
        continue
    if "-" in chunk:
        start, end = chunk.split("-", 1)
        indices.update(range(int(start), int(end) + 1))
    else:
        indices.add(int(chunk))
print(len(indices) * len(floors))
PY
}

count_completed() {
  python3 - "${OUTPUT_DIR}/${OUTPUT_FILE}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit

payload = json.loads(path.read_text())
results = payload.get("results", payload if isinstance(payload, list) else [])
print(len(results))
PY
}

container_state() {
  docker_cmd inspect "${CONTAINER_NAME}" --format '{{.State.Status}}' 2>/dev/null || true
}

TOTAL_TARGET="$(count_targets)"
DONE_COUNT="$(count_completed)"
STATE="$(container_state)"

echo "[launcher] target_pairs=${TOTAL_TARGET} completed_pairs=${DONE_COUNT} state=${STATE:-missing}" | tee -a "${LOG_PATH}"

if [[ "${DONE_COUNT}" -ge "${TOTAL_TARGET}" ]]; then
  echo "[launcher] tail shard already complete; nothing to do." | tee -a "${LOG_PATH}"
  exit 0
fi

if [[ "${STATE}" == "running" ]]; then
  echo "[launcher] container ${CONTAINER_NAME} already running; leaving as-is." | tee -a "${LOG_PATH}"
  exit 0
fi

if [[ -n "${STATE}" ]]; then
  echo "[launcher] removing stale container ${CONTAINER_NAME} (${STATE})." | tee -a "${LOG_PATH}"
  docker_cmd rm -f "${CONTAINER_NAME}" >/dev/null
fi

RUN_CMD="python3 /workspace/scripts/meltingpot_final.py --seed-indices ${SEED_INDICES} --floors ${FLOORS} --output-dir /outputs --output-file ${OUTPUT_FILE} >> /outputs/$(basename "${LOG_PATH}") 2>&1"

echo "[launcher] starting ${CONTAINER_NAME} with cpus=${CPU_LIMIT}, memory=${MEM_LIMIT}, threads=${THREADS}" | tee -a "${LOG_PATH}"

docker_cmd run -d \
  --name "${CONTAINER_NAME}" \
  --cpus "${CPU_LIMIT}" \
  --memory "${MEM_LIMIT}" \
  -e OMP_NUM_THREADS="${THREADS}" \
  -e MKL_NUM_THREADS="${THREADS}" \
  -e OPENBLAS_NUM_THREADS="${THREADS}" \
  -e NUMEXPR_NUM_THREADS="${THREADS}" \
  -e PYTHONUNBUFFERED=1 \
  -v "${ROOT_DIR}:/workspace" \
  -v "${OUTPUT_DIR}:/outputs" \
  "${IMAGE}" \
  sh -lc "${RUN_CMD}" >/dev/null

sleep 2
NEW_STATE="$(container_state)"
echo "[launcher] new_state=${NEW_STATE:-missing}" | tee -a "${LOG_PATH}"

if [[ "${NEW_STATE}" != "running" ]]; then
  echo "[launcher] container failed to stay running." | tee -a "${LOG_PATH}"
  exit 1
fi

echo "[launcher] tail shard launch succeeded." | tee -a "${LOG_PATH}"
