#!/usr/bin/env bash

set -Eeuo pipefail

readonly DEVICE_COUNT="${DEVICE_COUNT:-16}"
readonly STREAM_COUNT="${STREAM_COUNT:-16}"
readonly LANE_COUNT="${LANE_COUNT:-3}"
readonly TASK_COUNT="${TASK_COUNT:-512}"
readonly ITERATIONS="${ITERATIONS:-128}"
readonly WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-12}"
readonly SYNC_MODE="${SYNC_MODE:-event}"
readonly START_MODES="${START_MODES:-legacy process_barrier device_gate}"
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_start_modes}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"

validate_positive_integer()
{
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'error: %s must be a positive integer: %s\n' "${name}" "${value}" >&2
        return 2
    fi
}

validate_start_mode()
{
    case "$1" in
        legacy | process_barrier | device_gate) ;;
        *)
            printf 'error: unsupported start mode: %s\n' "$1" >&2
            return 2
            ;;
    esac
}

run_case()
{
    local mode="$1"
    local label="$2"
    local case_name="$3"
    local log_file="${LOG_DIR}/${label}_${mode}_${RUN_ID}.log"
    local -a command=(
        "${UCM_TOOLKIT_BIN}" run dev-sandbox copy
        -t "${case_name}"
        --io-mode glm
        -n "${TASK_COUNT}"
        -i "${ITERATIONS}"
        --warmup "${WARMUP_ITERATIONS}"
        -d "${DEVICE_COUNT}"
        -S "${STREAM_COUNT}"
        --sync-mode "${SYNC_MODE}"
    )
    local validate=0
    if [[ "${label}" == "ffts" ]]; then
        command+=( -L "${LANE_COUNT}" )
        validate=1
    fi

    printf '\n[%s][%s] running; log=%s\n' "${mode}" "${label}" "${log_file}"
    COPY_START_MODE="${mode}" COPY_START_TRACE_ITERATIONS=0 COPY_FFTS_VALIDATE="${validate}" \
        "${command[@]}" 2>&1 | tee "${log_file}"
}

main()
{
    validate_positive_integer DEVICE_COUNT "${DEVICE_COUNT}"
    validate_positive_integer STREAM_COUNT "${STREAM_COUNT}"
    validate_positive_integer LANE_COUNT "${LANE_COUNT}"
    validate_positive_integer TASK_COUNT "${TASK_COUNT}"
    validate_positive_integer ITERATIONS "${ITERATIONS}"
    validate_positive_integer WARMUP_ITERATIONS "${WARMUP_ITERATIONS}"
    if [[ "${SYNC_MODE}" != "event" ]]; then
        printf 'error: mode comparison currently requires SYNC_MODE=event\n' >&2
        return 2
    fi
    command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1 || {
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        return 127
    }

    local -a modes
    read -ra modes <<<"${START_MODES}"
    if ((${#modes[@]} == 0)); then
        printf 'error: START_MODES is empty\n' >&2
        return 2
    fi
    local mode
    for mode in "${modes[@]}"; do validate_start_mode "${mode}"; done

    mkdir -p -- "${LOG_DIR}"
    printf 'Ascend H2D start-mode comparison\n'
    printf 'devices=%s streams=%s ffts_lanes=%s tasks_per_device=%s iterations=%s warmup=%s modes=%s\n' \
        "${DEVICE_COUNT}" "${STREAM_COUNT}" "${LANE_COUNT}" "${TASK_COUNT}" \
        "${ITERATIONS}" "${WARMUP_ITERATIONS}" "${START_MODES}"
    printf 'legacy=no process barrier, streaming submit/execute overlap\n'
    printf 'process_barrier=barrier before submission, streaming submit/execute overlap\n'
    printf 'device_gate=pre-submit all work, barrier, then broadcast device release\n'
    printf 'timing_scope legacy/process_barrier=before first stream submission to completion\n'
    printf 'timing_scope device_gate=gate release to completion; submit is reported separately\n'

    for mode in "${modes[@]}"; do
        run_case "${mode}" ce one_share_host_to_all_device_ce_multi_stream
        run_case "${mode}" ffts one_share_host_to_all_device_ffts_direct_h2d
    done

    printf '\ncomparison_complete status=PASS log_dir=%s\n' "${LOG_DIR}"
}

main "$@"
