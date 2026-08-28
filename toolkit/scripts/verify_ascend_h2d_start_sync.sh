#!/usr/bin/env bash

set -Eeuo pipefail

readonly DEVICE_COUNT="${DEVICE_COUNT:-16}"
readonly STREAM_COUNT="${STREAM_COUNT:-16}"
readonly LANE_COUNT="${LANE_COUNT:-3}"
readonly TASK_COUNT="${TASK_COUNT:-64}"
readonly ITERATIONS="${ITERATIONS:-10}"
readonly WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-12}"
readonly TRACE_ITERATIONS="${TRACE_ITERATIONS:-${ITERATIONS}}"
readonly SYNC_MODE="${SYNC_MODE:-event}"
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"
readonly MAX_BARRIER_EXIT_SKEW_US="${MAX_BARRIER_EXIT_SKEW_US:-2000}"
readonly MAX_NOTIFY_SUBMIT_SKEW_US="${MAX_NOTIFY_SUBMIT_SKEW_US:-2000}"
readonly MAX_STREAM_START_SKEW_US="${MAX_STREAM_START_SKEW_US:-500}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_start_sync}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"

readonly CE_CASE="one_share_host_to_all_device_ce_multi_stream"
readonly FFTS_CASE="one_share_host_to_all_device_ffts_direct_h2d"
readonly ANALYZER="${SCRIPT_DIR}/analyze_ascend_h2d_start_trace.py"

usage()
{
    printf 'Usage: %s\n' "${0##*/}"
    printf '\n'
    printf 'Environment overrides:\n'
    printf '  DEVICE_COUNT=<n>                  Worker/device count (default: 16)\n'
    printf '  STREAM_COUNT=<n>                  Streams per device (default: 16)\n'
    printf '  LANE_COUNT=<n>                    FFTS ready lanes (default: 3)\n'
    printf '  TASK_COUNT=<n>                    GLM tasks per device (default: 64)\n'
    printf '  ITERATIONS=<n>                    Measured iterations (default: 10)\n'
    printf '  WARMUP_ITERATIONS=<n>             Warmup iterations (default: 12)\n'
    printf '  TRACE_ITERATIONS=<n>              Measured iterations to trace\n'
    printf '  SYNC_MODE=event|stream            Completion mode (default: event)\n'
    printf '  MAX_BARRIER_EXIT_SKEW_US=<us>     Process barrier limit (default: 2000)\n'
    printf '  MAX_NOTIFY_SUBMIT_SKEW_US=<us>    Notify submission limit (default: 2000)\n'
    printf '  MAX_STREAM_START_SKEW_US=<us>     Per-device stream limit (default: 500)\n'
    printf '  LOG_DIR=<path>                    Output directory\n'
}

validate_positive_integer()
{
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'error: %s must be a positive integer: %s\n' "${name}" "${value}" >&2
        return 2
    fi
}

print_command()
{
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
}

run_case()
{
    local case_name="$1"
    local label="$2"
    local log_file="${LOG_DIR}/${label}_${RUN_ID}.log"
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
    if [[ "${label}" == "ffts" ]]; then
        command+=( -L "${LANE_COUNT}" )
    fi

    printf '\n================================================================\n'
    printf 'case=%s devices=%s streams=%s tasks=%s iterations=%s warmup=%s trace_iterations=%s\n' \
        "${case_name}" "${DEVICE_COUNT}" "${STREAM_COUNT}" "${TASK_COUNT}" \
        "${ITERATIONS}" "${WARMUP_ITERATIONS}" "${TRACE_ITERATIONS}"
    printf 'log_file=%s\n' "${log_file}"
    print_command "${command[@]}"

    if ! COPY_START_MODE=device_gate COPY_START_TRACE_ITERATIONS="${TRACE_ITERATIONS}" \
        "${command[@]}" 2>&1 | tee "${log_file}"; then
        printf 'FAIL: benchmark failed for %s\n' "${label}" >&2
        return 1
    fi

    local -a analyzer_command=(
        python3 "${ANALYZER}" "${log_file}"
        --expected-devices "${DEVICE_COUNT}"
        --expected-iterations "${TRACE_ITERATIONS}"
        --expected-start-gate event_broadcast
        --max-barrier-exit-skew-us "${MAX_BARRIER_EXIT_SKEW_US}"
        --max-notify-submit-skew-us "${MAX_NOTIFY_SUBMIT_SKEW_US}"
        --max-stream-start-skew-us "${MAX_STREAM_START_SKEW_US}"
    )
    print_command "${analyzer_command[@]}"
    "${analyzer_command[@]}" | tee -a "${log_file}"
}

main()
{
    if (($# > 0)); then
        case "$1" in
            -h|--help)
                usage
                return 0
                ;;
            *)
                printf 'error: unknown argument: %s\n' "$1" >&2
                usage >&2
                return 2
                ;;
        esac
    fi

    validate_positive_integer DEVICE_COUNT "${DEVICE_COUNT}"
    validate_positive_integer STREAM_COUNT "${STREAM_COUNT}"
    validate_positive_integer LANE_COUNT "${LANE_COUNT}"
    validate_positive_integer TASK_COUNT "${TASK_COUNT}"
    validate_positive_integer ITERATIONS "${ITERATIONS}"
    validate_positive_integer WARMUP_ITERATIONS "${WARMUP_ITERATIONS}"
    validate_positive_integer TRACE_ITERATIONS "${TRACE_ITERATIONS}"
    if ((TRACE_ITERATIONS > ITERATIONS)); then
        printf 'error: TRACE_ITERATIONS must not exceed ITERATIONS\n' >&2
        return 2
    fi
    case "${SYNC_MODE}" in
        event|stream) ;;
        *)
            printf 'error: SYNC_MODE must be event or stream: %s\n' "${SYNC_MODE}" >&2
            return 2
            ;;
    esac
    command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1 || {
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        return 127
    }
    command -v python3 >/dev/null 2>&1 || {
        printf 'error: command not found: python3\n' >&2
        return 127
    }

    mkdir -p -- "${LOG_DIR}"
    printf 'Ascend H2D process/stream start synchronization verification\n'
    printf 'run_id=%s repo_root=%s log_dir=%s\n' "${RUN_ID}" "${REPO_ROOT}" "${LOG_DIR}"
    printf 'start_gate=event_broadcast\n'
    printf 'limits_us barrier_exit=%s notify_submit=%s stream_start=%s\n' \
        "${MAX_BARRIER_EXIT_SKEW_US}" "${MAX_NOTIFY_SUBMIT_SKEW_US}" \
        "${MAX_STREAM_START_SKEW_US}"

    local failures=0
    run_case "${CE_CASE}" ce || failures=$((failures + 1))
    run_case "${FFTS_CASE}" ffts || failures=$((failures + 1))
    if ((failures > 0)); then
        printf 'verification_complete status=FAIL failures=%s log_dir=%s\n' \
            "${failures}" "${LOG_DIR}" >&2
        return 1
    fi
    printf 'verification_complete status=PASS log_dir=%s\n' "${LOG_DIR}"
}

main "$@"
