#!/usr/bin/env bash

set -Eeuo pipefail

SYNC_MODE="${SYNC_MODE:-event}"

readonly DEVICE_COUNT=16
readonly TASK_COUNT="${TASK_COUNT:-500}"
readonly ITERATIONS="${ITERATIONS:-128}"
readonly IO_MODE="glm"
readonly IO_SIZES="128K,16K,32K"
readonly TASK_BYTES=$(((128 + 16 + 32) * 1024))
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"

read -r -a STREAM_COUNT_VALUES <<< "${STREAM_COUNTS:-1 4 8 16 32 64 128}"
read -r -a LANE_COUNT_VALUES <<< "${LANE_COUNTS:-1 2 3}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_glm_shared_sweep}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"
readonly LOG_FILE="${LOG_FILE:-${LOG_DIR}/ascend_h2d_glm_shared_sweep_${RUN_ID}.log}"

readonly CE_CASE="one_share_host_to_all_device_ce_multi_stream"
readonly FFTS_CASE="one_share_host_to_all_device_ffts_direct_h2d"

usage()
{
    printf 'Usage: %s [--sync-mode event|stream]\n' "${0##*/}"
    printf '\n'
    printf 'Environment overrides:\n'
    printf '  STREAM_COUNTS="1 4 8 ..."  Stream values (default: 1 4 8 16 32 64 128)\n'
    printf '  LANE_COUNTS="1 2 3"        FFTS ready-lane values (default: 1 2 3)\n'
    printf '  TASK_COUNT=<n>              GLM task count per device (default: 500)\n'
    printf '  ITERATIONS=<n>              Measured iterations (default: 128)\n'
    printf '  LOG_FILE=<path>             Output log path\n'
}

parse_args()
{
    while (($# > 0)); do
        case "$1" in
            --sync-mode)
                if (($# < 2)); then
                    printf 'error: --sync-mode requires event or stream\n' >&2
                    return 2
                fi
                SYNC_MODE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                printf 'error: unknown argument: %s\n' "$1" >&2
                usage >&2
                return 2
                ;;
        esac
    done

    case "${SYNC_MODE}" in
        event|stream) ;;
        *)
            printf 'error: invalid sync mode: %s; expected event or stream\n' \
                "${SYNC_MODE}" >&2
            return 2
            ;;
    esac
}

validate_positive_values()
{
    local label="$1"
    shift
    local value
    for value in "$@"; do
        if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
            printf 'error: %s must contain positive integers: %s\n' "${label}" "${value}" >&2
            return 2
        fi
    done
}

print_command()
{
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
}

run_one()
{
    local case_name="$1"
    local method="$2"
    local stream_count="$3"
    local lane_count="$4"
    local effective_streams="${stream_count}"
    if ((effective_streams > TASK_COUNT)); then
        effective_streams="${TASK_COUNT}"
    fi

    local effective_lanes="none"
    if [[ "${method}" == "ffts" ]]; then
        effective_lanes="${lane_count}"
        if ((effective_lanes > 3)); then
            effective_lanes=3
        fi
    fi

    local -a command=(
        "${UCM_TOOLKIT_BIN}" run dev-sandbox copy
        -t "${case_name}"
        --io-mode "${IO_MODE}"
        -n "${TASK_COUNT}"
        -i "${ITERATIONS}"
        -d "${DEVICE_COUNT}"
        -S "${stream_count}"
        --sync-mode "${SYNC_MODE}"
    )
    if [[ "${method}" == "ffts" ]]; then
        command+=( -L "${lane_count}" )
    fi

    local start_seconds
    start_seconds="$(date +%s)"
    printf '\n================================================================\n'
    printf 'case=%s method=%s devices=%s io_mode=%s io_sizes=%s tasks_per_device=%s ' \
        "${case_name}" "${method}" "${DEVICE_COUNT}" "${IO_MODE}" "${IO_SIZES}" \
        "${TASK_COUNT}"
    printf 'requested_streams=%s effective_streams=%s requested_lanes=%s effective_lanes=%s ' \
        "${stream_count}" "${effective_streams}" "${lane_count}" "${effective_lanes}"
    printf 'sync_mode=%s start=%s\n' "${SYNC_MODE}" "$(date --iso-8601=seconds)"
    printf 'task_bytes=%s bytes_per_device=%s aggregate_bytes=%s\n' \
        "${TASK_BYTES}" "$((TASK_BYTES * TASK_COUNT))" \
        "$((TASK_BYTES * TASK_COUNT * DEVICE_COUNT))"
    print_command "${command[@]}"

    local status=0
    "${command[@]}" || status=$?

    local end_seconds
    end_seconds="$(date +%s)"
    printf 'case=%s method=%s streams=%s lanes=%s status=%s elapsed_s=%s end=%s\n' \
        "${case_name}" "${method}" "${stream_count}" "${lane_count}" "${status}" \
        "$((end_seconds - start_seconds))" "$(date --iso-8601=seconds)"
    return "${status}"
}

main()
{
    parse_args "$@" || return $?
    readonly SYNC_MODE
    validate_positive_values STREAM_COUNTS "${STREAM_COUNT_VALUES[@]}" || return $?
    validate_positive_values LANE_COUNTS "${LANE_COUNT_VALUES[@]}" || return $?
    validate_positive_values TASK_COUNT "${TASK_COUNT}" || return $?
    validate_positive_values ITERATIONS "${ITERATIONS}" || return $?

    mkdir -p -- "$(dirname -- "${LOG_FILE}")"
    exec > >(tee -a "${LOG_FILE}") 2>&1

    if ! command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1; then
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        printf 'install the toolkit first: python -m pip install -e toolkit\n' >&2
        return 127
    fi

    printf 'Ascend H2D GLM shared-memory stream/lane sweep\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'log_file=%s\n' "${LOG_FILE}"
    printf 'devices=%s io_mode=%s io_sizes=%s task_bytes=%s tasks_per_device=%s iterations=%s\n' \
        "${DEVICE_COUNT}" "${IO_MODE}" "${IO_SIZES}" "${TASK_BYTES}" "${TASK_COUNT}" \
        "${ITERATIONS}"
    printf 'streams=%s\n' "${STREAM_COUNT_VALUES[*]}"
    printf 'ffts_lanes=%s\n' "${LANE_COUNT_VALUES[*]}"
    printf 'sync_mode=%s\n' "${SYNC_MODE}"
    printf 'git_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
    printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"

    local failures=0
    local stream_count
    local lane_count
    for stream_count in "${STREAM_COUNT_VALUES[@]}"; do
        if ! run_one "${CE_CASE}" ce "${stream_count}" none; then
            failures=$((failures + 1))
        fi
    done
    for stream_count in "${STREAM_COUNT_VALUES[@]}"; do
        for lane_count in "${LANE_COUNT_VALUES[@]}"; do
            if ! run_one "${FFTS_CASE}" ffts "${stream_count}" "${lane_count}"; then
                failures=$((failures + 1))
            fi
        done
    done

    local total_runs=$((
        ${#STREAM_COUNT_VALUES[@]} +
        ${#STREAM_COUNT_VALUES[@]} * ${#LANE_COUNT_VALUES[@]}
    ))
    printf '\n================================================================\n'
    printf 'sweep_complete failures=%s total_runs=%s log_file=%s\n' \
        "${failures}" "${total_runs}" "${LOG_FILE}"
    if ((failures > 0)); then
        return 1
    fi
}

main "$@"
