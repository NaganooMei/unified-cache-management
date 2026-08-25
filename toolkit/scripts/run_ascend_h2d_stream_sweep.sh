#!/usr/bin/env bash

set -Eeuo pipefail

readonly BLOCK_SIZE="32K"
readonly BLOCK_COUNT=500
readonly ITERATIONS=128
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"
readonly DEVICE_COUNTS=(1 8)
readonly STREAM_COUNTS=(1 4 8 16 32 64 128)

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_stream_sweep}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"
readonly LOG_FILE="${LOG_FILE:-${LOG_DIR}/ascend_h2d_stream_sweep_${RUN_ID}.log}"

readonly CASES=(
    "all_odirect_host_to_all_device_ce_multi_stream:ce"
    "all_odirect_host_to_all_device_ffts_direct_h2d:ffts"
    "one_share_host_to_all_device_ce_multi_stream:ce"
    "one_share_host_to_all_device_ffts_direct_h2d:ffts"
)

mkdir -p -- "$(dirname -- "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

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
    local device_count="$3"
    local stream_count="$4"
    local -a command=(
        "${UCM_TOOLKIT_BIN}" run dev-sandbox copy
        -t "${case_name}"
        -s "${BLOCK_SIZE}"
        -n "${BLOCK_COUNT}"
        -i "${ITERATIONS}"
        -d "${device_count}"
        -S "${stream_count}"
    )

    # FFTS multi-stream schedules IO/tasks across streams. With -f 3,
    # -n 500 means 500 independent tasks containing three 32K fragments each.
    if [[ "${method}" == "ffts" ]]; then
        command+=( -f 3 )
    fi

    local start_seconds
    start_seconds="$(date +%s)"
    printf '\n================================================================\n'
    printf 'case=%s method=%s devices=%s streams=%s start=%s\n' \
        "${case_name}" "${method}" "${device_count}" "${stream_count}" \
        "$(date --iso-8601=seconds)"
    print_command "${command[@]}"

    local status=0
    "${command[@]}" || status=$?

    local end_seconds
    end_seconds="$(date +%s)"
    printf 'case=%s method=%s devices=%s streams=%s status=%s elapsed_s=%s end=%s\n' \
        "${case_name}" "${method}" "${device_count}" "${stream_count}" \
        "${status}" "$((end_seconds - start_seconds))" "$(date --iso-8601=seconds)"
    return "${status}"
}

main()
{
    if ! command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1; then
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        printf 'install the toolkit first: python -m pip install -e toolkit\n' >&2
        return 127
    fi

    printf 'Ascend H2D stream sweep\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'log_file=%s\n' "${LOG_FILE}"
    printf 'block_size=%s block_count=%s iterations=%s ffts_frags=3\n' \
        "${BLOCK_SIZE}" "${BLOCK_COUNT}" "${ITERATIONS}"
    printf 'devices=%s\n' "${DEVICE_COUNTS[*]}"
    printf 'streams=%s\n' "${STREAM_COUNTS[*]}"
    printf 'git_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
    printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"

    local failures=0
    local entry
    local case_name
    local method
    local device_count
    local stream_count
    for entry in "${CASES[@]}"; do
        case_name="${entry%%:*}"
        method="${entry##*:}"
        for device_count in "${DEVICE_COUNTS[@]}"; do
            for stream_count in "${STREAM_COUNTS[@]}"; do
                if ! run_one "${case_name}" "${method}" "${device_count}" "${stream_count}"; then
                    failures=$((failures + 1))
                fi
            done
        done
    done

    printf '\n================================================================\n'
    printf 'sweep_complete failures=%s total_runs=%s log_file=%s\n' \
        "${failures}" \
        "$(( ${#CASES[@]} * ${#DEVICE_COUNTS[@]} * ${#STREAM_COUNTS[@]} ))" \
        "${LOG_FILE}"
    if ((failures > 0)); then
        return 1
    fi
}

main "$@"
