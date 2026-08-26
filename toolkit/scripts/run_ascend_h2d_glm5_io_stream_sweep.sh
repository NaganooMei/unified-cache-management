#!/usr/bin/env bash

set -Eeuo pipefail

SYNC_MODE="${SYNC_MODE:-event}"

readonly BLOCK_SIZE="32K"
readonly BLOCK_BYTES=$((32 * 1024))
readonly ITERATIONS=128
readonly TOKENS_PER_SHARD=128
readonly IO_PER_SHARD=3
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"
readonly DEVICE_COUNTS=(1 8)
readonly STREAM_COUNTS=(1 4 8 16 32 64 128)
readonly SEQUENCE_SPECS=(
    "8K:8192"
    "64K:65536"
    "128K:131072"
)

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_glm5_stream_sweep}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"
readonly LOG_FILE="${LOG_FILE:-${LOG_DIR}/ascend_h2d_glm5_stream_sweep_${RUN_ID}.log}"

readonly CASES=(
    "all_odirect_host_to_all_device_ce_multi_stream:ce"
    "all_odirect_host_to_all_device_ffts_direct_h2d:ffts"
    "one_share_host_to_all_device_ce_multi_stream:ce"
    "one_share_host_to_all_device_ffts_direct_h2d:ffts"
)

usage()
{
    printf 'Usage: %s [--sync-mode event|stream]\n' "${0##*/}"
    printf '\n'
    printf 'Options:\n'
    printf '  --sync-mode <mode>  Multi-stream completion mode (default: event)\n'
    printf '  -h, --help          Show this help\n'
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

print_command()
{
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
}

print_sequence_map()
{
    local spec
    local sequence_label
    local sequence_tokens
    local shard_count
    local io_count
    local bytes_per_device
    for spec in "${SEQUENCE_SPECS[@]}"; do
        sequence_label="${spec%%:*}"
        sequence_tokens="${spec##*:}"
        shard_count=$((sequence_tokens / TOKENS_PER_SHARD))
        io_count=$((shard_count * IO_PER_SHARD))
        bytes_per_device=$((io_count * BLOCK_BYTES))
        printf 'sequence=%s tokens=%s shards=%s io_per_shard=%s io_count=%s bytes_per_device=%s mib_per_device=%s\n' \
            "${sequence_label}" "${sequence_tokens}" "${shard_count}" \
            "${IO_PER_SHARD}" "${io_count}" "${bytes_per_device}" \
            "$((bytes_per_device / 1024 / 1024))"
    done
}

run_one()
{
    local case_name="$1"
    local method="$2"
    local device_count="$3"
    local stream_count="$4"
    local sequence_label="$5"
    local sequence_tokens="$6"
    local shard_count=$((sequence_tokens / TOKENS_PER_SHARD))
    local io_count=$((shard_count * IO_PER_SHARD))
    local io_bytes_per_device=$((io_count * BLOCK_BYTES))
    local aggregate_io_bytes=$((io_bytes_per_device * device_count))
    local number
    local scheduling_units

    if [[ "${method}" == "ffts" ]]; then
        # One FFTS task represents one shard and contains three 32K IO fragments.
        number="${shard_count}"
        scheduling_units="${shard_count}"
    else
        # CE submits the same number of physical 32K IOs as FFTS.
        number="${io_count}"
        scheduling_units="${io_count}"
    fi

    local effective_stream_count="${stream_count}"
    if ((effective_stream_count > scheduling_units)); then
        effective_stream_count="${scheduling_units}"
    fi

    local -a command=(
        "${UCM_TOOLKIT_BIN}" run dev-sandbox copy
        -t "${case_name}"
        -s "${BLOCK_SIZE}"
        -n "${number}"
        -i "${ITERATIONS}"
        -d "${device_count}"
        -S "${stream_count}"
        --sync-mode "${SYNC_MODE}"
    )
    if [[ "${method}" == "ffts" ]]; then
        command+=( -f "${IO_PER_SHARD}" )
    fi

    local start_seconds
    start_seconds="$(date +%s)"
    printf '\n================================================================\n'
    printf 'case=%s method=%s sequence=%s tokens=%s shards=%s io_count_per_device=%s '\
        "${case_name}" "${method}" "${sequence_label}" "${sequence_tokens}" \
        "${shard_count}" "${io_count}"
    printf 'devices=%s requested_streams=%s effective_streams=%s sync_mode=%s start=%s\n' \
        "${device_count}" "${stream_count}" "${effective_stream_count}" "${SYNC_MODE}" \
        "$(date --iso-8601=seconds)"
    printf 'io_bytes_per_device=%s aggregate_io_bytes=%s toolkit_n=%s ffts_frags=%s\n' \
        "${io_bytes_per_device}" "${aggregate_io_bytes}" "${number}" \
        "$([[ "${method}" == "ffts" ]] && printf '%s' "${IO_PER_SHARD}" || printf 'none')"
    print_command "${command[@]}"

    local status=0
    "${command[@]}" || status=$?

    local end_seconds
    end_seconds="$(date +%s)"
    printf 'case=%s method=%s sequence=%s devices=%s requested_streams=%s effective_streams=%s '\
        "${case_name}" "${method}" "${sequence_label}" "${device_count}" \
        "${stream_count}" "${effective_stream_count}"
    printf 'sync_mode=%s status=%s elapsed_s=%s end=%s\n' \
        "${SYNC_MODE}" "${status}" "$((end_seconds - start_seconds))" \
        "$(date --iso-8601=seconds)"
    return "${status}"
}

main()
{
    parse_args "$@" || return $?
    readonly SYNC_MODE

    mkdir -p -- "$(dirname -- "${LOG_FILE}")"
    exec > >(tee -a "${LOG_FILE}") 2>&1

    if ! command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1; then
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        printf 'install the toolkit first: python -m pip install -e toolkit\n' >&2
        return 127
    fi

    printf 'Ascend H2D GLM5 IO and stream sweep\n'
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'log_file=%s\n' "${LOG_FILE}"
    printf 'block_size=%s iterations=%s tokens_per_shard=%s io_per_shard=%s\n' \
        "${BLOCK_SIZE}" "${ITERATIONS}" "${TOKENS_PER_SHARD}" "${IO_PER_SHARD}"
    printf 'devices=%s\n' "${DEVICE_COUNTS[*]}"
    printf 'streams=%s\n' "${STREAM_COUNTS[*]}"
    printf 'sync_mode=%s\n' "${SYNC_MODE}"
    print_sequence_map
    printf 'git_branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || true)"
    printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"

    local failures=0
    local entry
    local case_name
    local method
    local device_count
    local spec
    local sequence_label
    local sequence_tokens
    local stream_count
    for entry in "${CASES[@]}"; do
        case_name="${entry%%:*}"
        method="${entry##*:}"
        for device_count in "${DEVICE_COUNTS[@]}"; do
            for spec in "${SEQUENCE_SPECS[@]}"; do
                sequence_label="${spec%%:*}"
                sequence_tokens="${spec##*:}"
                for stream_count in "${STREAM_COUNTS[@]}"; do
                    if ! run_one "${case_name}" "${method}" "${device_count}" \
                        "${stream_count}" "${sequence_label}" "${sequence_tokens}"; then
                        failures=$((failures + 1))
                    fi
                done
            done
        done
    done

    printf '\n================================================================\n'
    printf 'sweep_complete failures=%s total_runs=%s log_file=%s\n' \
        "${failures}" \
        "$(( ${#CASES[@]} * ${#DEVICE_COUNTS[@]} * ${#SEQUENCE_SPECS[@]} * ${#STREAM_COUNTS[@]} ))" \
        "${LOG_FILE}"
    if ((failures > 0)); then
        return 1
    fi
}

main "$@"
