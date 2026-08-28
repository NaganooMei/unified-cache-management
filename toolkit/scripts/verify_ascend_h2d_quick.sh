#!/usr/bin/env bash

set -Eeuo pipefail

readonly DEVICE_COUNT="${DEVICE_COUNT:-16}"
readonly STREAM_COUNT="${STREAM_COUNT:-16}"
readonly LANE_COUNT="${LANE_COUNT:-3}"
readonly TASK_COUNT="${TASK_COUNT:-64}"
readonly ITERATIONS="${ITERATIONS:-10}"
readonly WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-12}"
readonly TRACE_ITERATIONS="${TRACE_ITERATIONS:-3}"
readonly SYNC_MODE="${SYNC_MODE:-event}"
readonly UCM_TOOLKIT_BIN="${UCM_TOOLKIT_BIN:-ucm-toolkit}"
readonly MAX_BARRIER_EXIT_SKEW_US="${MAX_BARRIER_EXIT_SKEW_US:-2000}"
readonly MAX_NOTIFY_SUBMIT_SKEW_US="${MAX_NOTIFY_SUBMIT_SKEW_US:-2000}"
readonly MAX_STREAM_START_SKEW_US="${MAX_STREAM_START_SKEW_US:-500}"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
readonly LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/toolkit_h2d_quick}"
readonly RUN_ID="$(date '+%Y%m%d_%H%M%S')"
readonly ANALYZER="${SCRIPT_DIR}/analyze_ascend_h2d_start_trace.py"

validate_positive_integer()
{
    local name="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'error: %s must be a positive integer: %s\n' "${name}" "${value}" >&2
        return 2
    fi
}

summary_value()
{
    local key="$1"
    shift
    local token
    for token in "$@"; do
        if [[ "${token}" == "${key}="* ]]; then
            printf '%s' "${token#*=}"
            return 0
        fi
    done
    return 1
}

slash_value()
{
    local values="$1"
    local index="$2"
    local -a fields
    IFS='/' read -ra fields <<<"${values}"
    if ((index >= ${#fields[@]})); then
        return 1
    fi
    printf '%s' "${fields[index]}"
}

print_performance()
{
    local log_file="$1"
    local observed_sync_skew_us="$2"
    local row
    row="$(grep -E 'acl::shm::glm[[:space:]]+acl::device::all' "${log_file}" | tail -n 1)"
    if [[ -z "${row}" ]]; then
        printf 'error: benchmark result row is missing in %s\n' "${log_file}" >&2
        return 1
    fi

    row="$(sed -E 's/[[:space:]]*\/[[:space:]]*/\//g; s/^[[:space:]]+//' <<<"${row}")"
    local src dst method size_kb count submit_us copy_us wall_us dev_bw wall_bw extra
    read -r src dst method size_kb count submit_us copy_us wall_us dev_bw wall_bw extra <<<"${row}"
    if [[ -n "${extra:-}" || -z "${wall_bw:-}" ]]; then
        printf 'error: cannot parse benchmark result row: %s\n' "${row}" >&2
        return 1
    fi

    printf 'performance method=%s size_kb=%s count=%s\n' "${method}" "${size_kb}" "${count}"
    printf '  submit_us_min_max_avg_p50_p90=%s\n' "${submit_us}"
    printf '  copy_us_min_max_avg_p50_p90=%s dev_bw_gib_s=%s\n' "${copy_us}" "${dev_bw}"
    printf '  wall_us_min_max_avg_p50_p90=%s wall_bw_gib_s=%s\n' "${wall_us}" "${wall_bw}"

    local dev_avg_us dev_p50_us wall_avg_us wall_p50_us
    dev_avg_us="$(slash_value "${copy_us}" 2)"
    dev_p50_us="$(slash_value "${copy_us}" 3)"
    wall_avg_us="$(slash_value "${wall_us}" 2)"
    wall_p50_us="$(slash_value "${wall_us}" 3)"
    local avg_gap_us=$((wall_avg_us - dev_avg_us))
    local p50_gap_us=$((wall_p50_us - dev_p50_us))
    local avg_wall_over_dev avg_gap_percent gap_to_sync_ratio
    avg_wall_over_dev="$(awk -v wall="${wall_avg_us}" -v dev="${dev_avg_us}" \
        'BEGIN { if (dev == 0) print "inf"; else printf "%.3f", wall / dev }')"
    avg_gap_percent="$(awk -v gap="${avg_gap_us}" -v wall="${wall_avg_us}" \
        'BEGIN { if (wall == 0) print "0.000"; else printf "%.3f", gap * 100 / wall }')"
    gap_to_sync_ratio="$(awk -v gap="${avg_gap_us}" -v skew="${observed_sync_skew_us}" \
        'BEGIN { if (skew == 0) print "inf"; else printf "%.3f", gap / skew }')"
    printf 'timing_compare avg_dev_us=%s avg_wall_us=%s wall_minus_dev_us=%s wall_over_dev=%s gap_pct_of_wall=%s\n' \
        "${dev_avg_us}" "${wall_avg_us}" "${avg_gap_us}" "${avg_wall_over_dev}" \
        "${avg_gap_percent}"
    printf 'timing_compare p50_dev_us=%s p50_wall_us=%s wall_minus_dev_us=%s\n' \
        "${dev_p50_us}" "${wall_p50_us}" "${p50_gap_us}"
    printf 'timing_gap observed_max_sync_skew_us=%s avg_gap_to_sync_skew=%s\n' \
        "${observed_sync_skew_us}" "${gap_to_sync_ratio}"
    printf 'timing_scope wall=before_release_submit_to_after_host_sync dev=totalStart_to_totalEnd\n'
}

run_case()
{
    local label="$1"
    local case_name="$2"
    local log_file="${LOG_DIR}/${label}_${RUN_ID}.log"
    local analysis_file="${LOG_DIR}/${label}_${RUN_ID}.analysis"
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

    printf '\n[%s] running...\n' "${label}"
    if ! COPY_START_TRACE_ITERATIONS="${TRACE_ITERATIONS}" COPY_FFTS_VALIDATE=1 \
        "${command[@]}" >"${log_file}" 2>&1; then
        printf '[%s] FAIL: benchmark failed; tail of %s follows\n' "${label}" "${log_file}" >&2
        tail -n 80 "${log_file}" >&2
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
        --require-timing-breakdown
    )
    if ! "${analyzer_command[@]}" >"${analysis_file}" 2>&1; then
        printf '[%s] FAIL: synchronization analysis failed\n' "${label}" >&2
        cat "${analysis_file}" >&2
        return 1
    fi

    local summary_line
    summary_line="$(grep '^summary ' "${analysis_file}" | tail -n 1)"
    if [[ -z "${summary_line}" ]]; then
        printf '[%s] FAIL: synchronization summary is missing\n' "${label}" >&2
        return 1
    fi
    local -a summary_fields
    read -ra summary_fields <<<"${summary_line}"
    local timing_line
    timing_line="$(grep '^timing_summary ' "${analysis_file}" | tail -n 1)"
    if [[ -z "${timing_line}" ]]; then
        printf '[%s] FAIL: timing breakdown is missing\n' "${label}" >&2
        return 1
    fi
    local -a timing_fields
    read -ra timing_fields <<<"${timing_line}"
    local barrier_skew_us notify_skew_us stream_skew_us observed_sync_skew_us
    barrier_skew_us="$(summary_value max_barrier_exit_skew_us "${summary_fields[@]}")"
    notify_skew_us="$(summary_value max_notify_submit_skew_us "${summary_fields[@]}")"
    stream_skew_us="$(summary_value max_stream_start_skew_us "${summary_fields[@]}")"
    observed_sync_skew_us="$(awk -v barrier="${barrier_skew_us}" -v notify="${notify_skew_us}" \
        -v stream="${stream_skew_us}" \
        'BEGIN { max = barrier; if (notify > max) max = notify; if (stream > max) max = stream; printf "%.3f", max }')"

    if [[ "${label}" == "ffts" ]]; then
        if ! grep -Fq '[validation] one_share_host_to_all_device_ffts_direct_h2d PASS' \
            "${log_file}"; then
            printf '[%s] FAIL: payload validation PASS marker is missing\n' "${label}" >&2
            return 1
        fi
        printf 'payload_validation=PASS\n'
    else
        printf 'payload_validation=not_applicable\n'
    fi

    print_performance "${log_file}" "${observed_sync_skew_us}"
    printf 'process_sync max_barrier_exit_skew_us=%s max_notify_submit_skew_us=%s\n' \
        "${barrier_skew_us}" "${notify_skew_us}"
    printf 'stream_sync p50_skew_us=%s p95_skew_us=%s max_skew_us=%s\n' \
        "$(summary_value p50_stream_start_skew_us "${summary_fields[@]}")" \
        "$(summary_value p95_stream_start_skew_us "${summary_fields[@]}")" \
        "${stream_skew_us}"
    printf 'host_breakdown release_submit_avg_p95_us=%s/%s control_submit_avg_p95_us=%s/%s sync_wait_avg_p95_us=%s/%s\n' \
        "$(summary_value host_release_submit_avg_us "${timing_fields[@]}")" \
        "$(summary_value host_release_submit_p95_us "${timing_fields[@]}")" \
        "$(summary_value host_control_submit_avg_us "${timing_fields[@]}")" \
        "$(summary_value host_control_submit_p95_us "${timing_fields[@]}")" \
        "$(summary_value host_sync_wait_avg_us "${timing_fields[@]}")" \
        "$(summary_value host_sync_wait_p95_us "${timing_fields[@]}")"
    printf 'device_breakdown gate_avg_p95_us=%s/%s copy_avg_p95_us=%s/%s\n' \
        "$(summary_value device_gate_avg_us "${timing_fields[@]}")" \
        "$(summary_value device_gate_p95_us "${timing_fields[@]}")" \
        "$(summary_value device_copy_avg_us "${timing_fields[@]}")" \
        "$(summary_value device_copy_p95_us "${timing_fields[@]}")"
    printf 'trace_wall process_avg_p95_us=%s/%s outside_device_avg_p95_us=%s/%s\n' \
        "$(summary_value process_wall_avg_us "${timing_fields[@]}")" \
        "$(summary_value process_wall_p95_us "${timing_fields[@]}")" \
        "$(summary_value wall_minus_device_avg_us "${timing_fields[@]}")" \
        "$(summary_value wall_minus_device_p95_us "${timing_fields[@]}")"
    printf 'status=PASS raw_log=%s analysis=%s\n' "${log_file}" "${analysis_file}"
}

main()
{
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
    if [[ "${SYNC_MODE}" != "event" ]]; then
        printf 'error: this synchronized-start verification requires SYNC_MODE=event\n' >&2
        return 2
    fi
    command -v "${UCM_TOOLKIT_BIN}" >/dev/null 2>&1 || {
        printf 'error: command not found: %s\n' "${UCM_TOOLKIT_BIN}" >&2
        return 127
    }
    command -v python3 >/dev/null 2>&1 || {
        printf 'error: command not found: python3\n' >&2
        return 127
    }

    mkdir -p -- "${LOG_DIR}"
    printf 'Ascend H2D quick verification\n'
    printf 'devices=%s streams=%s ffts_lanes=%s tasks_per_device=%s iterations=%s warmup=%s trace=%s\n' \
        "${DEVICE_COUNT}" "${STREAM_COUNT}" "${LANE_COUNT}" "${TASK_COUNT}" \
        "${ITERATIONS}" "${WARMUP_ITERATIONS}" "${TRACE_ITERATIONS}"

    local failures=0
    run_case ce one_share_host_to_all_device_ce_multi_stream || failures=$((failures + 1))
    run_case ffts one_share_host_to_all_device_ffts_direct_h2d || failures=$((failures + 1))
    if ((failures > 0)); then
        printf '\nverification_complete status=FAIL failures=%s log_dir=%s\n' \
            "${failures}" "${LOG_DIR}" >&2
        return 1
    fi
    printf '\nverification_complete status=PASS log_dir=%s\n' "${LOG_DIR}"
}

main "$@"
