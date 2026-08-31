#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ucm-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


plt.style.use("seaborn-v0_8-whitegrid")
WINDOWS_CJK_FONT = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
if WINDOWS_CJK_FONT.exists():
    font_manager.fontManager.addfont(str(WINDOWS_CJK_FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(WINDOWS_CJK_FONT)
    ).get_name()
else:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
plt.rcParams["axes.unicode_minus"] = False


DOCS_DIR = Path(__file__).resolve().parent
REPORT_PATH = (
    DOCS_DIR
    / "source"
    / "developer-guide"
    / "ascend_h2d_glm512_ce_ffts_matrix_report.md"
)
DATA_PATH = (
    DOCS_DIR
    / "source"
    / "developer-guide"
    / "data"
    / "ascend_h2d_glm512_144_matrix.tsv"
)
FULL_DATA_PATH = (
    DOCS_DIR
    / "source"
    / "developer-guide"
    / "data"
    / "ascend_h2d_glm512_144_matrix_full.tsv"
)
IMAGE_DIR = DOCS_DIR / "source" / "_static" / "images"

METHODS = ("ce", "ffts")
HOST_MODES = ("one_share", "all")
DEVICES = (1, 8, 16)
STREAMS = (1, 4, 16)
SUBMIT_MODES = ("stream-major", "round-robin")
SYNC_MODES = ("event", "stream")

SERIES = (
    ("stream-major", "stream", "STREAM / stream-major", "#2f6bff", "-", "o"),
    ("round-robin", "stream", "STREAM / round-robin", "#2f6bff", "--", "s"),
    ("stream-major", "event", "EVENT / stream-major", "#e45756", "-", "o"),
    ("round-robin", "event", "EVENT / round-robin", "#e45756", "--", "s"),
)

STAT_NAMES = ("min", "max", "avg", "p50", "p90")
STATS_PATTERN = r"\d+\s*/\s*\d+\s*/\s*\d+\s*/\s*\d+\s*/\s*\d+"
SECTION_PATTERN = re.compile(
    r"^\[(?P<index>\d+)/144\]\s+"
    r"method=(?P<method>\S+)\s+host=(?P<host_mode>\S+)\s+"
    r"devices=(?P<devices>\d+)\s+streams=(?P<streams>\d+)\s+"
    r"submit=(?P<submit_mode>\S+)\s+stream_sync=(?P<stream_sync>\S+)",
    re.MULTILINE,
)
RESULT_PATTERN = re.compile(
    rf"^\s+acl::.*?(?P<submit>{STATS_PATTERN})\s+"
    rf"(?P<copy>N/A|{STATS_PATTERN})\s+"
    r"(?P<dev_bw>N/A|\d+(?:\.\d+)?)\s*$",
    re.MULTILINE,
)
PROCESS_PATTERN = re.compile(
    rf"ProcessSync=\S+\s+"
    rf"StartSkew\(us\)-\(Min/Max/Avg/P50/P90\)=(?P<start_skew>{STATS_PATTERN})\s+"
    rf"GroupWall\(us\)-\(Min/Max/Avg/P50/P90\)=(?P<group_wall>{STATS_PATTERN})\s+"
    r"WallBW\(GB/s\)=(?P<wall_bw>\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the GLM5.1 CE/FFTS matrix report and figures."
    )
    parser.add_argument(
        "summary",
        nargs="?",
        type=Path,
        help="Optional 144-row summary TSV to copy into the report data directory.",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        help="Raw all.log fragments used to build the full timing TSV.",
    )
    return parser.parse_args()


def load_rows(summary: Path | None) -> list[dict[str, object]]:
    if summary is not None:
        source = summary.resolve()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        if source != DATA_PATH.resolve():
            shutil.copyfile(source, DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"data file does not exist: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    numeric_ints = (
        "devices",
        "streams",
        "blocks",
        "iterations",
        "exit_code",
        "start_skew_avg_us",
        "group_wall_avg_us",
    )
    for row in rows:
        for field in numeric_ints:
            row[field] = int(row[field])
        row["wall_bw_gbps"] = float(row["wall_bw_gbps"])
        row["dev_bw_gbps"] = (
            None
            if str(row["dev_bw_gbps"]).upper() in {"N/A", "NA"}
            else float(row["dev_bw_gbps"])
        )
    return rows


def matrix_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["method"],
        row["host_mode"],
        int(row["devices"]),
        int(row["streams"]),
        row["submit_mode"],
        row["stream_sync"],
    )


def parse_stats(text: str) -> tuple[int, int, int, int, int]:
    values = tuple(int(value.strip()) for value in text.split("/"))
    if len(values) != 5:
        raise ValueError(f"invalid timing statistics: {text}")
    return values


def timing_record(
    section: re.Match[str], result: re.Match[str], process: re.Match[str]
) -> dict[str, object]:
    record: dict[str, object] = {
        "matrix_index": int(section.group("index")),
        "method": section.group("method"),
        "host_mode": section.group("host_mode"),
        "devices": int(section.group("devices")),
        "streams": int(section.group("streams")),
        "submit_mode": section.group("submit_mode"),
        "stream_sync": section.group("stream_sync"),
        "log_dev_bw_gbps": (
            None if result.group("dev_bw") == "N/A" else float(result.group("dev_bw"))
        ),
        "log_wall_bw_gbps": float(process.group("wall_bw")),
    }
    for prefix, value in (
        ("submit", result.group("submit")),
        ("start_skew", process.group("start_skew")),
        ("group_wall", process.group("group_wall")),
    ):
        for name, stat in zip(STAT_NAMES, parse_stats(value)):
            record[f"{prefix}_{name}_us"] = stat

    if result.group("copy") == "N/A":
        for name in STAT_NAMES:
            record[f"copy_{name}_us"] = None
    else:
        for name, stat in zip(STAT_NAMES, parse_stats(result.group("copy"))):
            record[f"copy_{name}_us"] = stat
    return record


def parse_log_fragments(log_paths: list[Path]) -> list[dict[str, object]]:
    by_index: dict[int, dict[str, object]] = {}
    for log_path in log_paths:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        sections = list(SECTION_PATTERN.finditer(text))
        for position, section in enumerate(sections):
            end = sections[position + 1].start() if position + 1 < len(sections) else len(text)
            block = text[section.end() : end]
            result = RESULT_PATTERN.search(block)
            process = PROCESS_PATTERN.search(block)
            if result is None or process is None:
                continue
            record = timing_record(section, result, process)
            index = int(record["matrix_index"])
            previous = by_index.get(index)
            if previous is not None and previous != record:
                raise ValueError(f"conflicting complete log sections for matrix index {index}")
            by_index[index] = record

    expected = set(range(1, 145))
    missing = sorted(expected - set(by_index))
    extra = sorted(set(by_index) - expected)
    if missing or extra:
        raise ValueError(f"log fragments are incomplete: missing={missing}, extra={extra}")
    return [by_index[index] for index in range(1, 145)]


def write_full_data(
    rows: list[dict[str, object]], timing_rows: list[dict[str, object]]
) -> None:
    summary_by_key = {matrix_key(row): row for row in rows}
    output_rows = []
    timing_fields = [
        "matrix_index",
        *(f"submit_{name}_us" for name in STAT_NAMES),
        *(f"copy_{name}_us" for name in STAT_NAMES),
        *(f"start_skew_{name}_us" for name in STAT_NAMES),
        *(f"group_wall_{name}_us" for name in STAT_NAMES),
        "log_dev_bw_gbps",
        "log_wall_bw_gbps",
    ]
    for timing in timing_rows:
        key = matrix_key(timing)
        if key not in summary_by_key:
            raise ValueError(f"log configuration is absent from summary: {key}")
        summary = summary_by_key[key]
        if int(timing["start_skew_avg_us"]) != int(summary["start_skew_avg_us"]):
            raise ValueError(f"StartSkew mismatch for {key}")
        if int(timing["group_wall_avg_us"]) != int(summary["group_wall_avg_us"]):
            raise ValueError(f"GroupWall mismatch for {key}")
        if abs(float(timing["log_wall_bw_gbps"]) - float(summary["wall_bw_gbps"])) > 0.0005:
            raise ValueError(f"WallBW mismatch for {key}")
        summary_dev_bw = summary["dev_bw_gbps"]
        log_dev_bw = timing["log_dev_bw_gbps"]
        if (summary_dev_bw is None) != (log_dev_bw is None):
            raise ValueError(f"DevBW availability mismatch for {key}")
        if summary_dev_bw is not None and abs(float(summary_dev_bw) - float(log_dev_bw)) > 0.0005:
            raise ValueError(f"DevBW mismatch for {key}")

        output = dict(summary)
        output.update({field: timing[field] for field in timing_fields})
        output_rows.append(output)

    output_rows.sort(key=lambda row: int(row["matrix_index"]))
    fieldnames = ["matrix_index", *[key for key in rows[0] if key != "matrix_index"]]
    fieldnames.extend(
        field
        for field in timing_fields
        if field != "matrix_index" and field not in fieldnames
    )
    FULL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FULL_DATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


def enrich_rows(rows: list[dict[str, object]], logs: list[Path] | None) -> None:
    if logs:
        timing_rows = parse_log_fragments([path.resolve() for path in logs])
        write_full_data(rows, timing_rows)
    if not FULL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"full timing data does not exist: {FULL_DATA_PATH}; provide --logs"
        )

    with FULL_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        full_rows = list(csv.DictReader(file, delimiter="\t"))
    full_by_key = {matrix_key(row): row for row in full_rows}
    if len(full_by_key) != 144:
        raise ValueError(f"expected 144 full timing rows, got {len(full_by_key)}")

    numeric_fields = [
        "matrix_index",
        *(f"submit_{name}_us" for name in STAT_NAMES),
        *(f"start_skew_{name}_us" for name in STAT_NAMES),
        *(f"group_wall_{name}_us" for name in STAT_NAMES),
    ]
    for row in rows:
        full = full_by_key[matrix_key(row)]
        for field in numeric_fields:
            row[field] = int(full[field])
        for field in (f"copy_{name}_us" for name in STAT_NAMES):
            row[field] = None if full[field] in {"", "None", "N/A"} else int(full[field])


def validate(rows: list[dict[str, object]]) -> None:
    if len(rows) != 144:
        raise ValueError(f"expected 144 rows, got {len(rows)}")
    if any(row["exit_code"] != 0 for row in rows):
        raise ValueError("matrix contains a failed row")

    for method in METHODS:
        for host_mode in HOST_MODES:
            for devices in DEVICES:
                for streams in STREAMS:
                    for submit_mode in SUBMIT_MODES:
                        for stream_sync in SYNC_MODES:
                            matches = [
                                row
                                for row in rows
                                if row["method"] == method
                                and row["host_mode"] == host_mode
                                and row["devices"] == devices
                                and row["streams"] == streams
                                and row["submit_mode"] == submit_mode
                                and row["stream_sync"] == stream_sync
                            ]
                            if len(matches) != 1:
                                raise ValueError(
                                    "matrix cell is not unique: "
                                    f"{method}/{host_mode}/d{devices}/s{streams}/"
                                    f"{submit_mode}/{stream_sync}"
                                )


def select_one(rows: list[dict[str, object]], **query: object) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if all(row[key] == value for key, value in query.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {query}, got {len(matches)}")
    return matches[0]


def method_label(method: str) -> str:
    return method.upper()


def host_label(host_mode: str) -> str:
    return "共享 Host 内存" if host_mode == "one_share" else "非共享 Host 内存"


def plot_absolute_bandwidth(
    rows: list[dict[str, object]], method: str, host_mode: str
) -> str:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=False)

    handles = []
    labels = []
    for axis, devices in zip(axes, DEVICES):
        panel_values = []
        for submit_mode, stream_sync, label, color, line_style, marker in SERIES:
            values = [
                float(
                    select_one(
                        rows,
                        method=method,
                        host_mode=host_mode,
                        devices=devices,
                        streams=streams,
                        submit_mode=submit_mode,
                        stream_sync=stream_sync,
                    )["wall_bw_gbps"]
                )
                for streams in STREAMS
            ]
            panel_values.extend(values)
            (line,) = axis.plot(
                STREAMS,
                values,
                color=color,
                linestyle=line_style,
                linewidth=2.4,
                marker=marker,
                markersize=6.5,
                markerfacecolor="white",
                markeredgewidth=2,
                label=label,
            )
            if devices == DEVICES[0]:
                handles.append(line)
                labels.append(label)

        axis.set_title(f"{devices} 卡")
        axis.set_xticks(STREAMS, labels=[str(stream) for stream in STREAMS])
        axis.set_xlim(0.5, 16.5)
        axis.set_ylim(0, max(panel_values) * 1.18)
        axis.tick_params(labelsize=10)
        axis.grid(True, color="#dfe4ec", linewidth=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel("Stream 数")

    axes[0].set_ylabel("WallBW (GiB/s)")
    figure.suptitle(
        f"{method_label(method)} · {host_label(host_mode)} · GLM5.1 绝对 WallBW",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.14, top=0.77, wspace=0.28)

    file_name = f"ascend_h2d_glm512_{method}_{host_mode}_wallbw.png"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(IMAGE_DIR / file_name, dpi=180, facecolor="white")
    plt.close(figure)
    return file_name


def markdown_table(headers: list[str], table_rows: list[list[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---:" for _ in headers) + " |"
    body = "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table_rows
    )
    return "\n".join((header, separator, body))


def format_stats(row: dict[str, object], prefix: str) -> str:
    values = [row[f"{prefix}_{name}_us"] for name in STAT_NAMES]
    if any(value is None for value in values):
        return "N/A"
    return " / ".join(str(value) for value in values)


def complete_timing_table(
    rows: list[dict[str, object]], method: str, host_mode: str
) -> str:
    table_rows = []
    for devices in DEVICES:
        for streams in STREAMS:
            for submit_mode in SUBMIT_MODES:
                for stream_sync in SYNC_MODES:
                    row = select_one(
                        rows,
                        method=method,
                        host_mode=host_mode,
                        devices=devices,
                        streams=streams,
                        submit_mode=submit_mode,
                        stream_sync=stream_sync,
                    )
                    table_rows.append(
                        [
                            devices,
                            streams,
                            "SM" if submit_mode == "stream-major" else "RR",
                            stream_sync.upper(),
                            format_stats(row, "submit"),
                            format_stats(row, "copy"),
                            (
                                "N/A"
                                if row["dev_bw_gbps"] is None
                                else f'{float(row["dev_bw_gbps"]):.3f}'
                            ),
                            format_stats(row, "start_skew"),
                            format_stats(row, "group_wall"),
                            f'{float(row["wall_bw_gbps"]):.3f}',
                            row["exit_code"],
                        ]
                    )

    return markdown_table(
        [
            "Devices",
            "Streams",
            "Submit",
            "Sync",
            "Submit us Min/Max/Avg/P50/P90",
            "Copy us Min/Max/Avg/P50/P90",
            "DevBW",
            "StartSkew us Min/Max/Avg/P50/P90",
            "GroupWall us Min/Max/Avg/P50/P90",
            "WallBW",
            "Exit",
        ],
        table_rows,
    )


def bw(
    rows: list[dict[str, object]],
    method: str,
    host_mode: str,
    devices: int,
    streams: int,
    submit_mode: str,
    stream_sync: str,
) -> float:
    return float(
        select_one(
            rows,
            method=method,
            host_mode=host_mode,
            devices=devices,
            streams=streams,
            submit_mode=submit_mode,
            stream_sync=stream_sync,
        )["wall_bw_gbps"]
    )


def build_report(rows: list[dict[str, object]], images: dict[tuple[str, str], str]) -> str:
    run_id = "未记录"
    log_path = str(rows[0]["log"])
    marker = "glm512_144_matrix/"
    if marker in log_path:
        run_id = log_path.split(marker, 1)[1].split("/", 1)[0]

    sections = []
    section_number = 1
    for method, host_mode in (
        ("ce", "one_share"),
        ("ffts", "one_share"),
        ("ce", "all"),
        ("ffts", "all"),
    ):
        sections.append(
            f"### 4.{section_number} {method_label(method)} · {host_label(host_mode)}\n\n"
            + complete_timing_table(rows, method, host_mode)
        )
        section_number += 1
    raw_tables = "\n\n".join(sections)

    method_comparison_rows = []
    for host_mode in HOST_MODES:
        for devices in DEVICES:
            ce = select_one(
                rows,
                method="ce",
                host_mode=host_mode,
                devices=devices,
                streams=4,
                submit_mode="stream-major",
                stream_sync="stream",
            )
            ffts = select_one(
                rows,
                method="ffts",
                host_mode=host_mode,
                devices=devices,
                streams=4,
                submit_mode="stream-major",
                stream_sync="stream",
            )
            method_comparison_rows.append(
                [
                    host_label(host_mode),
                    devices,
                    f'{float(ce["wall_bw_gbps"]):.3f}',
                    f'{float(ffts["wall_bw_gbps"]):.3f}',
                    ce["submit_avg_us"],
                    ffts["submit_avg_us"],
                    ce["group_wall_avg_us"],
                    ffts["group_wall_avg_us"],
                ]
            )
    method_comparison_table = markdown_table(
        [
            "Host 内存",
            "Devices",
            "CE WallBW",
            "FFTS WallBW",
            "CE Submit Avg",
            "FFTS Submit Avg",
            "CE GroupWall Avg",
            "FFTS GroupWall Avg",
        ],
        method_comparison_rows,
    )

    memory_comparison_table = markdown_table(
        ["Method", "Devices", "共享 Host WallBW", "非共享 Host WallBW"],
        [
            [
                method_label(method),
                devices,
                f'{bw(rows, method, "one_share", devices, 4, "stream-major", "stream"):.3f}',
                f'{bw(rows, method, "all", devices, 4, "stream-major", "stream"):.3f}',
            ]
            for method in METHODS
            for devices in DEVICES
        ],
    )

    device_scaling_table = markdown_table(
        ["Method", "Host 内存", "1 卡", "8 卡", "16 卡"],
        [
            [
                method_label(method),
                host_label(host_mode),
                *(
                    f'{bw(rows, method, host_mode, devices, 4, "stream-major", "stream"):.3f}'
                    for devices in DEVICES
                ),
            ]
            for method in METHODS
            for host_mode in HOST_MODES
        ],
    )

    stream_scaling_table = markdown_table(
        ["Method", "Host 内存", "1 stream", "4 streams", "16 streams"],
        [
            [
                method_label(method),
                host_label(host_mode),
                *(
                    f'{bw(rows, method, host_mode, 16, streams, "stream-major", "stream"):.3f}'
                    for streams in STREAMS
                ),
            ]
            for method in METHODS
            for host_mode in HOST_MODES
        ],
    )

    sync_comparison_table = markdown_table(
        [
            "Method",
            "Host 内存",
            "1 stream EVENT/STREAM",
            "4 streams EVENT/STREAM",
            "16 streams EVENT/STREAM",
        ],
        [
            [
                method_label(method),
                host_label(host_mode),
                *(
                    f'{bw(rows, method, host_mode, 16, streams, "stream-major", "event"):.3f} / '
                    f'{bw(rows, method, host_mode, 16, streams, "stream-major", "stream"):.3f}'
                    for streams in STREAMS
                ),
            ]
            for method in METHODS
            for host_mode in HOST_MODES
        ],
    )

    submit_comparison_table = markdown_table(
        [
            "Method",
            "Host 内存",
            "1 stream SM/RR",
            "4 streams SM/RR",
            "16 streams SM/RR",
        ],
        [
            [
                method_label(method),
                host_label(host_mode),
                *(
                    f'{bw(rows, method, host_mode, 16, streams, "stream-major", "stream"):.3f} / '
                    f'{bw(rows, method, host_mode, 16, streams, "round-robin", "stream"):.3f}'
                    for streams in STREAMS
                ),
            ]
            for method in METHODS
            for host_mode in HOST_MODES
        ],
    )

    event_ffts_all_s4 = select_one(
        rows,
        method="ffts",
        host_mode="all",
        devices=16,
        streams=4,
        submit_mode="stream-major",
        stream_sync="event",
    )
    event_ffts_all_s16 = select_one(
        rows,
        method="ffts",
        host_mode="all",
        devices=16,
        streams=16,
        submit_mode="stream-major",
        stream_sync="event",
    )
    stream_ffts_all_s16 = select_one(
        rows,
        method="ffts",
        host_mode="all",
        devices=16,
        streams=16,
        submit_mode="stream-major",
        stream_sync="stream",
    )
    submit_ffts_shared_s16_sm = select_one(
        rows,
        method="ffts",
        host_mode="one_share",
        devices=16,
        streams=16,
        submit_mode="stream-major",
        stream_sync="stream",
    )
    submit_ffts_shared_s16_rr = select_one(
        rows,
        method="ffts",
        host_mode="one_share",
        devices=16,
        streams=16,
        submit_mode="round-robin",
        stream_sync="stream",
    )

    report = f"""# Ascend H2D GLM5.1 CE/FFTS 144 组矩阵性能报告

## 1. 测试说明

本报告整理一次关闭 Trace 的 144 组完整矩阵，只展示实测绝对带宽。四张折线图中的每一个点都对应一个原始矩阵单元，合计覆盖全部 144 组数据。

### 1.1 测试矩阵与固定工作负载

- Method：`ce`、`ffts`
- Host 内存形态：共享 Host 内存、非共享 Host 内存
- 参数映射：共享 Host 内存对应 `one_share`；非共享 Host 内存对应 `all`
- Devices：1、8、16
- Streams：1、4、16
- Submit mode：`stream-major`、`round-robin`
- Stream sync：`event`、`stream`
- IO 模式：`glm5.1`
- 每个 task：128 KiB + 16 KiB + 32 KiB，共 176 KiB
- 每卡 task 数：512；每卡每轮传输量：88 MiB
- 正式迭代：128 轮
- FFTS ready lane 上限：3
- 进程同步：`barrier`；stream start gate：`off`
- 运行标识：`{run_id}`
- 结果状态：144/144 组 `exit_code=0`

这里的 task 不是一次单独的 memcpy。一个 GLM5.1 task 固定包含三段连续 IO：

```text
一个 task，共 176 KiB

Host:   [ 128 KiB ][ 16 KiB ][ 32 KiB ]
                    H2D
Device: [ 128 KiB ][ 16 KiB ][ 32 KiB ]

每卡每轮：512 tasks × 176 KiB = 88 MiB
```

### 1.2 CE 与 FFTS 表示什么

CE 和 FFTS 搬运的数据、task 数以及 task 到 stream 的分配完全相同，区别在于一个 task 如何提交给 Ascend：

```text
同一个 GLM5.1 task
        |
        +-- CE
        |     128 KiB -> aclrtMemcpyAsync --+
        |      16 KiB -> aclrtMemcpyAsync --+--> 同一条 stream
        |      32 KiB -> aclrtMemcpyAsync --+
        |
        +-- FFTS Direct H2D
              3 个 copy spec
                    |
                    +--> dispatcher.BuildCopies
                    +--> dispatcher.Launch(stream, readyCount)
                         一次 launch 描述该 task 的三段搬运
```

- **CE**：每个 task 调用三次 `aclrtMemcpyAsync`，三段 IO 都下发到该 task 选中的同一条 stream。
- **FFTS**：每个 task 构造三个 SDMA copy spec，再通过 FFTS Direct H2D dispatcher 做一次 launch。本轮固定 `FFTS_MAX_READY_LANES=3`。
- 两者都是异步提交。Host API 返回只表示任务已经入队，不表示数据已经搬完；完成由后面的 stream sync 阶段确认。

### 1.3 共享 Host 内存、非共享 Host 内存与设备进程

多卡 case 使用 fork runner。设备数为 N 时创建 N 个子进程；每个子进程只绑定一张 device，并在自己的进程内创建、提交和同步这张卡的 streams。不存在一个 submit 线程跨多个设备进程下发任务。

```text
                         父进程
              创建 barrier 和共享计时区
                           |
                    fork N 个子进程
                           |
        +------------------+------------------+
        |                  |                  |
     子进程 P0          子进程 P1         子进程 P(N-1)
     bind device 0       bind device 1      bind device N-1
     streams of D0       streams of D1      streams of D(N-1)

共享 Host 内存：

              同一个 POSIX Shared Host Region
                  |          |          |
                 P0         P1       P(N-1)
                  |          |          |
                 D0         D1       D(N-1)

非共享 Host 内存：

             Host 0      Host 1      Host N-1
                |           |            |
               P0          P1         P(N-1)
                |           |            |
               D0          D1         D(N-1)
```

- **共享 Host 内存**：所有设备子进程重新映射同一个 POSIX Shared Memory 对象，并分别为自己的 device 注册/映射这块 Host 内存。所有卡读取的是同一份 Host 数据源。原始测试参数名是 `one_share`。
- **非共享 Host 内存**：每个设备子进程创建自己的 Host buffer，卡与 Host 源一一对应，各进程之间不共享这个数据区。它避免多卡共同读取同一个 Host 源，更接近各卡独立供数时的带宽上限。原始测试参数名是 `all`。

### 1.4 submit mode 表示什么

submit mode 只改变 task 如何分给 streams，以及 Host 按什么顺序调用提交 API；它不改变 task 内容和总数据量。下面以 8 个 task、4 条 stream 为例：

```text
stream-major（SM）：连续分块，按 stream 提交

S0: T0 T1    S1: T2 T3    S2: T4 T5    S3: T6 T7
Host 提交顺序：S0(T0,T1) -> S1(T2,T3) -> S2(T4,T5) -> S3(T6,T7)

round-robin（RR）：taskIndex % streamCount，轮询提交

S0: T0 T4    S1: T1 T5    S2: T2 T6    S3: T3 T7
Host 提交顺序：S0(T0) -> S1(T1) -> S2(T2) -> S3(T3)
             -> S0(T4) -> S1(T5) -> S2(T6) -> S3(T7)
```

当 streams=1 时，SM 和 RR 都只能提交到 S0，二者在任务映射和提交顺序上等价。因此这组数据也可以作为独立运行波动的 sanity check。

### 1.5 stream sync 表示什么

stream sync 是**单个设备子进程内部**等待本设备所有 streams 完成的方式，不负责不同设备进程之间的起点对齐。

```text
EVENT：每条 stream 记录 end event，再汇聚到 S0

S0: [copies] -> endE0 ---------------------------> totalEnd -> sync S0
S1: [copies] -> endE1 -> S0 wait endE1 ---------^
S2: [copies] -> endE2 -> S0 wait endE2 ---------^
...                                               |
所有 stream 的 event 依赖到齐后，S0 上的 totalEnd 才能完成

STREAM：Host 直接逐条等待

Host: sync S0 -> sync S1 -> sync S2 -> ... -> sync S(last)
      某次调用可能阻塞；函数返回后，本设备所有 streams 均已完成
```

- **`event`**：在每条 stream 尾部记录 end event；stream 0 等待其他 streams 的 end event，再记录 total end，最后只同步 stream 0。该模式还能用 device Event 计算 `DevBW`。
- **`stream`**：不创建这些完成 Event，Host 对所有 streams 逐一调用 `aclrtSynchronizeStream`。虽然调用顺序是逐条的，但各 stream 上的设备任务此前已经异步下发，可以并行执行。
- 本轮 `stream_start_gate=off`，因此没有额外 Event 把同一设备内的各 streams 强制对齐到同一个 device 起点。

### 1.6 进程同步、GroupWall 与 WallBW

`process_sync=barrier` 位于所有设备子进程之间。每轮开始前，N 个子进程先在共享 barrier 会合，再从共同约定的 release time 继续；barrier 之后分别记录 Host start。每个子进程完成本设备的 submit 和 stream sync 后记录 Host end。

```text
                         process barrier
P0 / Device 0   -----------| start0 ===================== end0
P1 / Device 1   -----------|   start1 =============== end1
P2 / Device 2   -----------|  start2 ========================= end2
...                         |
P(N-1)          -----------| startN =================== endN
                            ^                               ^
                       min(start_i)                    max(end_i)
                            |<------ GroupWall ------------>|

StartSkew = max(start_i) - min(start_i)
GroupWall = max(end_i)   - min(start_i)
WallBW    = 所有设备总传输量 / GroupWall Avg
```

因此，barrier 只是尽量对齐各进程的 Host 起点，并不会让各设备同时完成。`StartSkew` 反映 barrier 释放后各子进程真正进入本轮的起点偏差；`GroupWall` 从最早设备开始一直覆盖到最慢设备结束，是多卡端到端完成区间。

本轮每卡每轮搬运 88 MiB。例如 16 卡总量为 16 × 88 MiB = 1.375 GiB；某个配置的 `GroupWall Avg` 为 5342 us，则：

```text
WallBW = 1.375 GiB / 0.005342 s = 257.4 GiB/s
```

## 2. 核心结论

1. **FFTS 在共享 Host 内存、16 卡同时拷贝场景下存在明显瓶颈。** STREAM + SM 下，1、4、16 streams 的 WallBW 分别为 **{bw(rows, "ffts", "one_share", 16, 1, "stream-major", "stream"):.3f}**、**{bw(rows, "ffts", "one_share", 16, 4, "stream-major", "stream"):.3f}**、**{bw(rows, "ffts", "one_share", 16, 16, "stream-major", "stream"):.3f}** GiB/s，增加 streams 没有继续提升；4 streams 时，8 卡和 16 卡分别为 **{bw(rows, "ffts", "one_share", 8, 4, "stream-major", "stream"):.3f}** 和 **{bw(rows, "ffts", "one_share", 16, 4, "stream-major", "stream"):.3f}** GiB/s，增加设备数也没有继续提升。作为对照，非共享 Host 内存、16 卡、4 streams 达到 **{bw(rows, "ffts", "all", 16, 4, "stream-major", "stream"):.3f}** GiB/s。

2. **在本轮重点的多卡多流 FFTS 场景下，STREAM sync 优于 EVENT sync。** 共享 Host 内存、16 卡、16 streams、SM 下，STREAM/EVENT 分别为 **{bw(rows, "ffts", "one_share", 16, 16, "stream-major", "stream"):.3f} / {bw(rows, "ffts", "one_share", 16, 16, "stream-major", "event"):.3f}** GiB/s；非共享 Host 内存的同配置分别为 **{bw(rows, "ffts", "all", 16, 16, "stream-major", "stream"):.3f} / {bw(rows, "ffts", "all", 16, 16, "stream-major", "event"):.3f}** GiB/s。因此 FFTS 多卡多流配置优先使用 STREAM sync。

## 3. 性能趋势图

本章按 Host 数据源是否被设备进程共享来分图：

- **共享 Host 内存**：所有设备进程从同一个 POSIX 共享内存区读取数据，观察多卡共同读取一个 Host 源时的表现。
- **非共享 Host 内存**：每个设备进程从自己的独立 Host buffer 读取数据，观察每卡独立供数时的表现。

每张图固定一种 Method 和 Host 内存形态，三个子图分别对应 1、8、16 卡。横轴为 stream 数，四条线完整保留 submit mode 和 sync mode。各子图纵轴都从 0 开始，并按该设备规模的绝对带宽单独设置上限。

### 3.1 CE · 共享 Host 内存

![CE shared Host memory absolute WallBW](../_static/images/{images[("ce", "one_share")]})

### 3.2 FFTS · 共享 Host 内存

![FFTS shared Host memory absolute WallBW](../_static/images/{images[("ffts", "one_share")]})

### 3.3 CE · 非共享 Host 内存

![CE non-shared Host memory absolute WallBW](../_static/images/{images[("ce", "all")]})

### 3.4 FFTS · 非共享 Host 内存

![FFTS non-shared Host memory absolute WallBW](../_static/images/{images[("ffts", "all")]})

## 4. 真实数据

以下四张表与四张图一一对应。SM 表示 `stream-major`，RR 表示 `round-robin`。每张表包含 36 个矩阵单元，四张表共覆盖全部 144 组数据。Submit、Copy、StartSkew 和 GroupWall 的顺序均为 `Min/Max/Avg/P50/P90`，时间单位为微秒；DevBW 和 WallBW 单位为 GiB/s。

多卡 Submit/Copy 时间先按每轮所有设备进程中的最大值合并，再对 128 轮计算统计量。STREAM 模式不使用 device Event 统计 Copy 时间，因此 Copy 和 DevBW 显示为 N/A；它仍然通过 GroupWall 和 WallBW 记录端到端完成时间。原始汇总保存在 `docs/source/developer-guide/data/ascend_h2d_glm512_144_matrix.tsv`，从完整日志解析出的全量数据保存在 `docs/source/developer-guide/data/ascend_h2d_glm512_144_matrix_full.tsv`。

{raw_tables}

## 4. 性能结果

### 4.1 六个矩阵维度分别验证什么

144 组不是把参数简单排列，而是用受控变量回答六个不同问题：

| 对比维度 | 对比时保持不变的条件 | 要回答的问题 |
|---|---|---|
| CE / FFTS | Host 内存、卡数、streams、submit、sync | 改变提交与执行机制后，端到端完成时间是否下降 |
| 共享 / 非共享 Host 内存 | Method、卡数、streams、submit、sync | 多卡共同读取一个 Host 源是否成为瓶颈 |
| 1 / 8 / 16 卡 | 其余五个维度不变 | 增加设备后聚合带宽能否继续扩展 |
| 1 / 4 / 16 streams | 其余五个维度不变 | 增加单卡并发是否有效，何时出现过量并发 |
| EVENT / STREAM | Method、Host 内存、卡数、streams、submit | 完成汇聚方式对端到端时间的影响 |
| SM / RR | Method、Host 内存、卡数、streams、sync | 连续分块提交和轮询提交是否改变执行结果 |

下面每个小节只改变一个维度。除非另行说明，数值均为 `WallBW`，单位 GiB/s。

### 4.2 CE 与 FFTS：提交机制对比

下表固定为 4 streams、SM、STREAM，只改变 Method：

{method_comparison_table}

六个 Host/卡数组合中 FFTS 都高于 CE。这里的收益不能简单解释成“FFTS 的 Host 下发更快”：共享 Host 内存 16 卡时，FFTS Submit Avg 为 **5166 us**，反而高于 CE 的 **2712 us**，但 FFTS GroupWall Avg 只有 **13272 us**，明显低于 CE 的 **30112 us**。说明主要收益落在任务执行和完成区间，而不是只由 Submit API 时间决定。

非共享 Host 内存 16 卡时，FFTS 同时把 Submit Avg 从 **3018 us** 降到 **2588 us**，把 GroupWall Avg 从 **19862 us** 降到 **5393 us**，对应 WallBW 从 **69.228** 提升到 **254.960 GiB/s**。

### 4.3 共享与非共享 Host 内存：Host 数据源对比

下表同样固定为 4 streams、SM、STREAM，只改变 Host 内存形态：

{memory_comparison_table}

单卡差异同时包含两种 Host buffer 创建和映射路径的差别，不能解释为多卡共享争用。进入 8 卡和 16 卡后，趋势更清楚：FFTS 的共享 Host 内存结果为 **102.139、103.602 GiB/s**，已经停在约 104 GiB/s；非共享 Host 内存则达到 **176.917、254.960 GiB/s**。这一现象与 FFTS 把设备侧搬运能力拉高后、多卡共同读取同一个 Host 源成为主要限制相符。

CE 也表现为非共享 Host 内存更高，但从共享到非共享的绝对差距小于 FFTS，因为 CE 本身的完成时间仍然较长，Host 共享源还没有成为唯一主导因素。

### 4.4 1、8、16 卡：设备扩展性对比

下表固定为 4 streams、SM、STREAM，只改变设备数：

{device_scaling_table}

FFTS 共享 Host 内存从 8 卡到 16 卡时，WallBW 仅从 **102.139** 变为 **103.602 GiB/s**。这期间总数据量翻倍，GroupWall Avg 也从 **6731 us** 增长到 **13272 us**，因此聚合带宽基本不变。

FFTS 非共享 Host 内存从 8 卡到 16 卡时，GroupWall Avg 只从 **3886 us** 增长到 **5393 us**，小于总数据量的增长，WallBW 因而从 **176.917** 继续提升到 **254.960 GiB/s**。这组对比直接区分了“设备扩展不足”和“共享 Host 源限制”。

### 4.5 1、4、16 streams：单卡并发度对比

下表固定为 16 卡、SM、STREAM，只改变 stream 数：

{stream_scaling_table}

- CE 共享 Host 内存从 1 到 4 streams 提升明显，4 到 16 streams 仍有收益但幅度收窄。
- CE 非共享 Host 内存从 **43.902**、**69.228** 持续提升到 **92.774 GiB/s**，说明当前配置仍能从更高的 stream 并发度获益。
- FFTS 共享 Host 内存在三档 stream 下分别为 **104.009、103.602、104.475 GiB/s**，几乎不变；增加 stream 无法突破共享源平台。
- FFTS 非共享 Host 内存在 4 streams 达到 **254.960 GiB/s**，但 16 streams 降到 **197.359 GiB/s**。增加到 16 streams 没有继续带来收益，额外队列、调度或资源竞争可能抵消了并发收益；当前数据不能进一步区分这些因素。

因此，“更多 streams 更快”只对部分 CE 组合成立；FFTS 的最佳点取决于 Host 内存形态和卡数。

### 4.6 EVENT 与 STREAM：完成方式对比

下表固定为 16 卡、SM；每个单元按 `EVENT / STREAM` 展示：

{sync_comparison_table}

EVENT 并非在所有位置都更慢。例如非共享 Host 内存、1 stream 时，CE 为 **63.702 / 43.902**，FFTS 为 **231.676 / 211.149**。此时只有一条 stream，不存在多 stream Event fan-in，EVENT 的设备计时路径不一定产生劣势。

随着 streams 增加，EVENT 的多 stream 汇聚成本开始显现。16 卡非共享 FFTS 的 EVENT Copy Avg 在 4 streams 和 16 streams 下分别为 **{event_ffts_all_s4["copy_avg_us"]} us** 和 **{event_ffts_all_s16["copy_avg_us"]} us**，设备 Event 覆盖的 Copy 时间几乎没有增加；但 GroupWall Avg 从 **{event_ffts_all_s4["group_wall_avg_us"]} us** 增长到 **{event_ffts_all_s16["group_wall_avg_us"]} us**，WallBW 从 **{float(event_ffts_all_s4["wall_bw_gbps"]):.3f}** 降到 **{float(event_ffts_all_s16["wall_bw_gbps"]):.3f}** GiB/s。这证明新增时间主要位于 Copy Event 计时区间之外；它可能来自 Host 下发、Event/同步完成处理或进程尾部，当前矩阵不能继续拆分。

同一 16-stream 配置改用 STREAM 后，GroupWall Avg 为 **{stream_ffts_all_s16["group_wall_avg_us"]} us**，WallBW 恢复到 **{float(stream_ffts_all_s16["wall_bw_gbps"]):.3f}** GiB/s。矩阵因此验证了 sync 方式与 stream 数存在明显交互，不能只在单 stream 上比较两种完成方式。

### 4.7 SM 与 RR：下发顺序对比

下表固定为 16 卡、STREAM；每个单元按 `SM / RR` 展示：

{submit_comparison_table}

没有一种 submit mode 在所有组合中稳定胜出。streams=1 时 SM 和 RR 的任务映射完全相同，但非共享 CE 仍得到 **43.902 / 56.040 GiB/s**，说明单次独立运行存在不可忽略的波动，不能把这一差值解释成下发顺序收益。

更有代表性的是共享 Host 内存、FFTS、16 streams：SM/RR 的 Submit Avg 分别为 **{submit_ffts_shared_s16_sm["submit_avg_us"]} / {submit_ffts_shared_s16_rr["submit_avg_us"]} us**，差异很大；但 GroupWall Avg 为 **{submit_ffts_shared_s16_sm["group_wall_avg_us"]} / {submit_ffts_shared_s16_rr["group_wall_avg_us"]} us**，WallBW 为 **{float(submit_ffts_shared_s16_sm["wall_bw_gbps"]):.3f} / {float(submit_ffts_shared_s16_rr["wall_bw_gbps"]):.3f}** GiB/s，几乎相同。这与共享 Host 源主导最终完成带宽相符，也说明 Submit Avg 的差异不会必然转化成 WallBW 差异。

"""
    report = report.split("\n## 4. 性能结果", 1)[0]
    return report


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary)
    validate(rows)
    enrich_rows(rows, args.logs)
    images = {
        (method, host_mode): plot_absolute_bandwidth(rows, method, host_mode)
        for method, host_mode in (
            ("ce", "one_share"),
            ("ffts", "one_share"),
            ("ce", "all"),
            ("ffts", "all"),
        )
    }
    REPORT_PATH.write_text(build_report(rows, images), encoding="utf-8")
    print(f"validated rows: {len(rows)}")
    print(f"report: {REPORT_PATH.relative_to(DOCS_DIR.parent)}")
    print(f"images: {len(images)} PNG files")


if __name__ == "__main__":
    main()
