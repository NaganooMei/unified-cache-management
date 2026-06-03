# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os

from cache_h2d_ffts_pipeline_test import (
    cache_buffer_capacity_gb_for_case,
    env_bool,
    env_float,
    env_int,
    parse_model_case,
    prepare_torch_backend,
    print_case_config,
    print_result,
    print_summary_table,
    run_transport,
)


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")

    case = parse_model_case()
    block_num = env_int("UCM_FFTS_BLOCK_NUM", 16)
    warmup = env_int("UCM_FFTS_WARMUP", 2)
    repeat = env_int("UCM_FFTS_REPEAT", 10)
    if block_num <= 0 or warmup < 0 or repeat <= 0:
        raise ValueError("UCM_FFTS_BLOCK_NUM and UCM_FFTS_REPEAT must be positive")
    device_type = os.getenv("UCM_FFTS_TORCH_DEVICE", "cuda")
    device_id = env_int("UCM_FFTS_DEVICE_ID", 0)
    prepare_torch_backend(device_type)
    min_gbps = env_float("UCM_FFTS_MIN_GBPS", 0.0)
    object_target_bytes = 0
    validate = env_bool("UCM_FFTS_VALIDATE", False)

    cache_buffer_capacity_gb = cache_buffer_capacity_gb_for_case(case.tensor_sizes)
    print_case_config(
        case,
        block_num,
        warmup,
        repeat,
        device_type,
        device_id,
        cache_buffer_capacity_gb,
        object_target_bytes,
        transport="ce",
        validate=validate,
    )
    result = run_transport(
        case.name,
        "ce",
        case.tensor_sizes,
        block_num,
        device_type,
        device_id,
        warmup,
        repeat,
        cache_buffer_capacity_gb,
        object_target_bytes,
        validate=validate,
    )
    print_result(result)

    if min_gbps > 0:
        assert result.gbps >= min_gbps, (
            f"CE bandwidth {result.gbps:.3f}GB/s is lower than "
            f"UCM_FFTS_MIN_GBPS={min_gbps:.3f}GB/s"
        )
    print_summary_table([result])


if __name__ == "__main__":
    main()
