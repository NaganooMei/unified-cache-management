#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark script for Unified Cache Management (UCM) prefix caching performance testing.

Usage example:
python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 0.5 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --seed 25 \
  --repeat 1

python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 0.5 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --seed 25 \
  --repeat 1


  python3 long_doc_pc_completion.py \
  --test-mode no-pc \
  --doc-len-list 2000 \
  --concurrency-list 1 \
  --hit-ratio 0.5 \
  --port 18081 \
  --model /home/models/QwQ-32B \
  --seed 25 \
  --repeat 1
"""

import argparse
import asyncio
import random
import time
import logging
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from openai import AsyncOpenAI, APIError, Timeout, RateLimitError
import string
import datetime
import os
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

@dataclass
class BenchmarkConfig:
    """Configuration dataclass for benchmark parameters."""
    num_docs: int
    doc_len: int
    output_len: int
    hit_ratio: float
    max_concurrency: int
    seed: int
    port: int
    model: str
    test_mode: str
    repeat: int

def validate_config(config: BenchmarkConfig) -> None:
    """
    Validate benchmark configuration parameters.
    
    Args:
        config: Benchmark configuration object
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if config.num_docs <= 0:
        raise ValueError("num-docs must be positive")
    if config.doc_len <= 0:
        raise ValueError("doc-len must be positive")
    if config.output_len <= 0:
        raise ValueError("output-len must be positive")
    if not (0 <= config.hit_ratio <= 1):
        raise ValueError("hit-ratio must be between 0 and 1")
    if config.max_concurrency <= 0:
        raise ValueError("max-concurrency must be positive")
    if config.port <= 0 or config.port > 65535:
        raise ValueError("port must be between 1 and 65535")
    if not config.model:
        raise ValueError("model path cannot be empty")
    if config.test_mode not in ["pc", "no-pc"]:
        raise ValueError("test-mode must be either 'pc' or 'no-pc'")
    if config.repeat <= 0:
        raise ValueError("repeat must be positive")

def rand_tokens(n: int, seed: int) -> str:
    """
    Generate a sequence of special characters and spaces, where length equals token count.
    
    Each character followed by a space approximates one token, providing consistent
    token-length text generation for benchmarking purposes.
    
    Args:
        n: Number of tokens to generate
        seed: Random seed for reproducible generation
        
    Returns:
        String containing n tokens (character + space pairs)
    """
    r = random.Random(seed)
    symbols = string.punctuation  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    chars = [r.choice(symbols) + " " for _ in range(n)]
    return "".join(chars)

def build_docs(num_docs: int, doc_len: int, seed: int) -> List[str]:
    """
    Construct benchmark documents with specified length and count.
    
    Args:
        num_docs: Number of documents to generate
        doc_len: Length of each document in tokens
        seed: Base random seed for document generation
        
    Returns:
        List of generated documents as strings
    """
    docs = []
    for i in range(num_docs):
        docs.append(f"{seed}_{i:02d} " + rand_tokens(doc_len, seed + i))
    return docs

def build_warmup_docs(seed: int, num_warmup: int = 1, doc_len: int = 2000) -> List[str]:
    """
    Generate warmup documents to prepare the system for benchmarking.
    
    Warmup documents use a special WARMUP_ prefix to avoid collision with
    regular benchmark documents and ensure cache is properly initialized.
    
    Args:
        seed: Random seed for warmup document generation
        num_warmup: Number of warmup documents to generate (default: 2)
        doc_len: Length of each warmup document in tokens (default: 8000)
        
    Returns:
        List of warmup documents as strings
    """
    docs = []
    for i in range(num_warmup):
        docs.append(f"WARMUP_Seed{seed}_{i:02d} " + rand_tokens(doc_len, seed + i))
    return docs

def make_prompts(docs: List[str], hit_ratio: float, miss_seed: int = 12345) -> List[str]:
    """
    Create prompts with controlled token-level cache hit ratios.
    
    This function splits each document into a hit portion (cached prefix) and
    a miss portion (new content), allowing precise control over cache hit rates
    for benchmarking prefix caching performance.
    
    Args:
        docs: List of original documents
        hit_ratio: Ratio of tokens to keep from original document (0-1)
        miss_seed: Random seed for generating miss portion content
        
    Returns:
        List of prompts with specified hit/miss composition
    """
    prompts = []
    for i, doc in enumerate(docs):
        r = random.Random(miss_seed + i)  # Deterministic miss portion
        tokens = doc.split()  # Space-separated tokens
        total = len(tokens)
        hit_n = int(total * hit_ratio + 0.5)  # Number of tokens to keep
        hit_part = " ".join(tokens[:hit_n])
        miss_n = total - hit_n
        # Generate miss portion: random symbols + spaces, length = miss_n tokens
        miss_part = "".join(r.choice(string.punctuation) + " " for _ in range(miss_n))
        # Combine: hit prefix | random miss suffix (without original tokens)
        prompts.append(hit_part + " " + miss_part.rstrip())
    return prompts

def has_content_completions(chunk):
    """
    Completions streaming emits text at choices[0].text.
    """
    return bool(chunk.choices) and (chunk.choices[0].text is not None)

def extract_content_completions(chunk):
    """
    Extract content from a Completions stream chunk.
    """
    return chunk.choices[0].text or ""

async def one_req(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    sem: asyncio.Semaphore,
    out_len: int,
    timeout: float = 300.0
) -> Dict[str, float]:
    """
    Execute a single request with comprehensive error handling and timeout.
    
    Args:
        client: AsyncOpenAI client instance
        model: Model identifier
        prompt: Input prompt text
        sem: Semaphore for concurrency control
        out_len: Maximum output length in tokens
        timeout: Request timeout in seconds (default: 300.0)
        
    Returns:
        Dictionary containing TTFT (Time To First Token) and total request time in milliseconds
    """
    async with sem:
        t0 = time.time()
        first = None
        
        try:
            # Create request with timeout
            response = await asyncio.wait_for(
                client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=out_len,
                    temperature=0,
                    stream=True,
                    stream_options={"include_usage": True}
                ),
                timeout=timeout
            )
            
            pieces = []
            async for chunk in response:
                if not has_content_completions(chunk):
                    continue

                content = extract_content_completions(chunk)
                if first is None and content.strip():
                    first = time.time()
                pieces.append(content)

            # ttft is never 0.0 so it is an immediate tell
            # that the request produced no output
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout after {timeout}s for prompt length {len(prompt)}")
            return {"ttft": float('inf'), "total": float('inf'), "error": "timeout"}
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return {"ttft": float('inf'), "total": float('inf'), "error": "rate_limit"}
        except APIError as e:
            logger.error(f"API error: {e}")
            return {"ttft": float('inf'), "total": float('inf'), "error": "api_error"}
        except Exception as e:
            logger.error(f"Unexpected error in request: {e}", exc_info=True)
            return {"ttft": float('inf'), "total": float('inf'), "error": "unknown"}
        
        ttft_ms = (first - t0) * 1000 if first else float('inf')
        total_ms = (time.time() - t0) * 1000
        
        logger.debug(
            f"Request completed - Start: {t0:.3f}, TTFT: {(first - t0) * 1000 if first else float('inf'):.3f}ms"
        )
        
        return {"ttft": ttft_ms, "total": total_ms, "error": None}

async def run_round(client, model, prompts, max_concurrency, out_len):
    sem = asyncio.Semaphore(max_concurrency)
    return await asyncio.gather(*(one_req(client, model, p, sem, out_len) for p in prompts))


async def run_prefix_caching_cycle(
    client: AsyncOpenAI,
    model: str,
    num_docs: int,
    doc_len: int,
    max_concurrency: int,
    hit_ratio: float,
    seed: int,
    out_len: int,
    test_mode: str
) -> Dict[str, float]:
    """
    Execute a complete benchmark cycle: prefix write → wait → prefix hit.
    
    This function performs the full benchmark workflow:
    1. Prefix write phase: Send documents to populate cache
    2. Wait phase: Allow cache to persist
    3. Prefix hit phase: Send queries with controlled hit ratios
    
    Args:
        client: AsyncOpenAI client instance
        model: Model identifier
        num_docs: Number of documents to process
        doc_len: Length of each document in tokens
        max_concurrency: Maximum concurrent requests
        hit_ratio: Cache hit ratio for query phase (0-1)
        seed: Random seed for reproducible results
        out_len: Maximum output length in tokens
        test_mode: Test mode - "pc" (prefix caching) or "no-pc" (baseline)
        
    Returns:
        Dictionary containing write_ttft and query_ttft metrics in milliseconds
    """
    
    # Phase 1: Prefix write phase
    phase_name = "prefix write" if test_mode == "pc" else "baseline (no caching)"
    logger.info(f">> Starting {phase_name} phase")
    
    docs = build_docs(num_docs, doc_len, seed)
    write_stats = await run_round(client, model, docs, max_concurrency, out_len)
    
    # Calculate average TTFT, filtering out failed requests
    successful_writes = [s for s in write_stats if s.get("error") is None]
    if not successful_writes:
        logger.error("All write requests failed!")
        write_ttft = float('inf')
    else:
        write_ttft = sum(s["ttft"] for s in successful_writes) / len(successful_writes)
    
    logger.info(f"Average write TTFT: {write_ttft:.2f} ms")
    
    if test_mode != "pc":
        logger.info("Baseline mode completed")
        logger.info(f"Document length: {doc_len}, Concurrency: {max_concurrency}, Mode: {test_mode}")
        logger.info(f"Baseline average TTFT: {write_ttft:.2f} ms")
        return {"write_ttft": write_ttft}
    
    # Phase 2: Wait for cache persistence
    wait_time = 8.0
    logger.info(f">> Waiting {wait_time}s for cache to persist...")
    await asyncio.sleep(wait_time)
    
    # Phase 3: Prefix hit phase
    logger.info(f">> Starting prefix hit phase (hit ratio: {hit_ratio:.0%})")
    query_prompts = make_prompts(docs, hit_ratio, miss_seed=seed + 10000)
    query_stats = await run_round(client, model, query_prompts, max_concurrency, out_len)
    
    # Calculate average query TTFT
    successful_queries = [s for s in query_stats if s.get("error") is None]
    if not successful_queries:
        logger.error("All query requests failed!")
        query_ttft = float('inf')
    else:
        query_ttft = sum(s["ttft"] for s in successful_queries) / len(successful_queries)
    
    # Log results
    logger.info("Benchmark cycle completed")
    logger.info(f"Document length: {doc_len}, Concurrency: {max_concurrency}, Hit ratio: {hit_ratio:.0%}")
    logger.info(f"Write TTFT: {write_ttft:.2f} ms")
    logger.info(f"Query TTFT: {query_ttft:.2f} ms")
    
    return {"write_ttft": write_ttft, "query_ttft": query_ttft}


async def main():
    """
    Main entry point for the UCM prefix caching benchmark.
    
    This function:
    1. Parses command-line arguments
    2. Validates configuration
    3. Initializes the OpenAI client
    4. Runs warmup phase
    5. Executes benchmark cycles for all parameter combinations
    6. Saves results to CSV file
    """
    
    parser = argparse.ArgumentParser(
        description="UCM Prefix Caching Benchmark - Test prefix caching performance under various conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prefix caching test
  python3 long_doc_pc.py --test-mode pc --doc-len 16000 --max-concurrency 4
  
  # Comprehensive benchmark with multiple document lengths and concurrency levels
  python3 long_doc_pc.py --test-mode pc --doc-len-list 4000 8000 16000 32000 \\
                         --concurrency-list 1 2 4 8 --repeat 3
  
  # Baseline test without prefix caching
  python3 long_doc_pc.py --test-mode no-pc --doc-len 16000 --max-concurrency 8
        """
    )
    
    # Required and optional parameters
    parser.add_argument(
        "--num-docs",
        type=int,
        default=4,
        help="Number of documents to generate per test (default: 4)"
    )
    parser.add_argument(
        "--doc-len",
        type=int,
        default=8000,
        help="Length of each document in tokens (default: 8000)"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="Maximum output length in tokens (default: 3)"
    )
    parser.add_argument(
        "--hit-ratio",
        type=float,
        default=0.7,
        help="Cache hit ratio for prefix caching tests, between 0 and 1 (default: 0.7)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum number of concurrent requests (semaphore slots) (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed for reproducible results (default: 5)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18081,
        help="OpenAI API server port (default: 18081)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/models/Qwen2.5-14B-Instruct",
        help="Path to the model (default: /home/models/Qwen2.5-14B-Instruct)"
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["pc", "no-pc"],
        default="pc",
        help="Test mode: 'pc' for prefix caching, 'no-pc' for baseline (default: pc)"
    )
    parser.add_argument(
        "--doc-len-list",
        type=int,
        nargs='+',
        help="List of document lengths to test (space-separated). If provided, overrides --doc-len"
    )
    parser.add_argument(
        "--concurrency-list",
        type=int,
        nargs='+',
        help="List of concurrency levels to test (space-separated). If provided, overrides --max-concurrency"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each test combination (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=50.0,
        help="Request timeout in seconds (default: 300.0)"
    )
    args = parser.parse_args() 

    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="sk-dummy")

    doc_len_list = args.doc_len_list if args.doc_len_list else [args.doc_len]
    concurrency_list = args.concurrency_list if args.concurrency_list else [args.max_concurrency]

    # Prepare CSV for results
    csv_path = "prefix_cache_benchmark.csv"
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["测试时间", "model", "输入长度", "输出长度", "并发请求数",
                             "测试模式", "命中率", "裸推TTFT(ms)", "写入TTFT(ms)", "命中TTFT(ms)"])
        
        # Warmup phase
        logger.info(">> Starting warmup phase (2 random requests)")
        warmup_prompts = build_warmup_docs(seed=args.seed)
        warmup_results = await run_round(client, args.model, warmup_prompts, args.max_concurrency, args.output_len)
        
        # Check warmup success rate
        successful_warmup = [r for r in warmup_results if r.get("error") is None]
        logger.info(f"Warmup completed: {len(successful_warmup)}/{len(warmup_results)} successful")
        
        await asyncio.sleep(1)
        warmup_results = await run_round(client, args.model, warmup_prompts, args.max_concurrency, args.output_len)
        
        successful_warmup = [r for r in warmup_results if r.get("error") is None]
        logger.info(f"Second warmup completed: {len(successful_warmup)}/{len(warmup_results)} successful")
        #await asyncio.sleep(5)
        
        # Main benchmark
        total_combinations = len(concurrency_list) * len(doc_len_list)
        current_combination = 0
        
        for concurrency_doc_nums in concurrency_list:
            for doc_len in doc_len_list:
                current_combination += 1
                combo_seed = int(f"{args.seed}{concurrency_doc_nums:02d}{doc_len:05d}")
                
                logger.info(
                    f"\n================ Test combination {current_combination}/{total_combinations}: "
                    f"doc_len={doc_len}, concurrency={concurrency_doc_nums} ================="
                )
                ret = await run_prefix_caching_cycle(
                    client=client,
                    model=args.model,
                    doc_len=doc_len,
                    num_docs=concurrency_doc_nums * args.repeat,
                    max_concurrency=concurrency_doc_nums,
                    hit_ratio=args.hit_ratio,
                    seed=combo_seed,
                    out_len=args.output_len,
                    test_mode=args.test_mode
                )
                if args.test_mode == "no-pc":
                    logger.info(
                        f"\n{'='*70}\n"
                        f"Combination {current_combination}/{total_combinations} finished (baseline)\n"
                        f"Params: doc_len={doc_len}, concurrency={concurrency_doc_nums}\n"
                        f"Result: raw_ttft={ret['write_ttft']:.2f}ms\n"
                        f"{'='*70}"
                    )
                else:
                    logger.info(
                        f"\n{'='*70}\n"
                        f"Combination {current_combination}/{total_combinations} finished (prefix caching)\n"
                        f"Params: doc_len={doc_len}, concurrency={concurrency_doc_nums}, hit_ratio={args.hit_ratio:.0%}\n"
                        f"Results: write_ttft={ret['write_ttft']:.2f}ms, read_ttft={ret['query_ttft']:.2f}ms\n"
                        f"{'='*70}"
                    )
                # Write results to CSV
                row = [
                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    args.model,
                    doc_len,
                    args.output_len,
                    concurrency_doc_nums,
                    args.test_mode,
                    f"{args.hit_ratio:.0%}" if args.test_mode == "pc" else "N/A"
                ]
                if args.test_mode == "no-pc":
                    row.extend([f"{ret['write_ttft']:.2f}", "", ""])
                else:
                    row.extend(["", f"{ret['write_ttft']:.2f}", f"{ret['query_ttft']:.2f}"])
                writer.writerow(row)
                f.flush()          
                await asyncio.sleep(2)

    logger.info(f"Benchmark completed. Results saved to: {os.path.abspath(csv_path)}")
if __name__ == "__main__":
    asyncio.run(main())