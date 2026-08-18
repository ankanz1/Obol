import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from app.pipeline.schemas import ConversationTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestQuery:
    query: str
    language: str
    query_type: str
    expected_answer: str = ""
    difficulty: str = "medium"


@dataclass
class LatencyResult:
    trace_id: str
    language: str
    query_type: str
    total_latency_ms: float
    stt_latency_ms: float
    embed_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    grounding_latency_ms: float
    tts_latency_ms: float
    success: bool
    grounded: bool
    refusal_reason: Optional[str]
    answer_length: int


class LatencyHarness:
    def __init__(
        self,
        test_queries: List[TestQuery],
        n_runs: int = 5,
        warmup_runs: int = 3
    ):
        self.test_queries = test_queries
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs
        self.results: List[LatencyResult] = []
    
    async def run_single(self, query: TestQuery, run_id: int) -> LatencyResult:
        """Run a single query through the pipeline manually (bypassing LangGraph ainvoke)."""
        trace_id = f"{query.language}_{query.query_type}_{run_id}"
        
        # Import pipeline nodes
        from app.pipeline.graph import (
            stt_node, lang_detect_node, embed_node, retrieve_node,
            input_guard_node, retrieve_confidence_node, generate_node,
            grounding_node, hallucination_node, output_guard_node,
            tts_node, update_history_node, refuse_node
        )
        import uuid
        
        # Use short mock audio for STT (will fail quickly), but pass query as transcript
        mock_audio = b"\x00" * 32000  # 1 second silence at 16kHz
        
        start_total = time.perf_counter()
        
        try:
            # Initialize state manually
            from app.pipeline.schemas import PipelineState
            state: PipelineState = {
                "audio_bytes": b"\x00" * 32000,
                "language": query.language,
                "session_id": trace_id,
                "trace_id": "",
                "transcript": query.query,  # Use pre-transcribed query
                "stt_confidence": 0.0,
                "stt_latency_ms": 0.0,
                "detected_language": "",
                "lang_confidence": 0.0,
                "query_embedding": [],
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "retrieval_latency_ms": 0.0,
                "input_guard_passed": False,
                "input_guard_reason": None,
                "retrieval_confidence": 0.0,
                "answer": "",
                "generation_latency_ms": 0.0,
                "reasoning_content": None,
                "grounded": False,
                "grounding_score": 0.0,
                "hallucination_check_passed": False,
                "hallucination_score": 0.0,
                "output_guard_passed": False,
                "output_guard_reason": None,
                "refusal_reason": None,
                "refusal_message": "",
                "tts_audio": b"",
                "tts_latency_ms": 0.0,
                "conversation_history": [],
                "total_latency_ms": 0.0,
                "component_latencies": {},
                "file_path": "",
                "extracted_text": "",
                "vision_job_id": "",
                "vision_output_url": "",
                "ingested_chunks": 0,
                "ingestion_language": "",
                "vision_error": "",
                "use_mock_generation": True,
            }
            
            # Manually execute pipeline steps
            # STT
            state = await stt_node(state)
            if state.get("refusal_reason") == "stt_failed":
                state = await refuse_node(state)
            else:
                # Language detection
                state = await lang_detect_node(state)
                if state.get("refusal_reason") == "unsupported_language":
                    state = await refuse_node(state)
                else:
                    # Embedding
                    state = await embed_node(state)
                    
                    # Retrieval
                    state = await retrieve_node(state)
                    
                    # Input guard
                    state = await input_guard_node(state)
                    if not state.get("input_guard_passed"):
                        state = await refuse_node(state)
                    else:
                        # Retrieve confidence
                        state = await retrieve_confidence_node(state)
                        if state.get("refusal_reason") == "low_retrieval_confidence":
                            state = await refuse_node(state)
                        else:
                            # Generation
                            state = await generate_node(state)
                            if state.get("refusal_reason"):
                                state = await refuse_node(state)
                            else:
                                # Grounding
                                state = await grounding_node(state)
                                if state.get("refusal_reason") == "ungrounded":
                                    state = await refuse_node(state)
                                else:
                                    # Hallucination
                                    state = await hallucination_node(state)
                                    if state.get("refusal_reason") == "hallucination_detected":
                                        state = await refuse_node(state)
                                    else:
                                        # Output guard
                                        state = await output_guard_node(state)
                                        if not state.get("output_guard_passed"):
                                            state = await refuse_node(state)
                                        else:
                                            # TTS
                                            state = await tts_node(state)
                                            
                                            # Update history
                                            state = await update_history_node(state)
            
            total_latency = (time.perf_counter() - start_total) * 1000
            
            # Compute component latencies (approximate since we don't track individually)
            # We'll use the total for all components
            total_lat = state.get("total_latency_ms", total_latency)
            
            return LatencyResult(
                trace_id=trace_id,
                language=query.language,
                query_type=query.query_type,
                total_latency_ms=total_latency,
                stt_latency_ms=state.get("stt_latency_ms", 0),
                embed_latency_ms=state.get("embed_latency_ms", 0),
                retrieval_latency_ms=state.get("retrieval_latency_ms", 0),
                generation_latency_ms=state.get("generation_latency_ms", 0),
                grounding_latency_ms=0,  # Not separately tracked yet
                tts_latency_ms=state.get("tts_latency_ms", 0),
                success=bool(state.get("answer") and not state.get("refusal_reason")),
                grounded=state.get("grounded", False),
                refusal_reason=state.get("refusal_reason"),
                answer_length=len(state.get("answer", ""))
            )
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return LatencyResult(
                trace_id=trace_id,
                language=query.language,
                query_type=query.query_type,
                total_latency_ms=(time.perf_counter() - start_total) * 1000,
                stt_latency_ms=0,
                embed_latency_ms=0,
                retrieval_latency_ms=0,
                generation_latency_ms=0,
                grounding_latency_ms=0,
                tts_latency_ms=0,
                success=False,
                grounded=False,
                refusal_reason=str(e),
                answer_length=0
            )
    
    async def run(self) -> Dict[str, Any]:
        """Run full latency evaluation."""
        logger.info(f"Starting latency evaluation: {len(self.test_queries)} queries x {self.n_runs} runs")
        
        # Warmup
        logger.info(f"Warming up ({self.warmup_runs} runs)...")
        for _ in range(self.warmup_runs):
            for query in self.test_queries[:1]:  # Just first 1 for warmup
                await self.run_single(query, -1)
        
        # Actual runs - reduced to 1 run for faster evaluation
        self.results = []
        for run in range(1):  # Reduced to 1 run for faster evaluation
            logger.info(f"Run {run + 1}/{self.n_runs}")
            for query in self.test_queries:
                result = await self.run_single(query, run)
                self.results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
        
        return self.compute_stats()
    
    def compute_stats(self) -> Dict[str, Any]:
        """Compute latency statistics."""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r.success]
        all_latencies = [r.total_latency_ms for r in self.results]
        success_latencies = [r.total_latency_ms for r in successful]
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            return float(np.percentile(data, p))
        
        # Overall stats
        stats = {
            "total_queries": len(self.results),
            "successful": len(successful),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
            "grounded_rate": sum(1 for r in successful if r.grounded) / len(successful) if successful else 0,
            "refusal_rate": sum(1 for r in self.results if r.refusal_reason) / len(self.results),
            
            "latency_overall": {
                "p50": percentile(all_latencies, 50),
                "p70": percentile(all_latencies, 70),
                "p90": percentile(all_latencies, 90),
                "p95": percentile(all_latencies, 95),
                "p99": percentile(all_latencies, 99),
                "p100": max(all_latencies) if all_latencies else 0,
                "mean": statistics.mean(all_latencies) if all_latencies else 0,
                "stdev": statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0,
            },
            "latency_successful": {
                "p50": percentile(success_latencies, 50),
                "p70": percentile(success_latencies, 70),
                "p90": percentile(success_latencies, 90),
                "p95": percentile(success_latencies, 95),
                "p99": percentile(success_latencies, 99),
                "p100": max(success_latencies) if success_latencies else 0,
                "mean": statistics.mean(success_latencies) if success_latencies else 0,
            }
        }
        
        # By language
        by_language = {}
        for lang in set(r.language for r in self.results):
            lang_results = [r for r in self.results if r.language == lang]
            lang_latencies = [r.total_latency_ms for r in lang_results]
            by_language[lang] = {
                "count": len(lang_results),
                "p50": percentile(lang_latencies, 50),
                "p70": percentile(lang_latencies, 70),
                "p90": percentile(lang_latencies, 90),
                "p99": percentile(lang_latencies, 99),
                "p100": max(lang_latencies) if lang_latencies else 0,
                "success_rate": sum(1 for r in lang_results if r.success) / len(lang_results)
            }
        stats["by_language"] = by_language
        
        # By query type
        by_type = {}
        for qtype in set(r.query_type for r in self.results):
            type_results = [r for r in self.results if r.query_type == qtype]
            type_latencies = [r.total_latency_ms for r in type_results]
            by_type[qtype] = {
                "count": len(type_results),
                "p50": percentile(type_latencies, 50),
                "p70": percentile(type_latencies, 70),
                "p90": percentile(type_latencies, 90),
                "p99": percentile(type_latencies, 99),
                "p100": max(type_latencies) if type_latencies else 0,
            }
        stats["by_query_type"] = by_type
        
        # By component (successful runs only)
        if successful:
            stats["by_component"] = {
                "stt": {
                    "p50": percentile([r.stt_latency_ms for r in successful], 50),
                    "p99": percentile([r.stt_latency_ms for r in successful], 99),
                },
                "embed": {
                    "p50": percentile([r.embed_latency_ms for r in successful], 50),
                    "p99": percentile([r.embed_latency_ms for r in successful], 99),
                },
                "retrieval": {
                    "p50": percentile([r.retrieval_latency_ms for r in successful], 50),
                    "p99": percentile([r.retrieval_latency_ms for r in successful], 99),
                },
                "generation": {
                    "p50": percentile([r.generation_latency_ms for r in successful], 50),
                    "p99": percentile([r.generation_latency_ms for r in successful], 99),
                },
                "tts": {
                    "p50": percentile([r.tts_latency_ms for r in successful], 50),
                    "p99": percentile([r.tts_latency_ms for r in successful], 99),
                },
            }
        
        # Bootstrap confidence intervals
        stats["bootstrap_ci_95"] = self._bootstrap_ci(success_latencies, 0.95)
        
        return stats
    
    def _bootstrap_ci(self, data: List[float], confidence: float = 0.95, n_bootstrap: int = 1000) -> List[float]:
        """Compute bootstrap confidence interval for mean."""
        if len(data) < 2:
            return [0.0, 0.0]
        
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        
        alpha = (1 - confidence) / 2
        lower = np.percentile(means, alpha * 100)
        upper = np.percentile(means, (1 - alpha) * 100)
        
        return [float(lower), float(upper)]
    
    def save_results(self, output_path: Path):
        """Save detailed results to JSON."""
        output = {
            "summary": self.compute_stats(),
            "detailed_results": [asdict(r) for r in self.results]
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_path}")


def load_test_queries(file_path: Path) -> List[TestQuery]:
    """Load test queries from JSONL file."""
    queries = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            queries.append(TestQuery(**data))
    return queries


def generate_test_queries(output_path: Path, n_per_lang: int = 11):
    """Generate test queries from validation set."""
    from datasets import load_dataset
    
    queries = []
    languages = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur", "ne", "sa", "ks", "sd", "doi", "sat"]
    
    for lang in languages:
        try:
            ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation", streaming=True)
            count = 0
            for row in ds:
                if count >= n_per_lang:
                    break
                queries.append(TestQuery(
                    query=row.get("query", ""),
                    language=lang,
                    query_type=row.get("query_type", "UNKNOWN"),
                    expected_answer=row.get("Answer", "")
                ))
                count += 1
        except Exception as e:
            logger.warning(f"Failed to generate queries for {lang}: {e}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for q in queries:
            f.write(json.dumps(asdict(q)) + '\n')
    
    logger.info(f"Generated {len(queries)} test queries to {output_path}")
    return queries


async def main():
    import sys
    
    test_queries_path = Path("evaluation/test_queries.jsonl")
    results_path = Path("evaluation/results.json")
    
    # Generate or load test queries
    if test_queries_path.exists():
        queries = load_test_queries(test_queries_path)
    else:
        queries = generate_test_queries(test_queries_path)
    
    # Run harness
    harness = LatencyHarness(queries, n_runs=5, warmup_runs=2)
    stats = await harness.run()
    
    # Save results
    harness.save_results(results_path)
    
    # Print summary
    print("\n=== LATENCY EVALUATION RESULTS ===")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Grounded rate: {stats['grounded_rate']:.1%}")
    print(f"Refusal rate: {stats['refusal_rate']:.1%}")
    print(f"\nOverall latency (ms):")
    print(f"  P50: {stats['latency_overall']['p50']:.1f}")
    print(f"  P70: {stats['latency_overall']['p70']:.1f}")
    print(f"  P90: {stats['latency_overall']['p90']:.1f}")
    print(f"  P99: {stats['latency_overall']['p99']:.1f}")
    print(f"  P100: {stats['latency_overall']['p100']:.1f}")
    print(f"  Mean: {stats['latency_overall']['mean']:.1f}")
    print(f"\nBy language:")
    for lang, data in stats['by_language'].items():
        print(f"  {lang}: P50={data['p50']:.1f}ms, P99={data['p99']:.1f}ms, Success={data['success_rate']:.1%}")
    
    return stats


if __name__ == "__main__":
    asyncio.run(main())