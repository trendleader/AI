"""
ShopEase RAG Benchmark – Streamlit App
Evaluates retrieval quality and generation quality of the ShopEase agentic RAG pipeline.
"""

from __future__ import annotations

import math
import time

import anthropic
import streamlit as st

from benchmark_data import ANTHROPIC_PRICING, GROUND_TRUTH
from knowledge_base import build_knowledge_base, search_knowledge_base_raw
from agent import run_agent

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ShopEase RAG Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL = "claude-sonnet-4-6"
PRICING = ANTHROPIC_PRICING[MODEL]

# ---------------------------------------------------------------------------
# Knowledge base – cached
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_collection():
    return build_knowledge_base()

collection = get_collection()

# ---------------------------------------------------------------------------
# Retrieval metric helpers
# ---------------------------------------------------------------------------

def hit_rate_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """1.0 if any expected doc appears in the top-k retrieved results, else 0.0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(eid in top_k for eid in expected_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1/rank of the first retrieved doc that matches any expected doc."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k.
    Relevance is binary: 1 if doc is in expected_ids, else 0.
    """
    def dcg(ids: list[str]) -> float:
        score = 0.0
        for i, doc_id in enumerate(ids[:k], start=1):
            rel = 1 if doc_id in expected_ids else 0
            score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_ids)
    # Ideal: put all relevant docs first
    ideal_order = [eid for eid in expected_ids if eid in expected_ids][:k]
    # Pad with non-relevant if needed
    ideal_dcg = dcg(ideal_order + [""] * k)
    if ideal_dcg == 0:
        return 1.0  # all queries should have at least one expected doc
    return min(actual_dcg / ideal_dcg, 1.0)


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
_judge_client = anthropic.Anthropic()

def llm_judge(query: str, answer: str, expected_topics: list[str]) -> dict:
    """
    Use claude-sonnet-4-6 to score faithfulness and answer relevancy.
    Returns {"faithfulness": float, "answer_relevancy": float, "reasoning": str,
             "input_tokens": int, "output_tokens": int}.
    """
    topics_str = ", ".join(expected_topics)
    prompt = f"""You are an expert evaluator for a customer support RAG system.

Query: {query}

Agent Answer:
{answer}

Expected topics the answer should cover: {topics_str}

Rate the answer on two dimensions (0.0 to 1.0):

1. **faithfulness**: Does the answer contain only accurate information consistent with standard e-commerce policies? Penalise hallucinations or invented facts.
2. **answer_relevancy**: Does the answer address what the customer actually asked, covering the expected topics?

Respond ONLY with valid JSON in exactly this format (no markdown fences):
{{"faithfulness": 0.00, "answer_relevancy": 0.00, "reasoning": "one sentence"}}"""

    resp = _judge_client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = resp.content[0].text.strip()
    input_tokens = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens

    try:
        import json
        data = json.loads(raw)
        return {
            "faithfulness": float(data.get("faithfulness", 0.5)),
            "answer_relevancy": float(data.get("answer_relevancy", 0.5)),
            "reasoning": data.get("reasoning", ""),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    except Exception:
        return {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "reasoning": "Parse error – defaulted to 0.5",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }


# ---------------------------------------------------------------------------
# Token estimation for run_agent (no token fields on AgentResponse)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)


def _compute_cost(input_tok: int, output_tok: int) -> float:
    return (
        input_tok / 1_000_000 * PRICING["input_per_mtok"]
        + output_tok / 1_000_000 * PRICING["output_per_mtok"]
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_queries: int, k: int, progress_bar) -> list[dict]:
    """Run the benchmark for n_queries items. Returns list of result dicts."""
    subset = GROUND_TRUTH[:n_queries]
    results = []

    for i, item in enumerate(subset):
        progress_bar.progress((i) / n_queries, text=f"Running query {i + 1}/{n_queries}…")
        t0 = time.perf_counter()

        # ── Retrieval ─────────────────────────────────────────────────────
        raw_results = search_knowledge_base_raw(collection, item["query"], n_results=k)
        retrieved_ids = [r["doc_id"] for r in raw_results]

        hr = hit_rate_at_k(retrieved_ids, item["expected_doc_ids"], k)
        rr = reciprocal_rank(retrieved_ids, item["expected_doc_ids"])
        ndcg = ndcg_at_k(retrieved_ids, item["expected_doc_ids"], k)

        # ── Generation ────────────────────────────────────────────────────
        agent_resp = run_agent(
            user_message=item["query"],
            conversation_history=[],
            collection=collection,
        )
        answer = agent_resp.answer
        escalated = agent_resp.escalated

        latency_ms = (time.perf_counter() - t0) * 1000

        # Token estimation for agent call (no usage object on AgentResponse)
        est_input_tok = _estimate_tokens(item["query"]) + 800   # system + tools overhead
        est_output_tok = _estimate_tokens(answer)
        agent_cost = _compute_cost(est_input_tok, est_output_tok)

        # ── LLM Judge ─────────────────────────────────────────────────────
        judge = llm_judge(item["query"], answer, item["expected_answer_topics"])
        judge_cost = _compute_cost(judge["input_tokens"], judge["output_tokens"])

        results.append({
            "query": item["query"],
            "category": item["category"],
            "expected_doc_ids": item["expected_doc_ids"],
            "retrieved_ids": retrieved_ids,
            "hit_rate": hr,
            "mrr": rr,
            "ndcg": ndcg,
            "answer": answer,
            "faithfulness": judge["faithfulness"],
            "answer_relevancy": judge["answer_relevancy"],
            "judge_reasoning": judge["reasoning"],
            "latency_ms": latency_ms,
            "agent_cost_usd": agent_cost,
            "judge_cost_usd": judge_cost,
            "total_cost_usd": agent_cost + judge_cost,
            "escalated": escalated,
            "should_escalate": item["should_escalate"],
            "escalation_correct": escalated == item["should_escalate"],
        })

    progress_bar.progress(1.0, text="Benchmark complete.")
    return results


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(math.ceil(p / 100 * len(sorted_vals))) - 1)
    return sorted_vals[idx]


def _health_badge(hit_rate: float, mrr: float, faith: float) -> str:
    score = (hit_rate + mrr + faith) / 3
    if score >= 0.75:
        return "Excellent"
    elif score >= 0.55:
        return "Good"
    else:
        return "Needs Work"


def _metric_color(value: float, threshold_good: float, threshold_ok: float) -> str:
    if value >= threshold_good:
        return "normal"
    elif value >= threshold_ok:
        return "off"
    return "inverse"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "benchmark_results" not in st.session_state:
    st.session_state["benchmark_results"] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ ShopEase RAG Benchmark")
    st.divider()

    n_queries = st.slider("Number of queries", min_value=5, max_value=20, value=10, step=5)
    k_val = st.select_slider("Retrieval K", options=[1, 3, 5], value=3)
    st.divider()

    csat_input = st.slider("CSAT score (out of 5)", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
    nps_input = st.slider("NPS score", min_value=-100, max_value=100, value=42)
    st.divider()

    run_btn = st.button("Run Benchmark", type="primary", use_container_width=True)
    progress_bar = st.progress(0, text="")

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
if run_btn:
    st.session_state["benchmark_results"] = run_benchmark(n_queries, k_val, progress_bar)

results = st.session_state["benchmark_results"]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview Dashboard",
    "Retrieval Metrics",
    "Generation Quality",
    "Latency & Cost",
    "Business Metrics",
])

# ===========================================================================
# TAB 1: Overview Dashboard
# ===========================================================================
with tab1:
    st.header("Overview Dashboard")

    if results is None:
        st.info("Run the benchmark from the sidebar to see results.")
    else:
        hit_rates = [r["hit_rate"] for r in results]
        mrrs = [r["mrr"] for r in results]
        ndcgs = [r["ndcg"] for r in results]
        faiths = [r["faithfulness"] for r in results]
        relevancies = [r["answer_relevancy"] for r in results]
        latencies = sorted([r["latency_ms"] for r in results])
        costs = [r["total_cost_usd"] for r in results]

        avg_hr = _avg(hit_rates)
        avg_mrr = _avg(mrrs)
        avg_ndcg = _avg(ndcgs)
        avg_faith = _avg(faiths)
        avg_rel = _avg(relevancies)
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg_cost = _avg(costs)
        monthly_proj = avg_cost * 30 * 1000  # ~1000 queries/day

        escalation_correct = [r for r in results if r["escalation_correct"]]
        resolution_rate = len(escalation_correct) / len(results)

        badge = _health_badge(avg_hr, avg_mrr, avg_faith)
        badge_color = {"Excellent": "green", "Good": "orange", "Needs Work": "red"}[badge]
        st.markdown(f"### Benchmark Health: :{badge_color}[**{badge}**]")

        st.subheader("Retrieval")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Hit Rate@{k_val}", f"{avg_hr:.3f}",
                    delta="Good" if avg_hr >= 0.7 else "Low")
        col2.metric("MRR", f"{avg_mrr:.3f}",
                    delta="Good" if avg_mrr >= 0.5 else "Low")
        col3.metric(f"NDCG@{k_val}", f"{avg_ndcg:.3f}")

        st.subheader("Generation Quality")
        col4, col5 = st.columns(2)
        col4.metric("Faithfulness", f"{avg_faith:.3f}",
                    delta="Good" if avg_faith >= 0.8 else "Low")
        col5.metric("Answer Relevancy", f"{avg_rel:.3f}",
                    delta="Good" if avg_rel >= 0.8 else "Low")

        st.subheader("Latency")
        col6, col7, col8 = st.columns(3)
        col6.metric("P50 Latency", f"{p50:.0f} ms")
        col7.metric("P95 Latency", f"{p95:.0f} ms")
        col8.metric("P99 Latency", f"{p99:.0f} ms")

        st.subheader("Cost & Business")
        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Avg Cost/Query", f"${avg_cost:.5f}")
        col10.metric("Monthly Projection", f"${monthly_proj:.2f}")
        col11.metric("Resolution Rate", f"{resolution_rate:.1%}")
        col12.metric("CSAT", f"{csat_input:.1f}/5.0")

# ===========================================================================
# TAB 2: Retrieval Metrics
# ===========================================================================
with tab2:
    st.header("Retrieval Metrics")

    if results is None:
        st.info("Run the benchmark to see retrieval metrics.")
    else:
        st.info(
            f"**Hit Rate@{k_val}** measures whether any ground-truth document appears in the top-{k_val} results.  "
            f"**MRR** (Mean Reciprocal Rank) scores how highly the first relevant result is ranked.  "
            f"**NDCG@{k_val}** (Normalised Discounted Cumulative Gain) rewards relevant results appearing earlier."
        )

        # Per-query chart data
        chart_data = {
            f"Hit Rate@{k_val}": [r["hit_rate"] for r in results],
            "MRR": [r["mrr"] for r in results],
            f"NDCG@{k_val}": [r["ndcg"] for r in results],
        }
        st.subheader("Per-Query Scores")
        st.bar_chart(chart_data)

        # Detailed table
        st.subheader("Per-Query Detail")
        table_rows = []
        for i, r in enumerate(results):
            table_rows.append({
                "#": i + 1,
                "Category": r["category"],
                "Query (truncated)": r["query"][:60] + "…",
                f"Hit@{k_val}": round(r["hit_rate"], 3),
                "MRR": round(r["mrr"], 3),
                f"NDCG@{k_val}": round(r["ndcg"], 3),
                "Retrieved": ", ".join(r["retrieved_ids"]),
                "Expected": ", ".join(r["expected_doc_ids"]),
            })
        st.dataframe(table_rows, use_container_width=True)

        # Overall stats
        st.subheader("Overall Statistics")
        hit_rates = [r["hit_rate"] for r in results]
        mrrs = [r["mrr"] for r in results]
        ndcgs = [r["ndcg"] for r in results]

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric(f"Mean Hit Rate@{k_val}", f"{_avg(hit_rates):.3f}")
        oc1.metric(f"Min Hit Rate@{k_val}", f"{min(hit_rates):.3f}")
        oc2.metric("Mean MRR", f"{_avg(mrrs):.3f}")
        oc2.metric("Min MRR", f"{min(mrrs):.3f}")
        oc3.metric(f"Mean NDCG@{k_val}", f"{_avg(ndcgs):.3f}")
        oc3.metric(f"Min NDCG@{k_val}", f"{min(ndcgs):.3f}")

# ===========================================================================
# TAB 3: Generation Quality
# ===========================================================================
with tab3:
    st.header("Generation Quality")

    if results is None:
        st.info("Run the benchmark to see generation quality metrics.")
    else:
        faiths = [r["faithfulness"] for r in results]
        relevancies = [r["answer_relevancy"] for r in results]

        gc1, gc2 = st.columns(2)
        gc1.metric("Avg Faithfulness", f"{_avg(faiths):.3f}",
                   delta="Above threshold" if _avg(faiths) >= 0.8 else "Below threshold")
        gc2.metric("Avg Answer Relevancy", f"{_avg(relevancies):.3f}",
                   delta="Above threshold" if _avg(relevancies) >= 0.8 else "Below threshold")

        # Styled scores table
        st.subheader("LLM Judge Scores")
        judge_rows = []
        for i, r in enumerate(results):
            faith_flag = "✅" if r["faithfulness"] >= 0.8 else ("⚠️" if r["faithfulness"] >= 0.6 else "❌")
            rel_flag = "✅" if r["answer_relevancy"] >= 0.8 else ("⚠️" if r["answer_relevancy"] >= 0.6 else "❌")
            judge_rows.append({
                "#": i + 1,
                "Category": r["category"],
                "Query": r["query"][:55] + "…",
                "Faithfulness": f"{r['faithfulness']:.2f} {faith_flag}",
                "Relevancy": f"{r['answer_relevancy']:.2f} {rel_flag}",
                "Reasoning": r["judge_reasoning"][:80] + "…" if len(r["judge_reasoning"]) > 80 else r["judge_reasoning"],
            })
        st.dataframe(judge_rows, use_container_width=True)

        # Per-query chart
        gen_chart = {
            "Faithfulness": faiths,
            "Answer Relevancy": relevancies,
        }
        st.bar_chart(gen_chart)

        # Expanders for full answers
        st.subheader("Full Answers")
        for i, r in enumerate(results):
            label = f"Query {i+1}: {r['query'][:60]}…"
            with st.expander(label, expanded=False):
                st.markdown(f"**Category:** {r['category']}")
                st.markdown(f"**Faithfulness:** {r['faithfulness']:.2f}  |  **Relevancy:** {r['answer_relevancy']:.2f}")
                st.markdown(f"**Judge reasoning:** {r['judge_reasoning']}")
                st.divider()
                st.markdown(r["answer"])

        # Escalation accuracy
        st.subheader("Escalation Accuracy")
        esc_results = [r for r in results if r["should_escalate"]]
        if esc_results:
            correct_esc = sum(1 for r in esc_results if r["escalation_correct"])
            esc_acc = correct_esc / len(esc_results)
            st.metric("Escalation Accuracy", f"{esc_acc:.1%}",
                      help="Fraction of should-escalate queries where the agent correctly escalated.")
        else:
            st.caption("No escalation queries in this subset.")

# ===========================================================================
# TAB 4: Latency & Cost
# ===========================================================================
with tab4:
    st.header("Latency & Cost")

    if results is None:
        st.info("Run the benchmark to see latency and cost data.")
    else:
        latencies = sorted([r["latency_ms"] for r in results])
        agent_costs = [r["agent_cost_usd"] for r in results]
        judge_costs = [r["judge_cost_usd"] for r in results]
        total_costs = [r["total_cost_usd"] for r in results]

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

        # Latency percentiles
        st.subheader("Latency Percentiles")
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("P50", f"{p50:.0f} ms")
        lc2.metric("P95", f"{p95:.0f} ms")
        lc3.metric("P99", f"{p99:.0f} ms")

        # Latency histogram (approximate via bar chart)
        st.subheader("Latency Distribution (ms)")
        lat_vals = [r["latency_ms"] for r in results]
        # Build bucket counts
        if lat_vals:
            min_l, max_l = min(lat_vals), max(lat_vals)
            n_buckets = min(10, len(lat_vals))
            bucket_size = max((max_l - min_l) / n_buckets, 1)
            buckets: dict[str, int] = {}
            for v in lat_vals:
                bucket_label = f"{int((v - min_l) // bucket_size * bucket_size + min_l)}"
                buckets[bucket_label] = buckets.get(bucket_label, 0) + 1
            st.bar_chart(buckets)

        # Per-query latency
        st.subheader("Per-Query Latency (ms)")
        st.bar_chart({"Latency (ms)": lat_vals})

        # Cost breakdown table
        st.subheader("Cost Breakdown")
        cost_rows = []
        for i, r in enumerate(results):
            cost_rows.append({
                "#": i + 1,
                "Category": r["category"],
                "Query": r["query"][:50] + "…",
                "Agent Cost ($)": f"{r['agent_cost_usd']:.6f}",
                "Judge Cost ($)": f"{r['judge_cost_usd']:.6f}",
                "Total Cost ($)": f"{r['total_cost_usd']:.6f}",
                "Latency (ms)": f"{r['latency_ms']:.0f}",
            })
        st.dataframe(cost_rows, use_container_width=True)

        # Summary costs
        avg_cost = _avg(total_costs)
        total_run = sum(total_costs)
        monthly_proj = avg_cost * 30 * 1000

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Avg Cost / Query", f"${avg_cost:.5f}")
        cc2.metric("Total Benchmark Cost", f"${total_run:.4f}")
        cc3.metric("Monthly Projection (1k qpd)", f"${monthly_proj:.2f}")

        # Cost vs quality (as paired bar chart)
        st.subheader("Cost vs Faithfulness (per query)")
        cvq = {
            "Total Cost (×1000 $)": [r["total_cost_usd"] * 1000 for r in results],
            "Faithfulness": [r["faithfulness"] for r in results],
        }
        st.bar_chart(cvq)

# ===========================================================================
# TAB 5: Business Metrics
# ===========================================================================
with tab5:
    st.header("Business Metrics")

    if results is None:
        st.info("Run the benchmark to see business metrics.")
    else:
        total = len(results)
        correct_escalations = sum(1 for r in results if r["escalation_correct"])
        resolution_rate = correct_escalations / total

        # Resolution rate with progress bar
        st.subheader("Resolution Rate")
        st.progress(resolution_rate, text=f"{resolution_rate:.1%} queries handled correctly")

        # CSAT with star display
        st.subheader("Customer Satisfaction (CSAT)")
        stars = int(round(csat_input))
        star_display = "★" * stars + "☆" * (5 - stars)
        st.markdown(f"## {star_display}")
        st.metric("CSAT Score", f"{csat_input:.1f} / 5.0")

        # NPS zone
        st.subheader("Net Promoter Score (NPS)")
        if nps_input >= 50:
            nps_zone = "Excellent"
            nps_color = "green"
        elif nps_input >= 0:
            nps_zone = "Good"
            nps_color = "blue"
        elif nps_input >= -10:
            nps_zone = "Needs Improvement"
            nps_color = "orange"
        else:
            nps_zone = "Critical"
            nps_color = "red"

        st.metric("NPS", nps_input)
        st.markdown(f"**Zone:** :{nps_color}[{nps_zone}]")

        # Escalation log
        st.subheader("Escalation Log")
        should_esc = [r for r in results if r["should_escalate"]]
        did_esc = [r for r in results if r["escalated"]]
        false_pos = [r for r in results if r["escalated"] and not r["should_escalate"]]
        false_neg = [r for r in results if not r["escalated"] and r["should_escalate"]]

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Should Escalate", len(should_esc))
        ec2.metric("Did Escalate", len(did_esc))
        ec3.metric("False Positives", len(false_pos), delta=f"-{len(false_pos)}" if false_pos else None,
                   delta_color="inverse")
        ec4.metric("Missed Escalations", len(false_neg), delta=f"-{len(false_neg)}" if false_neg else None,
                   delta_color="inverse")

        if false_pos:
            with st.expander(f"False Positive Escalations ({len(false_pos)})", expanded=False):
                for r in false_pos:
                    st.markdown(f"- **{r['category']}**: {r['query'][:80]}")

        if false_neg:
            with st.expander(f"Missed Escalations ({len(false_neg)})", expanded=True):
                for r in false_neg:
                    st.markdown(f"- **{r['category']}**: {r['query'][:80]}")

        # Category breakdown
        st.subheader("Performance by Category")
        cats = {}
        for r in results:
            c = r["category"]
            if c not in cats:
                cats[c] = {"hit_rate": [], "faithfulness": [], "count": 0}
            cats[c]["hit_rate"].append(r["hit_rate"])
            cats[c]["faithfulness"].append(r["faithfulness"])
            cats[c]["count"] += 1

        cat_rows = []
        for cat, vals in cats.items():
            cat_rows.append({
                "Category": cat.title(),
                "Queries": vals["count"],
                f"Avg Hit Rate@{k_val}": f"{_avg(vals['hit_rate']):.3f}",
                "Avg Faithfulness": f"{_avg(vals['faithfulness']):.3f}",
            })
        st.dataframe(cat_rows, use_container_width=True)
