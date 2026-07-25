"""
Optional LLM-powered deep-dive pass for the cybersecurity review agent.

Takes the static findings from vuln_scanner.py plus the source code and asks
Claude to reason like a defensive security assessor: validate/prioritize the
findings, describe conceptual exploitation scenarios, flag business-logic or
design issues pattern rules can't catch, and produce a remediation plan.

This performs source-code review only -- it never sends network requests to
the target application and produces no working exploit code against live
systems. It is intended for reviewing code the user owns or is authorized to
assess.
"""

from typing import List

from vuln_scanner import Finding, ScanResult

SYSTEM_PROMPT = """\
You are a senior application security engineer performing an authorized, \
defensive static code review for the person who submitted this code (they \
own it or are otherwise authorized to review it). You are NOT interacting \
with any live system, network, or third party -- you only see the source \
text and a list of findings from an automated scanner.

Your job:
1. Briefly validate or push back on the automated findings (note any that \
look like false positives given the surrounding code).
2. Identify additional vulnerabilities or risky design/business-logic \
patterns the automated scanner would miss (e.g. broken access control, \
missing rate limiting, IDOR, race conditions, trust boundary issues, secrets \
handling, unsafe defaults).
3. For each significant issue, describe the exploitation scenario \
conceptually (attacker preconditions, what they'd achieve) -- do not produce \
ready-to-run exploit payloads or attack scripts targeting real infrastructure.
4. Provide a prioritized remediation plan (quick wins first, ordered by \
risk reduction per effort).

Format your response in Markdown with clear headings. Be concise and concrete \
-- reference specific file/line locations from the findings or code where \
possible. If the code is small and looks intentionally benign (e.g. a demo \
script with no untrusted input), say so plainly instead of inventing risk.
"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for length] ..."


def build_user_prompt(result: ScanResult, code_samples: List[tuple], max_code_chars: int = 40_000) -> str:
    parts = ["## Automated static findings\n"]
    if not result.findings:
        parts.append("(No pattern-based findings were triggered.)\n")
    for f in result.sorted_findings():
        parts.append(
            f"- **[{f.severity.value}] {f.title}** ({f.cwe}) at `{f.file}:{f.line}` -- "
            f"`{f.snippet}`"
        )

    parts.append("\n## Source code\n")
    budget = max_code_chars
    for name, content in code_samples:
        if budget <= 0:
            parts.append(f"\n(Remaining files omitted for length: {name}, ...)")
            break
        chunk = _truncate(content, min(budget, 8_000))
        budget -= len(chunk)
        parts.append(f"\n### `{name}`\n```\n{chunk}\n```")

    return "\n".join(parts)


def run_llm_review(api_key: str, result: ScanResult, code_samples: List[tuple], model: str = "claude-sonnet-5") -> str:
    """code_samples: list of (filename, source_text) tuples to include as context."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(result, code_samples)

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
