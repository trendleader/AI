# AI

## Cybersecurity Review Agent

`cybersecurity_agent_app.py` is a Streamlit app that statically scans uploaded
source code (single files or a `.zip` of a codebase) for common
vulnerabilities -- hardcoded secrets, SQL/command injection, insecure
deserialization, weak crypto, disabled TLS verification, path traversal,
XSS-prone rendering, JWT bypass, and more. Optionally, it sends the findings
and code to Claude for a deeper reasoning pass that explains conceptual
exploitation scenarios and produces a prioritized remediation plan.

Run it with:

```bash
streamlit run cybersecurity_agent_app.py
```

**Scope:** this is a static code-review tool. It never sends traffic to a
live system -- only upload/paste code you own or are authorized to assess.
Core logic lives in `vuln_scanner.py` (rule engine) and
`llm_security_review.py` (optional AI deep-dive).