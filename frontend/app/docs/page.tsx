/**
 * app/docs/page.tsx — API + Developer Documentation
 * ===================================================
 * Server Component — fully static.
 *
 * WHAT:
 *   Developer-facing reference page explaining how to use the DeepCoin
 *   API, what each endpoint returns, and how the classification pipeline
 *   works under the hood.
 *
 *   Covers:
 *     - POST /api/classify — the main inference endpoint
 *     - GET  /api/history  — paginated history
 *     - GET  /api/health   — liveness + readiness check
 *     - Authentication model (X-API-Key / dev passthrough)
 *     - Rate limits
 *     - Agent routing logic
 *     - PDF report download
 *
 * WHY Server Component:
 *   All content is static markdown-ish JSX — no state, no hooks.
 *   Renders as plain HTML, perfect for SEO and zero JS cost.
 */

import Link from "next/link";
import { ExternalLink, Cpu, Database, FileText, Shield, Clock,
         Zap, Code2, GitBranch } from "lucide-react";

/* ── static data ──────────────────────────────────────────────────────────── */

type Endpoint = {
  method: "GET" | "POST" | "DELETE";
  path:   string;
  auth:   boolean;
  desc:   string;
  body?:  { field: string; type: string; required: boolean; desc: string }[];
  resp?:  { field: string; type: string; desc: string }[];
  notes?: string;
};

const ENDPOINTS: Endpoint[] = [
  {
    method: "POST",
    path:   "/api/classify",
    auth:   false,
    desc:   "Upload a coin photograph and receive a full AI analysis. The response includes the CNN prediction, the agent route taken, and a URL to the generated PDF report.",
    body: [
      { field: "file",    type: "multipart/file", required: true,  desc: "JPEG / PNG coin image. Max 10 MB. Auto-cropped and CLAHE-enhanced before inference." },
      { field: "use_tta", type: "boolean",         required: false, desc: "Enable Test-Time Augmentation (8 passes). Improves accuracy by ~0.8% at the cost of ~4× latency. Default: false." },
    ],
    resp: [
      { field: "id",           type: "string",           desc: "Unique analysis ID (UUID4)." },
      { field: "cnn",          type: "CnnResult",         desc: "Top-5 predictions, confidence score, TTA metadata." },
      { field: "route_taken",  type: "string",           desc: "Agent route: historian | validator | investigator." },
      { field: "historian",    type: "object | null",    desc: "Historical narrative, KB metadata, Gemini citation." },
      { field: "validator",    type: "object | null",    desc: "HSV forensic material check (validator route only)." },
      { field: "investigator", type: "object | null",    desc: "VLM / OpenCV visual attributes (investigator route only)." },
      { field: "report",       type: "string",           desc: "Plain-text summary of all findings." },
      { field: "pdf_url",      type: "string | null",    desc: "Relative URL to the generated PDF report, e.g. /api/reports/abc123.pdf." },
    ],
  },
  {
    method: "GET",
    path:   "/api/history",
    auth:   false,
    desc:   "Returns a paginated list of all analyses, newest first.",
    resp: [
      { field: "items",  type: "HistorySummary[]", desc: "Array of summary records for the requested page." },
      { field: "total",  type: "number",           desc: "Total number of analyses across all pages." },
      { field: "skip",   type: "number",           desc: "Offset applied to this response." },
      { field: "limit",  type: "number",           desc: "Page size applied to this response." },
    ],
    notes: "Query params: ?skip=0&limit=20 (default: skip=0, limit=20). SQL OFFSET/LIMIT — O(log n).",
  },
  {
    method: "GET",
    path:   "/api/history/:id",
    auth:   false,
    desc:   "Returns the full ClassifyResponse for a single analysis by ID.",
  },
  {
    method: "DELETE",
    path:   "/api/history/:id",
    auth:   false,
    desc:   "Deletes an analysis record and its associated PDF file. Returns 204 No Content on success, 404 if not found.",
  },
  {
    method: "GET",
    path:   "/api/health",
    auth:   false,
    desc:   "Returns system health status across 5 components: API, CNN model, knowledge base, RAG engine, and PDF writer.",
    resp: [
      { field: "status",     type: "string", desc: "healthy | degraded | down" },
      { field: "components", type: "object", desc: "Per-component status: ok | warning | error" },
      { field: "version",    type: "string", desc: "API version from src/__init__.py" },
    ],
  },
  {
    method: "GET",
    path:   "/api/metrics",
    auth:   true,
    desc:   "Prometheus-format metrics: total_analyses_count, cnn_model_accuracy, rag_chunks_count, pdf_reports_count, system_uptime_seconds.",
    notes:  "Requires X-API-Key header. Set DEEPCOIN_API_KEY env var on the server.",
  },
  {
    method: "GET",
    path:   "/api/reports/:filename",
    auth:   false,
    desc:   "Download a generated PDF report. Path traversal protected — only files in the reports/ directory are served.",
  },
  {
    method: "POST",
    path:   "/api/history/:id/feedback",
    auth:   false,
    desc:   "Submit a correction for an analysis (wrong classification flag).",
    body: [
      { field: "is_wrong",       type: "boolean", required: true,  desc: "true if the classification is incorrect." },
      { field: "correct_label",  type: "string",  required: false, desc: "Optional correct CN type ID." },
      { field: "note",           type: "string",  required: false, desc: "Free-text note from the reviewer." },
    ],
  },
];

const ROUTING_LOGIC = [
  {
    threshold: "conf > 85%",
    agent:     "Historian",
    color:     "#3b82f6",
    what:      "Retrieves the CN type record from the knowledge base, injects 5 structured RAG context blocks, and asks Gemini to write a grounded historical narrative. The LLM may only cite facts present in the context — no hallucination.",
  },
  {
    threshold: "40% ≤ conf ≤ 85%",
    agent:     "Validator",
    color:     "#f59e0b",
    what:      "Runs a multi-scale HSV forensic check to detect the metal colour (gold / bronze / silver) across three crop sizes. Compares against the KB expected material. Flags patina-ambiguity for ancient silver (Ag₂S sulphide) to prevent false mismatches. Also calls the Historian for contextual narrative.",
  },
  {
    threshold: "conf < 40%",
    agent:     "Investigator",
    color:     "#8b5cf6",
    what:      "Sends the image to a Vision-Language Model (Gemini or local Ollama qwen3-vl:4b) for attribute extraction, or falls back to pure OpenCV (HSV histogram + Sobel edge density) when no vision key is set. The extracted attributes are used to search ALL 9,541 CN types in the knowledge base — not just the 438 CNN-trained ones.",
  },
];

/* ── helpers ──────────────────────────────────────────────────────────────── */

function MethodBadge({ method }: { method: string }) {
  const colors: Record<string, [string, string]> = {
    GET:    ["#22c55e20", "#22c55e"],
    POST:   ["#3b82f620", "#3b82f6"],
    DELETE: ["#ef444420", "#ef4444"],
  };
  const [bg, fg] = colors[method] ?? ["#6b728020", "#6b7280"];
  return (
    <span className="text-[10px] font-black uppercase px-2 py-0.5 rounded-md" style={{ backgroundColor: bg, color: fg }}>
      {method}
    </span>
  );
}

/* ── component ────────────────────────────────────────────────────────────── */

export default function DocsPage() {
  return (
    <div className="py-10 max-w-4xl space-y-16">

      {/* Header */}
      <section className="space-y-3">
        <p className="text-xs font-black uppercase tracking-widest" style={{ color: "var(--brand-gold)" }}>
          Documentation
        </p>
        <h1 className="text-3xl font-black" style={{ color: "var(--text-primary)" }}>
          DeepCoin API
        </h1>
        <p className="text-sm max-w-xl" style={{ color: "var(--text-secondary)" }}>
          REST API reference, routing logic, and integration guide.
          The interactive Swagger UI is available locally at{" "}
          <code
            className="px-1.5 py-0.5 rounded text-xs"
            style={{ backgroundColor: "var(--surface-2)", color: "var(--text-primary)" }}
          >
            http://127.0.0.1:8000/docs
          </code>
          {" "}when the server is running.
        </p>
        <div className="flex flex-wrap gap-3 pt-2">
          <a
            href="http://127.0.0.1:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-lg transition-opacity hover:opacity-80"
            style={{ backgroundColor: "#10b98120", color: "#10b981", border: "1px solid #10b98140" }}
          >
            <Zap size={12} /> Swagger UI <ExternalLink size={10} />
          </a>
          <a
            href="https://github.com/ChaiebDhia/DeepCoin-Core"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs px-4 py-2 rounded-lg transition-opacity hover:opacity-80"
            style={{ backgroundColor: "var(--surface-1)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            <Code2 size={12} /> GitHub <ExternalLink size={10} />
          </a>
        </div>
      </section>

      {/* Quick-ref cards */}
      <section className="space-y-4">
        <h2 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>Quick reference</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {[
            { icon: Shield, label: "Auth",        val: "X-API-Key",  color: "#d4a853" },
            { icon: Clock,  label: "Rate limit",  val: "10 / min",   color: "#ef4444" },
            { icon: Cpu,    label: "Max upload",  val: "10 MB",      color: "#8b5cf6" },
            { icon: Database, label: "Base URL",  val: "/api",       color: "#3b82f6" },
          ].map(({ icon: Icon, label, val, color }) => (
            <div
              key={label}
              className="rounded-xl border p-4 text-center"
              style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
            >
              <Icon size={16} className="mx-auto mb-2" style={{ color }} />
              <p className="text-sm font-black" style={{ color }}>{val}</p>
              <p className="text-[10px]" style={{ color: "var(--text-muted)" }}>{label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Endpoints */}
      <section className="space-y-6">
        <h2 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>Endpoints</h2>
        {ENDPOINTS.map((ep) => (
          <div
            key={`${ep.method}-${ep.path}`}
            className="rounded-2xl border overflow-hidden"
            style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
          >
            {/* Header */}
            <div className="flex items-center gap-3 px-5 py-4 border-b" style={{ borderColor: "var(--border)" }}>
              <MethodBadge method={ep.method} />
              <code
                className="text-sm font-mono font-bold"
                style={{ color: "var(--text-primary)" }}
              >
                {ep.path}
              </code>
              {ep.auth && (
                <span
                  className="ml-auto text-[10px] font-bold px-2 py-0.5 rounded-full"
                  style={{ backgroundColor: "#d4a85320", color: "#d4a853" }}
                >
                  🔑 API Key
                </span>
              )}
            </div>

            <div className="p-5 space-y-4">
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>{ep.desc}</p>

              {ep.notes && (
                <p className="text-xs px-3 py-2 rounded-lg" style={{ backgroundColor: "var(--surface-2)", color: "var(--text-muted)" }}>
                  {ep.notes}
                </p>
              )}

              {ep.body && (
                <div className="space-y-2">
                  <p className="text-xs font-bold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Request body</p>
                  <div className="overflow-x-auto rounded-xl" style={{ border: "1px solid var(--border)" }}>
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ borderBottom: "1px solid var(--border)", backgroundColor: "var(--surface-2)" }}>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Field</th>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Type</th>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Required</th>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ep.body.map(f => (
                          <tr key={f.field} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                            <td className="px-4 py-2 font-mono" style={{ color: "#d4a853" }}>{f.field}</td>
                            <td className="px-4 py-2 font-mono text-[10px]" style={{ color: "#8b5cf6" }}>{f.type}</td>
                            <td className="px-4 py-2">
                              <span style={{ color: f.required ? "#22c55e" : "var(--text-muted)" }}>
                                {f.required ? "yes" : "no"}
                              </span>
                            </td>
                            <td className="px-4 py-2" style={{ color: "var(--text-secondary)" }}>{f.desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {ep.resp && (
                <div className="space-y-2">
                  <p className="text-xs font-bold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Response fields</p>
                  <div className="overflow-x-auto rounded-xl" style={{ border: "1px solid var(--border)" }}>
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ borderBottom: "1px solid var(--border)", backgroundColor: "var(--surface-2)" }}>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Field</th>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Type</th>
                          <th className="px-4 py-2 text-left font-medium" style={{ color: "var(--text-muted)" }}>Description</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ep.resp.map(f => (
                          <tr key={f.field} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                            <td className="px-4 py-2 font-mono" style={{ color: "#d4a853" }}>{f.field}</td>
                            <td className="px-4 py-2 font-mono text-[10px]" style={{ color: "#8b5cf6" }}>{f.type}</td>
                            <td className="px-4 py-2" style={{ color: "var(--text-secondary)" }}>{f.desc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </section>

      {/* Routing logic */}
      <section className="space-y-5">
        <div className="flex items-center gap-2">
          <GitBranch size={16} style={{ color: "var(--brand-gold)" }} />
          <h2 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>Agent routing</h2>
        </div>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          After CNN inference, a LangGraph state machine routes the result to the appropriate specialist agent
          based on the top-1 confidence score:
        </p>
        <div className="space-y-4">
          {ROUTING_LOGIC.map(({ threshold, agent, color, what }) => (
            <div
              key={agent}
              className="rounded-xl border-l-4 p-5"
              style={{
                borderLeftColor: color,
                border:          `1px solid var(--border)`,
                borderLeft:      `4px solid ${color}`,
                backgroundColor: "var(--surface-1)",
              }}
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="font-mono text-xs font-bold" style={{ color }}>{threshold}</span>
                <span className="text-sm font-bold" style={{ color: "var(--text-primary)" }}>→ {agent}</span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>{what}</p>
            </div>
          ))}
        </div>
      </section>

      {/* cURL example */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <Code2 size={16} style={{ color: "var(--brand-gold)" }} />
          <h2 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>Example — cURL</h2>
        </div>
        <pre
          className="p-5 rounded-xl text-xs overflow-x-auto leading-relaxed"
          style={{ backgroundColor: "#0d1520", color: "#a5f3fc", border: "1px solid var(--border)" }}
        >{`# Upload a coin image and get a full analysis
curl -X POST http://127.0.0.1:8000/api/classify \\
  -F "file=@/path/to/coin.jpg" \\
  -F "use_tta=false"

# Paginated history (page 2, 10 items per page)
curl "http://127.0.0.1:8000/api/history?skip=10&limit=10"

# Health check
curl http://127.0.0.1:8000/api/health

# Protected metrics endpoint
curl http://127.0.0.1:8000/api/metrics \\
  -H "X-API-Key: your_secret_key"`}</pre>
      </section>

      {/* Python example */}
      <section className="space-y-4">
        <h2 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>Example — Python</h2>
        <pre
          className="p-5 rounded-xl text-xs overflow-x-auto leading-relaxed"
          style={{ backgroundColor: "#0d1520", color: "#a5f3fc", border: "1px solid var(--border)" }}
        >{`import requests

with open("coin.jpg", "rb") as f:
    resp = requests.post(
        "http://127.0.0.1:8000/api/classify",
        files={"file": f},
        data={"use_tta": "false"},
    )

data = resp.json()
print(data["cnn"]["label"])        # e.g. "1015"
print(data["cnn"]["confidence"])   # e.g. 0.911
print(data["route_taken"])         # e.g. "historian"
print(data["pdf_url"])             # e.g. "/api/reports/abc123.pdf"`}</pre>
      </section>

      {/* Full docs link */}
      <section
        className="rounded-2xl border p-6 flex items-center justify-between gap-4"
        style={{ borderColor: "var(--border)", backgroundColor: "var(--surface-1)" }}
      >
        <div>
          <p className="font-bold text-sm" style={{ color: "var(--text-primary)" }}>Full engineering documentation</p>
          <p className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>
            Layer-by-layer build log, bug registry, architecture decisions, and academic context.
          </p>
        </div>
        <a
          href="https://github.com/ChaiebDhia/DeepCoin-Core/blob/main/ENGINEERING_JOURNAL.md"
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 flex items-center gap-2 text-xs px-4 py-2 rounded-lg transition-opacity hover:opacity-80"
          style={{ backgroundColor: "rgba(212,168,83,0.1)", color: "var(--brand-gold)", border: "1px solid rgba(212,168,83,0.3)" }}
        >
          <FileText size={13} /> Engineering Journal <ExternalLink size={11} />
        </a>
      </section>
    </div>
  );
}
