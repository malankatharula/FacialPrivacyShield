import { useState, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function NoiseOverlay() {
  return (
    <div style={{ position: "absolute", inset: 0, background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(19,192,163,0.03) 2px, rgba(19,192,163,0.03) 4px)", pointerEvents: "none", animation: "scanline 8s linear infinite" }} />
  );
}

function Badge({ color, label }) {
  return <span style={{ display: "inline-block", padding: "2px 10px", borderRadius: 3, fontSize: 11, fontFamily: "monospace", letterSpacing: "0.08em", background: color + "22", color: color, border: `1px solid ${color}44`, fontWeight: 600 }}>{label}</span>;
}

function ImageCard({ title, badge, badgeColor, src, overlay, hint }) {
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
        <span style={{ fontSize: 12, fontFamily: "monospace", color: "#64748b", letterSpacing: "0.1em", textTransform: "uppercase" }}>{title}</span>
        {badge && <Badge color={badgeColor} label={badge} />}
      </div>
      <div style={{ position: "relative", background: "#0d1117", borderRadius: 8, overflow: "hidden", border: "1px solid #1e293b", aspectRatio: "1", display: "flex", alignItems: "center", justifyContent: "center" }}>
        {src ? (<>{<img src={src} alt={title} style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />}{overlay && <NoiseOverlay />}</>) : (<span style={{ color: "#334155", fontSize: 13, fontFamily: "monospace" }}>{hint}</span>)}
      </div>
    </div>
  );
}

function ProgressBar({ elapsed, estimatedTotal }) {
  const pct = Math.min((elapsed / estimatedTotal) * 100, 95);
  return (
    <div>
      <div style={{ height: 3, background: "#1e293b", borderRadius: 2, overflow: "hidden", marginBottom: 8 }}>
        <div style={{ height: "100%", width: `${pct}%`, background: "linear-gradient(90deg, #13c4a3, #0b9478)", borderRadius: 2, transition: "width 1s linear" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, fontFamily: "monospace", color: "#64748b" }}>
        <span>PGD ensemble optimization running…</span>
        <span>{Math.round(elapsed)}s elapsed</span>
      </div>
    </div>
  );
}

export default function App() {
  const [originalSrc, setOriginalSrc] = useState(null);
  const [protectedSrc, setProtectedSrc] = useState(null);
  const [protectedBlob, setProtectedBlob] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [processingTime, setProcessingTime] = useState(null);
  const [dragging, setDragging] = useState(false);
  const timerRef = useRef(null);
  const fileRef = useRef(null);
  const ESTIMATED_SECONDS = 25;

  const startTimer = () => { setElapsed(0); timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000); };
  const stopTimer = () => { if (timerRef.current) clearInterval(timerRef.current); };

  const handleFile = useCallback(async (file) => {
    if (!file || !file.type.startsWith("image/")) { setErrorMsg("Please upload a JPEG or PNG image."); setStatus("error"); return; }
    const reader = new FileReader();
    reader.onload = (e) => setOriginalSrc(e.target.result);
    reader.readAsDataURL(file);
    setProtectedSrc(null); setProtectedBlob(null); setProcessingTime(null); setStatus("processing"); setErrorMsg(""); startTimer();
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/protect`, { method: "POST", body: formData });
      stopTimer();
      if (!res.ok) { const err = await res.json().catch(() => ({ detail: "Unknown error" })); throw new Error(err.detail || `HTTP ${res.status}`); }
      const t = res.headers.get("X-Processing-Time");
      if (t) setProcessingTime(t);
      const blob = await res.blob();
      setProtectedBlob(blob); setProtectedSrc(URL.createObjectURL(blob)); setStatus("done");
    } catch (e) { stopTimer(); setErrorMsg(e.message || "Protection failed. Is the backend running?"); setStatus("error"); }
  }, []);

  const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); const file = e.dataTransfer.files[0]; if (file) handleFile(file); }, [handleFile]);
  const downloadProtected = () => { if (!protectedBlob) return; const a = document.createElement("a"); a.href = URL.createObjectURL(protectedBlob); a.download = "protected.jpg"; a.click(); };
  const reset = () => { setOriginalSrc(null); setProtectedSrc(null); setProtectedBlob(null); setStatus("idle"); setErrorMsg(""); setElapsed(0); setProcessingTime(null); if (fileRef.current) fileRef.current.value = ""; };
  return (
    <>
      <style>{`* { box-sizing: border-box; margin: 0; padding: 0; } body { background: #060b12; color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; } @keyframes scanline { 0% { background-position: 0 0; } 100% { background-position: 0 100px; } } @keyframes pulse-ring { 0% { box-shadow: 0 0 0 0 rgba(19,196,163,0.3); } 70% { box-shadow: 0 0 0 12px rgba(19,196,163,0); } 100% { box-shadow: 0 0 0 0 rgba(19,196,163,0); } } button { cursor: pointer; } a { color: #13c4a3; }`}</style>
      <header style={{ borderBottom: "1px solid #0f1e2e", padding: "18px 32px", display: "flex", alignItems: "center", gap: 14 }}>
        <div style={{ width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg, #13c4a3, #0b9478)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>🛡</div>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.01em" }}>FacialPrivacyShield</div>
          <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", letterSpacing: "0.05em" }}>Ensemble Proxy Adversarial Protection · FC110534</div>
        </div>
        <div style={{ marginLeft: "auto" }}><Badge color="#13c4a3" label="ONLINE" /></div>
      </header>
      <div style={{ textAlign: "center", padding: "52px 24px 40px", maxWidth: 680, margin: "0 auto" }}>
        <h1 style={{ fontSize: "clamp(24px, 5vw, 38px)", fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.15, marginBottom: 16 }}>Protect your face from<br /><span style={{ color: "#13c4a3" }}>unauthorized recognition</span></h1>
        <p style={{ color: "#94a3b8", fontSize: 15, lineHeight: 1.6, maxWidth: 480, margin: "0 auto" }}>Upload a photo. We add imperceptible adversarial noise that corrupts how facial recognition models learn from your image — invisible to humans, disruptive to AI.</p>
      </div>
      <main style={{ maxWidth: 900, margin: "0 auto", padding: "0 24px 80px" }}>
        {status === "idle" && (
          <div onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop} onClick={() => fileRef.current?.click()} style={{ border: `2px dashed ${dragging ? "#13c4a3" : "#1e3a4f"}`, borderRadius: 12, padding: "56px 24px", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s", background: dragging ? "rgba(19,196,163,0.04)" : "transparent" }}>
            <div style={{ fontSize: 40, marginBottom: 16 }}>📷</div>
            <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 8 }}>Drop your photo here</div>
            <div style={{ color: "#64748b", fontSize: 13 }}>or click to browse — JPEG or PNG, max 10MB</div>
            <input ref={fileRef} type="file" accept="image/jpeg,image/png" style={{ display: "none" }} onChange={(e) => { const f = e.target.files[0]; if (f) handleFile(f); }} />
          </div>
        )}
        {status === "processing" && (
          <div>
            <div style={{ display: "flex", gap: 20, marginBottom: 28, flexWrap: "wrap" }}>
              <ImageCard title="Original" badge="UNPROTECTED" badgeColor="#f59e0b" src={originalSrc} hint="waiting..." />
              <ImageCard title="Protected" hint="computing..." />
            </div>
            <div style={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 10, padding: "20px 24px" }}>
              <ProgressBar elapsed={elapsed} estimatedTotal={ESTIMATED_SECONDS} />
              <div style={{ marginTop: 16, fontSize: 12, color: "#475569", fontFamily: "monospace" }}>Running PGD ensemble optimization over ArcFace · FaceNet · VGG-Face proxies (GPU · ε=8/255 · 40 iterations)</div>
            </div>
          </div>
        )}
        {status === "done" && (
          <div>
            <div style={{ display: "flex", gap: 20, marginBottom: 24, flexWrap: "wrap" }}>
              <ImageCard title="Original" badge="UNPROTECTED" badgeColor="#f59e0b" src={originalSrc} />
              <ImageCard title="Protected" badge="ADVERSARIALLY CLOAKED" badgeColor="#13c4a3" src={protectedSrc} overlay={true} />
            </div>
            <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
              {[{ label: "Method", value: "Ensemble PGD" }, { label: "Proxies", value: "ArcFace · FaceNet · VGG-Face" }, { label: "ε budget", value: "8/255 L∞" }, { label: "Time", value: processingTime || `${elapsed}s` }].map(({ label, value }) => (
                <div key={label} style={{ background: "#0d1117", border: "1px solid #1e293b", borderRadius: 8, padding: "10px 16px", flex: "1 1 160px" }}>
                  <div style={{ fontSize: 10, color: "#475569", fontFamily: "monospace", letterSpacing: "0.1em", marginBottom: 4 }}>{label.toUpperCase()}</div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", fontFamily: "monospace" }}>{value}</div>
                </div>
              ))}
            </div>
            <div style={{ display: "flex", gap: 12 }}>
              <button onClick={downloadProtected} style={{ flex: 1, padding: "14px 24px", background: "linear-gradient(135deg, #13c4a3, #0b9478)", color: "#fff", border: "none", borderRadius: 8, fontWeight: 700, fontSize: 14 }}>Download Protected Image</button>
              <button onClick={reset} style={{ padding: "14px 24px", background: "transparent", color: "#64748b", border: "1px solid #1e293b", borderRadius: 8, fontWeight: 600, fontSize: 14 }}>Protect Another</button>
            </div>
          </div>
        )}
        {status === "error" && (
          <div style={{ background: "#1a0a0a", border: "1px solid #7f1d1d", borderRadius: 10, padding: "24px", marginBottom: 20 }}>
            <div style={{ color: "#fca5a5", fontWeight: 600, marginBottom: 8 }}>⚠ Protection failed</div>
            <div style={{ color: "#ef4444", fontSize: 13, fontFamily: "monospace", marginBottom: 16 }}>{errorMsg}</div>
            <button onClick={reset} style={{ padding: "10px 20px", background: "transparent", color: "#94a3b8", border: "1px solid #374151", borderRadius: 6, fontSize: 13, fontWeight: 600 }}>Try again</button>
          </div>
        )}
        <div style={{ marginTop: 48, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
          {[{ icon: "👁", title: "Imperceptible", body: "Perturbations constrained to ε=8/255 L∞ — invisible to the human eye." }, { icon: "🔀", title: "Cross-architecture", body: "Optimized over 3 diverse proxy models, transferring to unseen FR architectures." }, { icon: "🔒", title: "Privacy-first", body: "Your image is processed and immediately discarded. Nothing is stored." }].map(({ icon, title, body }) => (
            <div key={title} style={{ background: "#0a1018", border: "1px solid #0f1e2e", borderRadius: 10, padding: "20px" }}>
              <div style={{ fontSize: 24, marginBottom: 10 }}>{icon}</div>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>{title}</div>
              <div style={{ color: "#64748b", fontSize: 13, lineHeight: 1.6 }}>{body}</div>
            </div>
          ))}
        </div>
      </main>
      <footer style={{ borderTop: "1px solid #0f1e2e", padding: "20px 32px", textAlign: "center", color: "#334155", fontSize: 12, fontFamily: "monospace" }}>
        FacialPrivacyShield · H.Y.M.T.P. Wickramasinghe · FC110534 · Faculty of Computing, University of Sri Jayewardenepura · 2026
      </footer>
    </>
  );
}