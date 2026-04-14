import { useState, useRef, useCallback, useEffect } from "react";

// ─── CSM Engine (Pure JS port of CSM_Calculator.m) ─────────────────────
function hanningPeriodic(N) {
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / N));
  const norm = Math.sqrt(0.375);
  for (let i = 0; i < N; i++) w[i] /= norm;
  return w;
}

function fftReal(re, im) {
  const N = re.length;
  if (N <= 1) return;
  if (N & (N - 1)) throw new Error("FFT requires power-of-2 length");
  let j = 0;
  for (let i = 1; i < N - 1; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = (2 * Math.PI) / len;
    const wR = Math.cos(ang), wI = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curR = 1, curI = 0;
      for (let k = 0; k < len / 2; k++) {
        const tR = curR * re[i + k + len / 2] - curI * im[i + k + len / 2];
        const tI = curR * im[i + k + len / 2] + curI * re[i + k + len / 2];
        re[i + k + len / 2] = re[i + k] - tR;
        im[i + k + len / 2] = im[i + k] - tI;
        re[i + k] += tR;
        im[i + k] += tI;
        const nR = curR * wR - curI * wI;
        curI = curR * wI + curI * wR;
        curR = nR;
      }
    }
  }
}

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

function computeCSM(data, nChannels, fs, nRec, overlapPct) {
  const N = data.length / nChannels;
  const overlapFrac = overlapPct / 100;
  const nBlocks = Math.floor((N - nRec) / (nRec * (1 - overlapFrac))) + 1;
  const window = hanningPeriodic(nRec);
  const nRecP2 = nextPow2(nRec);

  // CSM accumulator: [freq][ch_i][ch_j] as flat arrays (real + imag)
  const csmRe = new Float64Array(nRecP2 * nChannels * nChannels);
  const csmIm = new Float64Array(nRecP2 * nChannels * nChannels);

  const sRe = new Float64Array(nRecP2 * nChannels);
  const sIm = new Float64Array(nRecP2 * nChannels);

  for (let n = 0; n < nBlocks; n++) {
    const start = Math.floor(n * nRec * (1 - overlapFrac));
    // Window and FFT each channel
    for (let m = 0; m < nChannels; m++) {
      const off = m * nRecP2;
      for (let i = 0; i < nRecP2; i++) { sRe[off + i] = 0; sIm[off + i] = 0; }
      for (let i = 0; i < nRec; i++) sRe[off + i] = window[i] * data[(start + i) * nChannels + m];
      const chRe = sRe.subarray(off, off + nRecP2);
      const chIm = sIm.subarray(off, off + nRecP2);
      fftReal(chRe, chIm);
    }
    // Accumulate outer product
    for (let f = 0; f < nRecP2; f++) {
      for (let i = 0; i < nChannels; i++) {
        const siR = sRe[i * nRecP2 + f], siI = sIm[i * nRecP2 + f];
        for (let j = 0; j < nChannels; j++) {
          const sjR = sRe[j * nRecP2 + f], sjI = -sIm[j * nRecP2 + f]; // conj
          const idx = (f * nChannels + i) * nChannels + j;
          csmRe[idx] += siR * sjR - siI * sjI;
          csmIm[idx] += siR * sjI + siI * sjR;
        }
      }
    }
  }

  const scale = 2.0 / (nRec * fs * nBlocks);
  for (let i = 0; i < csmRe.length; i++) { csmRe[i] *= scale; csmIm[i] *= scale; }

  const df = fs / nRecP2;
  const freq = Array.from({ length: nRecP2 }, (_, i) => i * df);
  return { csmRe, csmIm, freq, nFreq: nRecP2, nChannels };
}

function extractAutospectra(csm) {
  const { csmRe, freq, nFreq, nChannels } = csm;
  const spectra = [];
  for (let ch = 0; ch < nChannels; ch++) {
    const psd = new Float64Array(nFreq);
    for (let f = 0; f < nFreq; f++) psd[f] = csmRe[(f * nChannels + ch) * nChannels + ch];
    spectra.push(psd);
  }
  return { freq, spectra, nChannels };
}

function extractCoherence(csm, chI, chJ) {
  const { csmRe, csmIm, freq, nFreq, nChannels } = csm;
  const coh = new Float64Array(nFreq);
  for (let f = 0; f < nFreq; f++) {
    const ii = csmRe[(f * nChannels + chI) * nChannels + chI];
    const jj = csmRe[(f * nChannels + chJ) * nChannels + chJ];
    const ijR = csmRe[(f * nChannels + chI) * nChannels + chJ];
    const ijI = csmIm[(f * nChannels + chI) * nChannels + chJ];
    const magSq = ijR * ijR + ijI * ijI;
    coh[f] = ii * jj > 0 ? magSq / (ii * jj) : 0;
  }
  return { freq, coh };
}

function computeCorrelation(data, nChannels, fs, maxSamples = 4000) {
  const N = Math.min(data.length / nChannels, maxSamples);
  const results = [];
  for (let ch = 0; ch < nChannels; ch++) {
    const x = [];
    for (let i = 0; i < N; i++) x.push(data[i * nChannels + ch]);
    const norm = Math.sqrt(x.reduce((s, v) => s + v * v, 0));
    const maxLag = Math.min(N - 1, 500);
    const corr = [];
    const taus = [];
    for (let lag = -maxLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < N; i++) {
        const j = i + lag;
        if (j >= 0 && j < N) sum += x[i] * x[j];
      }
      corr.push(norm > 0 ? sum / (norm * norm) : 0);
      taus.push(lag / fs * 1000);
    }
    results.push({ tau: taus, corr, ch });
  }
  return results;
}

// ─── CSV Parser ─────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
  let startRow = 0;
  const firstVals = lines[0].split(/[,\t;]/).map(Number);
  if (firstVals.some(isNaN)) startRow = 1;

  const rows = [];
  for (let i = startRow; i < lines.length; i++) {
    const vals = lines[i].split(/[,\t;]/).map(Number);
    if (vals.length > 0 && !vals.every(isNaN)) rows.push(vals.filter(v => !isNaN(v)));
  }
  const nCh = rows[0].length;
  const flat = new Float64Array(rows.length * nCh);
  for (let i = 0; i < rows.length; i++) for (let j = 0; j < nCh; j++) flat[i * nCh + j] = rows[i][j] || 0;
  return { data: flat, nSamples: rows.length, nChannels: nCh };
}

// ─── Chart Component (Canvas-based) ─────────────────────────────────────
function ChartCanvas({ series, xLabel, yLabel, title, logX = false, width = 700, height = 320, colors }) {
  const canvasRef = useRef(null);
  const defaultColors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"];
  const palette = colors || defaultColors;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !series || series.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const pad = { top: 40, right: 20, bottom: 50, left: 72 };
    const w = width - pad.left - pad.right;
    const h = height - pad.top - pad.bottom;

    // Gather bounds
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of series) {
      for (let i = 0; i < s.x.length; i++) {
        const xv = s.x[i];
        if (logX && xv <= 0) continue;
        if (xv < xMin) xMin = xv;
        if (xv > xMax) xMax = xv;
        if (isFinite(s.y[i])) {
          if (s.y[i] < yMin) yMin = s.y[i];
          if (s.y[i] > yMax) yMax = s.y[i];
        }
      }
    }
    const yPad = (yMax - yMin) * 0.08 || 1;
    yMin -= yPad; yMax += yPad;

    const toX = logX
      ? (v) => pad.left + ((Math.log10(Math.max(v, xMin)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * w
      : (v) => pad.left + ((v - xMin) / (xMax - xMin)) * w;
    const toY = (v) => pad.top + h - ((v - yMin) / (yMax - yMin)) * h;

    // Background
    ctx.fillStyle = "#0c0f1a";
    ctx.fillRect(0, 0, width, height);

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    const nYTicks = 6;
    for (let i = 0; i <= nYTicks; i++) {
      const y = pad.top + (h / nYTicks) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + w, y); ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.font = "11px 'DM Mono', monospace";
    ctx.textAlign = "right";
    for (let i = 0; i <= nYTicks; i++) {
      const val = yMax - ((yMax - yMin) / nYTicks) * i;
      const y = pad.top + (h / nYTicks) * i;
      ctx.fillText(val.toFixed(1), pad.left - 8, y + 4);
    }
    ctx.textAlign = "center";
    if (logX) {
      const decades = [10, 100, 1000, 10000, 100000];
      for (const d of decades) {
        if (d >= xMin && d <= xMax) {
          const x = toX(d);
          ctx.fillText(d >= 1000 ? (d / 1000) + "k" : String(d), x, pad.top + h + 18);
          ctx.strokeStyle = "rgba(255,255,255,0.05)";
          ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + h); ctx.stroke();
        }
      }
    } else {
      const nXT = 6;
      for (let i = 0; i <= nXT; i++) {
        const val = xMin + ((xMax - xMin) / nXT) * i;
        ctx.fillText(val.toFixed(1), toX(val), pad.top + h + 18);
      }
    }

    // Axis titles
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "12px 'DM Sans', sans-serif";
    ctx.fillText(xLabel || "", pad.left + w / 2, height - 6);
    ctx.save();
    ctx.translate(14, pad.top + h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel || "", 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.font = "bold 13px 'DM Sans', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(title || "", pad.left, pad.top - 14);

    // Lines
    ctx.lineWidth = 1.5;
    for (let si = 0; si < series.length; si++) {
      const s = series[si];
      ctx.strokeStyle = palette[si % palette.length];
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i++) {
        if (logX && s.x[i] <= 0) continue;
        if (!isFinite(s.y[i])) continue;
        const px = toX(s.x[i]), py = toY(s.y[i]);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Legend
    if (series.length > 1 || series[0].label) {
      ctx.font = "11px 'DM Mono', monospace";
      let lx = pad.left + w - 10;
      ctx.textAlign = "right";
      for (let si = series.length - 1; si >= 0; si--) {
        const label = series[si].label || `Series ${si + 1}`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = palette[si % palette.length];
        ctx.fillRect(lx - tw - 18, pad.top + 6 + si * 18, 12, 3);
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.fillText(label, lx, pad.top + 14 + si * 18);
      }
    }
  }, [series, xLabel, yLabel, title, logX, width, height, palette]);

  return <canvas ref={canvasRef} style={{ width, height, borderRadius: 8 }} />;
}

// ─── Main App ──────────────────────────────────────────────────────────
const TABS = ["Spectra", "Coherence", "Correlation"];

export default function CSMApp() {
  const [params, setParams] = useState({ fs: 51200, nRec: 4096, overlap: 50 });
  const [fileInfo, setFileInfo] = useState(null);
  const [rawData, setRawData] = useState(null);
  const [results, setResults] = useState(null);
  const [tab, setTab] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [cohPair, setCohPair] = useState([0, 1]);
  const fileRef = useRef();

  const handleFile = useCallback((e) => {
    const file = e.target.files[0];
    if (!file) return;
    setError(null); setResults(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const { data, nSamples, nChannels } = parseCSV(ev.target.result);
        setRawData({ data, nSamples, nChannels });
        setFileInfo({ name: file.name, nSamples, nChannels, size: file.size });
      } catch (err) { setError("Could not parse file: " + err.message); }
    };
    reader.readAsText(file);
  }, []);

  const run = useCallback(() => {
    if (!rawData) return;
    setProcessing(true); setError(null);
    setTimeout(() => {
      try {
        const { data, nChannels } = rawData;
        const csm = computeCSM(data, nChannels, params.fs, params.nRec, params.overlap);
        const auto = extractAutospectra(csm);
        const coh = nChannels >= 2 ? extractCoherence(csm, cohPair[0], cohPair[1]) : null;
        const corr = computeCorrelation(data, nChannels, params.fs);
        setResults({ csm, auto, coh, corr });
      } catch (err) { setError(err.message); }
      setProcessing(false);
    }, 50);
  }, [rawData, params, cohPair]);

  const maxFreq = params.fs / 2;

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(160deg, #070a14 0%, #0f1629 50%, #0c1220 100%)", color: "#e2e8f0", fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input[type=number] { -moz-appearance: textfield; }
        input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { -webkit-appearance: none; }
      `}</style>

      {/* Header */}
      <div style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", padding: "20px 32px", display: "flex", alignItems: "center", gap: 16 }}>
        <div style={{ width: 36, height: 36, borderRadius: 8, background: "linear-gradient(135deg, #3b82f6, #8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, fontWeight: 700, fontFamily: "'DM Mono', monospace" }}>
          Φ
        </div>
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em", color: "#f1f5f9" }}>CSM Batch Processor</h1>
          <p style={{ fontSize: 11, color: "#64748b", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>Cross-Spectral Matrix Calculator · Python Edition</p>
        </div>
      </div>

      <div style={{ display: "flex", gap: 0, minHeight: "calc(100vh - 77px)" }}>
        {/* Sidebar */}
        <div style={{ width: 300, flexShrink: 0, borderRight: "1px solid rgba(255,255,255,0.06)", padding: "24px 20px", display: "flex", flexDirection: "column", gap: 20 }}>
          {/* File input */}
          <div>
            <label style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b", display: "block", marginBottom: 8 }}>Data File (CSV)</label>
            <div
              onClick={() => fileRef.current?.click()}
              style={{ border: "1px dashed rgba(255,255,255,0.12)", borderRadius: 8, padding: "16px 12px", textAlign: "center", cursor: "pointer", transition: "border-color 0.2s", background: "rgba(255,255,255,0.02)" }}
              onMouseEnter={e => e.currentTarget.style.borderColor = "rgba(59,130,246,0.4)"}
              onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)"}
            >
              <input ref={fileRef} type="file" accept=".csv,.tsv,.txt,.dat" onChange={handleFile} style={{ display: "none" }} />
              {fileInfo ? (
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#3b82f6" }}>{fileInfo.name}</div>
                  <div style={{ fontSize: 11, color: "#64748b", marginTop: 4, fontFamily: "'DM Mono', monospace" }}>
                    {fileInfo.nSamples.toLocaleString()} samples × {fileInfo.nChannels} ch
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: 12, color: "#475569" }}>Click to select CSV file</div>
              )}
            </div>
          </div>

          {/* Parameters */}
          <div>
            <label style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b", display: "block", marginBottom: 12 }}>Parameters</label>
            {[
              { key: "fs", label: "Sampling Rate", unit: "Hz", step: 100 },
              { key: "nRec", label: "Record Length", unit: "pts", step: 256 },
              { key: "overlap", label: "Overlap", unit: "%", step: 5 },
            ].map(({ key, label, unit, step }) => (
              <div key={key} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, color: "#94a3b8" }}>{label}</span>
                  <span style={{ fontSize: 10, color: "#475569", fontFamily: "'DM Mono', monospace" }}>{unit}</span>
                </div>
                <input
                  type="number"
                  value={params[key]}
                  step={step}
                  onChange={e => setParams(p => ({ ...p, [key]: Number(e.target.value) }))}
                  style={{ width: "100%", padding: "8px 12px", borderRadius: 6, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", color: "#e2e8f0", fontSize: 14, fontFamily: "'DM Mono', monospace", outline: "none" }}
                />
              </div>
            ))}
          </div>

          {/* Derived info */}
          <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: 12, fontSize: 11, fontFamily: "'DM Mono', monospace", color: "#64748b", lineHeight: 1.8 }}>
            <div>df = <span style={{ color: "#94a3b8" }}>{(params.fs / params.nRec).toFixed(2)} Hz</span></div>
            <div>f_max = <span style={{ color: "#94a3b8" }}>{(params.fs / 2).toLocaleString()} Hz</span></div>
            <div>T_block = <span style={{ color: "#94a3b8" }}>{(params.nRec / params.fs * 1000).toFixed(1)} ms</span></div>
            {fileInfo && <div>N_blocks ≈ <span style={{ color: "#94a3b8" }}>{Math.max(1, Math.floor((fileInfo.nSamples - params.nRec) / (params.nRec * (1 - params.overlap / 100)) + 1))}</span></div>}
          </div>

          {/* Run button */}
          <button
            onClick={run}
            disabled={!rawData || processing}
            style={{
              width: "100%", padding: "12px", borderRadius: 8, border: "none",
              background: !rawData ? "rgba(255,255,255,0.05)" : "linear-gradient(135deg, #3b82f6, #6366f1)",
              color: !rawData ? "#475569" : "#fff", fontSize: 14, fontWeight: 600,
              cursor: !rawData ? "default" : "pointer", transition: "all 0.2s",
              opacity: processing ? 0.7 : 1,
            }}
          >
            {processing ? "Computing…" : "Compute CSM"}
          </button>

          {error && <div style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 6, padding: 10, fontSize: 12, color: "#fca5a5" }}>{error}</div>}
        </div>

        {/* Main content */}
        <div style={{ flex: 1, padding: "24px 28px", overflow: "auto" }}>
          {!results ? (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.4 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 48, marginBottom: 12 }}>Φ</div>
                <p style={{ fontSize: 14, color: "#64748b" }}>Load a CSV file and hit Compute</p>
                <p style={{ fontSize: 11, color: "#475569", marginTop: 8, maxWidth: 340, lineHeight: 1.6 }}>
                  Expects a CSV with numeric columns representing channels. Each row is a time sample.
                </p>
              </div>
            </div>
          ) : (
            <div>
              {/* Tabs */}
              <div style={{ display: "flex", gap: 2, marginBottom: 20, background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: 3 }}>
                {TABS.map((t, i) => (
                  <button
                    key={t}
                    onClick={() => setTab(i)}
                    style={{
                      flex: 1, padding: "8px 0", borderRadius: 6, border: "none",
                      background: tab === i ? "rgba(59,130,246,0.15)" : "transparent",
                      color: tab === i ? "#60a5fa" : "#64748b",
                      fontSize: 12, fontWeight: 600, cursor: "pointer", transition: "all 0.15s",
                    }}
                  >{t}</button>
                ))}
              </div>

              {/* Spectra tab */}
              {tab === 0 && results.auto && (
                <div>
                  <ChartCanvas
                    series={results.auto.spectra.map((psd, i) => {
                      const x = [], y = [];
                      for (let f = 1; f < results.auto.freq.length / 2; f++) {
                        x.push(results.auto.freq[f]);
                        y.push(10 * Math.log10(Math.max(psd[f], 1e-30)));
                      }
                      return { x, y, label: `Ch ${i + 1}` };
                    })}
                    xLabel="Frequency [Hz]"
                    yLabel="PSD [dB]"
                    title="Auto-Spectra (Narrowband)"
                    logX={true}
                  />
                </div>
              )}

              {/* Coherence tab */}
              {tab === 1 && (
                <div>
                  {results.auto.nChannels < 2 ? (
                    <p style={{ color: "#64748b", fontSize: 13 }}>Coherence requires at least 2 channels.</p>
                  ) : (
                    <div>
                      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center" }}>
                        <span style={{ fontSize: 11, color: "#64748b" }}>Channels:</span>
                        {[0, 1].map(idx => (
                          <select
                            key={idx}
                            value={cohPair[idx]}
                            onChange={e => {
                              const newPair = [...cohPair];
                              newPair[idx] = Number(e.target.value);
                              setCohPair(newPair);
                              // Recompute coherence
                              const coh = extractCoherence(results.csm, newPair[0], newPair[1]);
                              setResults(r => ({ ...r, coh }));
                            }}
                            style={{ padding: "4px 8px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.05)", color: "#e2e8f0", fontSize: 12 }}
                          >
                            {Array.from({ length: results.auto.nChannels }, (_, i) => (
                              <option key={i} value={i}>Ch {i + 1}</option>
                            ))}
                          </select>
                        ))}
                      </div>
                      {results.coh && (
                        <ChartCanvas
                          series={[{
                            x: results.coh.freq.slice(1, results.coh.freq.length / 2),
                            y: Array.from(results.coh.coh).slice(1, results.coh.freq.length / 2),
                            label: `γ² Ch${cohPair[0]+1}–Ch${cohPair[1]+1}`
                          }]}
                          xLabel="Frequency [Hz]"
                          yLabel="Coherence γ²"
                          title={`Coherence — Ch ${cohPair[0]+1} vs Ch ${cohPair[1]+1}`}
                          logX={true}
                          colors={["#8b5cf6"]}
                        />
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Correlation tab */}
              {tab === 2 && results.corr && (
                <ChartCanvas
                  series={results.corr.map(({ tau, corr, ch }) => ({
                    x: tau, y: corr, label: `Ch ${ch + 1}`
                  }))}
                  xLabel="Lag τ [ms]"
                  yLabel="Normalised Correlation"
                  title="Auto-Correlation"
                  logX={false}
                />
              )}

              {/* Stats bar */}
              <div style={{ marginTop: 20, display: "flex", gap: 12, flexWrap: "wrap" }}>
                {[
                  { label: "Channels", value: results.auto.nChannels },
                  { label: "Freq bins", value: results.csm.nFreq },
                  { label: "df", value: (params.fs / params.nRec).toFixed(2) + " Hz" },
                  { label: "f_max", value: (params.fs / 2).toLocaleString() + " Hz" },
                ].map(({ label, value }) => (
                  <div key={label} style={{ background: "rgba(255,255,255,0.03)", borderRadius: 6, padding: "8px 14px", flex: "1 1 120px" }}>
                    <div style={{ fontSize: 10, color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: "#94a3b8", fontFamily: "'DM Mono', monospace", marginTop: 2 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
