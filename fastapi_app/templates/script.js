/* ==========================================================================
   Churn Signal — form interactions + prediction call
   Expects a FastAPI endpoint POST /predict that accepts the raw Telco-style
   fields below as JSON and returns JSON like:
     { "churn_probability": 0.73 }
   Adjust ENDPOINT and the response key in renderResult() if your app.py
   returns a different shape.
   ========================================================================== */

const ENDPOINT = "/predict";

const form = document.getElementById("churn-form");
const runBtn = document.getElementById("run-btn");
const resetBtn = document.getElementById("reset-btn");

const gaugeWrap = document.getElementById("gauge-wrap");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeValue = document.getElementById("gauge-value");
const riskPill = document.getElementById("risk-pill");
const riskText = document.getElementById("risk-text");
const resultCopy = document.getElementById("result-copy");
const signalBreakdown = document.getElementById("signal-breakdown");
const detailConfidence = document.getElementById("detail-confidence");
const detailAction = document.getElementById("detail-action");

const GAUGE_CIRCUMFERENCE = 578; // 2 * PI * r(92), matches CSS stroke-dasharray

/* ---------------------------------------------------------------------
   Segmented controls
   --------------------------------------------------------------------- */
document.querySelectorAll(".segmented").forEach((group) => {
  group.addEventListener("click", (e) => {
    const btn = e.target.closest(".seg-btn");
    if (!btn) return;
    group.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
  });
});

function getSegmentedValue(fieldName) {
  const group = document.querySelector(`.segmented[data-field="${fieldName}"]`);
  const active = group.querySelector(".seg-btn.active");
  return active ? active.dataset.value : null;
}

/* ---------------------------------------------------------------------
   Toggles
   --------------------------------------------------------------------- */
document.querySelectorAll(".toggle").forEach((toggle) => {
  toggle.addEventListener("click", () => {
    const isActive = toggle.classList.toggle("active");
    toggle.setAttribute("aria-pressed", String(isActive));
    toggle.dataset.value = isActive ? "Yes" : "No";
  });
});

function getToggleValue(fieldName) {
  const el = document.getElementById(fieldName);
  return el ? el.dataset.value : "No";
}

/* ---------------------------------------------------------------------
   Range sliders — live labels + fill track
   --------------------------------------------------------------------- */
const tenureInput = document.getElementById("tenure");
const tenureOut = document.getElementById("tenure-out");
const monthlyInput = document.getElementById("MonthlyCharges");
const monthlyOut = document.getElementById("monthly-out");
const totalInput = document.getElementById("TotalCharges");

let totalChargesTouched = false;
totalInput.addEventListener("input", () => { totalChargesTouched = true; });

function syncRangeFill(input) {
  const min = Number(input.min), max = Number(input.max), val = Number(input.value);
  const pct = ((val - min) / (max - min)) * 100;
  input.style.setProperty("--fill", `${pct}%`);
}

function updateTenureLabel() {
  tenureOut.textContent = `${tenureInput.value} mo`;
  syncRangeFill(tenureInput);
  maybeRecalcTotal();
}

function updateMonthlyLabel() {
  monthlyOut.textContent = `$${Number(monthlyInput.value).toFixed(2)}`;
  syncRangeFill(monthlyInput);
  maybeRecalcTotal();
}

function maybeRecalcTotal() {
  if (totalChargesTouched) return;
  const est = Number(tenureInput.value) * Number(monthlyInput.value);
  totalInput.value = est.toFixed(2);
}

tenureInput.addEventListener("input", updateTenureLabel);
monthlyInput.addEventListener("input", updateMonthlyLabel);
updateTenureLabel();
updateMonthlyLabel();

/* ---------------------------------------------------------------------
   Collect payload
   --------------------------------------------------------------------- */
function collectPayload() {
  return {
    gender: getSegmentedValue("gender"),
    SeniorCitizen: getToggleValue("SeniorCitizen") === "Yes" ? 1 : 0,
    Partner: getToggleValue("Partner"),
    Dependents: getToggleValue("Dependents"),
    tenure: Number(tenureInput.value),
    PhoneService: getToggleValue("PhoneService"),
    MultipleLines: document.getElementById("MultipleLines").value,
    InternetService: getSegmentedValue("InternetService"),
    OnlineSecurity: document.getElementById("OnlineSecurity").value,
    OnlineBackup: document.getElementById("OnlineBackup").value,
    DeviceProtection: document.getElementById("DeviceProtection").value,
    TechSupport: document.getElementById("TechSupport").value,
    StreamingTV: document.getElementById("StreamingTV").value,
    StreamingMovies: document.getElementById("StreamingMovies").value,
    Contract: getSegmentedValue("Contract"),
    PaperlessBilling: getToggleValue("PaperlessBilling"),
    PaymentMethod: document.getElementById("PaymentMethod").value,
    MonthlyCharges: Number(monthlyInput.value),
    TotalCharges: Number(totalInput.value),
  };
}

/* ---------------------------------------------------------------------
   Result rendering
   --------------------------------------------------------------------- */
function setGaugeState(state) {
  gaugeWrap.dataset.state = state;
}

function animateProbability(target) {
  const duration = 900;
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = target * eased;
    gaugeValue.textContent = `${(current * 100).toFixed(1)}%`;
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function riskLevelFor(prob) {
  if (prob < 0.30) return "low";
  if (prob < 0.60) return "medium";
  return "high";
}

const RISK_COPY = {
  low: {
    label: "Low risk",
    copy: "This account looks stable. No retention action needed right now — standard engagement is fine.",
    action: "Monitor only",
  },
  medium: {
    label: "Medium risk",
    copy: "Some churn signal here. Consider a proactive check-in or a loyalty offer before renewal.",
    action: "Proactive outreach",
  },
  high: {
    label: "High risk",
    copy: "Strong churn signal. Prioritize this account for a retention offer or a support call this week.",
    action: "Immediate retention offer",
  },
};

const RISK_COLOR = { low: "var(--safe)", medium: "var(--warn)", high: "var(--danger)" };

function renderResult(probability) {
  const prob = Math.max(0, Math.min(1, probability));
  const level = riskLevelFor(prob);
  const copy = RISK_COPY[level];

  setGaugeState("done");
  gaugeFill.style.stroke = RISK_COLOR[level];
  const offset = GAUGE_CIRCUMFERENCE * (1 - prob);
  gaugeFill.style.strokeDashoffset = String(offset);
  animateProbability(prob);

  riskPill.dataset.level = level;
  riskText.textContent = copy.label;

  resultCopy.textContent = copy.copy;

  signalBreakdown.hidden = false;
  detailConfidence.textContent = `${(prob * 100).toFixed(1)}%`;
  detailAction.textContent = copy.action;
}

function renderError(message) {
  setGaugeState("idle");
  gaugeValue.textContent = "—";
  riskPill.dataset.level = "idle";
  riskText.textContent = "Assessment failed";
  resultCopy.textContent = message || "Something went wrong reaching the model. Try again in a moment.";
  signalBreakdown.hidden = true;
}

/* ---------------------------------------------------------------------
   Submit
   --------------------------------------------------------------------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  runBtn.disabled = true;
  setGaugeState("loading");
  gaugeValue.textContent = "···";
  riskPill.dataset.level = "idle";
  riskText.textContent = "Reading signal…";
  resultCopy.textContent = "Scoring account against the churn model.";
  signalBreakdown.hidden = true;

  const payload = collectPayload();

  try {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`Server responded ${res.status}`);

    const data = await res.json();
    // Adjust this key to match whatever app.py actually returns
    const probability = data.churn_probability ?? data.probability ?? data.prediction;

    if (typeof probability !== "number") {
      throw new Error("Response did not include a numeric probability");
    }

    renderResult(probability);
  } catch (err) {
    renderError(err.message);
  } finally {
    runBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  setGaugeState("idle");
  gaugeFill.style.stroke = "var(--text-faint)";
  gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
  gaugeValue.textContent = "—";
  riskPill.dataset.level = "idle";
  riskText.textContent = "Awaiting input";
  resultCopy.textContent = "Fill in the account profile and run an assessment to see the retention read here.";
  signalBreakdown.hidden = true;
  totalChargesTouched = false;
  updateTenureLabel();
  updateMonthlyLabel();
});

// Initialize gauge dash state
gaugeFill.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
