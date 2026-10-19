const LOCAL_API_URL = "http://127.0.0.1:8000";
const DEFAULT_API_URL = "https://crypto-time-series-forecasting.onrender.com";
let apiBaseUrl = null;
let selectedSymbol = null;

const coinNames = { BTC: "Bitcoin", ETH: "Ethereum", LTC: "Litecoin", XRP: "Ripple" };

document.addEventListener("DOMContentLoaded", async () => {
  setupCryptoButtons();
  document.getElementById("predictBtn").addEventListener("click", predict);
  await configureApi();
});

async function configureApi() {
  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname) || window.location.protocol === "file:";
  const queryUrl = new URLSearchParams(window.location.search).get("api");
  const override = window.CRYPTO_API_URL || queryUrl;
  try {
    const response = await fetch("/api/config", { headers: { Accept: "application/json" } });
    if (response.ok) {
      const config = await response.json();
      apiBaseUrl = config.apiBaseUrl || null;
    }
  } catch (_) {
    // The static frontend may be served outside Vercel; use the local fallback below.
  }
  apiBaseUrl = normalizeUrl(override || apiBaseUrl || (isLocal ? LOCAL_API_URL : DEFAULT_API_URL));
  const connected = await pingHealth(apiBaseUrl);
  if (!connected && isLocal && !override && apiBaseUrl.includes("127.0.0.1")) {
    const remote = normalizeUrl(DEFAULT_API_URL);
    if (await pingHealth(remote)) {
      apiBaseUrl = remote;
      markEngine(true);
      return;
    }
  }
  if (connected) {
    markEngine(true);
    return;
  }
  markEngine(false);
  if (!override && !apiBaseUrl.includes("127.0.0.1")) {
    showError("The configured API could not be reached. In Vercel, set CRYPTO_API_URL to your Render service URL (for example, https://your-service.onrender.com), then redeploy.");
  }
}

async function pingHealth(url) {
  try {
    const response = await fetch(`${url}/health`, { headers: { Accept: "application/json" } });
    return response.ok;
  } catch (_) {
    return false;
  }
}

function markEngine(online) {
  const status = document.getElementById("engineStatus");
  status.classList.toggle("connected", online);
  status.classList.toggle("offline", !online);
  document.getElementById("statusText").textContent = online ? "Engine online" : "Engine offline";
}

function normalizeUrl(value) {
  return String(value).trim().replace(/\/+$/, "");
}

function setupCryptoButtons() {
  const buttons = [...document.querySelectorAll(".asset")];
  buttons.forEach((button) => button.addEventListener("click", () => {
    buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    selectedSymbol = button.dataset.symbol;
    clearResults();
    document.getElementById("predictBtn").disabled = false;
    document.getElementById("selectionHint").textContent = `${coinNames[selectedSymbol]} selected · ready for a next-day estimate.`;
  }));
}

async function predict() {
  if (!selectedSymbol || !apiBaseUrl) return showError("Choose a cryptocurrency and make sure the forecast engine is online.");
  const button = document.getElementById("predictBtn");
  button.disabled = true;
  clearResults({ keepOutlook: true });
  showOutlook();
  document.getElementById("loading").hidden = false;
  try {
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ symbol: selectedSymbol }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
      throw new Error(detail ? `API error (${response.status}): ${detail}` : `API error (${response.status}).`);
    }
    const price = Number(data.predicted_close);
    const confidence = Number(data.confidence);
    if (!Number.isFinite(price) || !Number.isFinite(confidence)) throw new Error("The API returned an invalid prediction. Check the backend logs.");
    document.getElementById("prediction").textContent = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(price);
    document.getElementById("confidence").textContent = `${confidence.toFixed(1)}%`;
    document.getElementById("confidenceBar").style.width = `${Math.max(0, Math.min(100, confidence))}%`;
    document.getElementById("predictionFoot").textContent = `${coinNames[data.symbol] || data.symbol} · USD · next daily close`;
    document.getElementById("forecastDate").textContent = "Next day";
    renderPriceChart(data.history, price, data.symbol);
  } catch (error) {
    showError(`${error.message} API: ${apiBaseUrl}`);
  } finally {
    document.getElementById("loading").hidden = true;
    button.disabled = false;
  }
}

function clearResults({ keepOutlook = false } = {}) {
  document.getElementById("prediction").textContent = "—";
  document.getElementById("confidence").textContent = "—";
  document.getElementById("confidenceBar").style.width = "0%";
  document.getElementById("predictionFoot").textContent = "USD · next daily close";
  document.getElementById("error").hidden = true;
  document.getElementById("loading").hidden = true;
  document.getElementById("chartSection").hidden = true;
  if (!keepOutlook) document.getElementById("outlook").hidden = true;
}

function showOutlook() {
  document.getElementById("outlook").hidden = false;
}

function showError(message) {
  showOutlook();
  const error = document.getElementById("error");
  error.textContent = message;
  error.hidden = false;
}

function renderPriceChart(values, forecast, symbol) {
  const history = Array.isArray(values) ? values.map(Number).filter(Number.isFinite).slice(-30) : [];
  const chart = document.getElementById("chartSection");
  if (history.length < 2) {
    chart.hidden = true;
    return;
  }

  const svg = document.getElementById("priceChart");
  const allValues = [...history, forecast];
  let min = Math.min(...allValues);
  let max = Math.max(...allValues);
  const padding = Math.max((max - min) * 0.12, Math.abs(max) * 0.015, 0.01);
  min -= padding;
  max += padding;

  const left = 74;
  const right = 782;
  const top = 22;
  const bottom = 232;
  const xFor = (index) => left + (right - left) * index / history.length;
  const yFor = (value) => bottom - (value - min) / (max - min) * (bottom - top);
  const historicalPoints = history.map((value, index) => [xFor(index), yFor(value)]);
  const forecastPoint = [xFor(history.length), yFor(forecast)];
  const linePath = historicalPoints.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${historicalPoints.at(-1)[0].toFixed(1)},${bottom} L${historicalPoints[0][0].toFixed(1)},${bottom} Z`;
  const formatter = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 });
  const grid = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const y = top + ratio * (bottom - top);
    const value = max - ratio * (max - min);
    return `<line class="chart-grid" x1="${left}" y1="${y.toFixed(1)}" x2="${right}" y2="${y.toFixed(1)}"/><text class="chart-y-label" x="${left - 10}" y="${(y + 3).toFixed(1)}" text-anchor="end">${formatter.format(value)}</text>`;
  }).join("");
  const [lastX, lastY] = historicalPoints.at(-1);
  svg.innerHTML = `<defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#0f766e" stop-opacity=".16"/><stop offset="100%" stop-color="#0f766e" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${areaPath}"/><path class="chart-history" d="${linePath}"/><path class="chart-forecast" d="M${lastX.toFixed(1)},${lastY.toFixed(1)} L${forecastPoint[0].toFixed(1)},${forecastPoint[1].toFixed(1)}"/><circle class="chart-point" cx="${lastX.toFixed(1)}" cy="${lastY.toFixed(1)}" r="4.5"/><circle class="chart-point-forecast" cx="${forecastPoint[0].toFixed(1)}" cy="${forecastPoint[1].toFixed(1)}" r="5.5"/>`;
  svg.setAttribute("aria-label", `${coinNames[symbol] || symbol} daily closing prices and next-day estimate`);
  document.getElementById("chartAsset").textContent = `${coinNames[symbol] || symbol} (${symbol})`;
  chart.hidden = false;
}
