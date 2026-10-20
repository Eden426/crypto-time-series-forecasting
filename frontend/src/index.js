const LOCAL_API_URL = "http://127.0.0.1:8000";
const DEFAULT_API_URL = "https://crypto-time-series-forecasting.onrender.com";
const coinNames = { BTC: "Bitcoin", ETH: "Ethereum", LTC: "Litecoin", XRP: "XRP" };
const byId = (id) => document.getElementById(id);
const currency = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
let apiBaseUrl = DEFAULT_API_URL;
let selectedSymbol = "BTC";
let isBusy = false;
let apiConfigured;
let chartData = null;
let chartWidth = 0;
let successfulPrediction = false;

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".asset").forEach((button) => button.addEventListener("click", () => {
    if (isBusy) return;
    selectedSymbol = button.dataset.symbol;
    document.querySelectorAll(".asset").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    byId("outlookTitle").textContent = `${coinNames[selectedSymbol]} outlook`;
    byId("selectionHint").textContent = `${coinNames[selectedSymbol]} selected. Ready to explore?`;
    updateButton();
    clearResults();
  }));
  byId("predictBtn").addEventListener("click", predict);
  byId("retryBtn").addEventListener("click", predict);
  new ResizeObserver(() => {
    if (chartData && Math.floor(byId("priceChart").getBoundingClientRect().width) !== chartWidth) {
      renderPriceChart(...chartData);
    }
  }).observe(byId("chartSection"));
  apiConfigured = configureApi();
});

async function configureApi() {
  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname) || window.location.protocol === "file:";
  const override = window.CRYPTO_API_URL || new URLSearchParams(window.location.search).get("api");
  let config = {};
  try {
    const response = await fetch("/api/config", { headers: { Accept: "application/json" }, signal: AbortSignal.timeout(8000) });
    if (response.ok) config = await response.json();
  } catch (_) { /* Static previews can run without a configuration endpoint. */ }
  apiBaseUrl = String(override || (config.sameOrigin ? window.location.origin : config.apiBaseUrl) || (isLocal ? LOCAL_API_URL : DEFAULT_API_URL)).trim().replace(/\/+$/, "");
  // Health is informative; forecasts remain available while a sleeping service wakes up.
  checkHealth(isLocal && !override && !config.sameOrigin && !config.apiBaseUrl);
}

async function checkHealth(allowRemoteFallback) {
  try {
    let response = await fetch(`${apiBaseUrl}/health`, { signal: AbortSignal.timeout(65000) });
    if (!response.ok) throw new Error("Service unavailable");
    markEngine(true);
  } catch (_) {
    if (allowRemoteFallback && !isBusy) {
      apiBaseUrl = DEFAULT_API_URL;
      return checkHealth(false);
    }
    if (!successfulPrediction) markEngine(false);
  }
}

function markEngine(online) {
  byId("engineStatus").classList.toggle("connected", online);
  byId("engineStatus").classList.toggle("offline", !online);
  byId("statusText").textContent = online ? "Forecast service ready" : "Service unavailable";
}

function updateButton() {
  byId("predictBtn").textContent = isBusy ? "Generating forecast…" : `Generate ${coinNames[selectedSymbol]} forecast`;
}

function setBusy(busy) {
  isBusy = busy;
  byId("outlook").setAttribute("aria-busy", String(busy));
  byId("predictBtn").disabled = busy;
  byId("retryBtn").disabled = busy;
  document.querySelectorAll(".asset").forEach((button) => { button.disabled = busy; });
  updateButton();
}

function clearResults() {
  ["readout", "errorBox", "loading", "chartSection"].forEach((id) => { byId(id).hidden = true; });
  byId("emptyState").hidden = false;
  byId("forecastAnnouncement").textContent = "";
  byId("errorDetails").textContent = "";
  byId("errorBox").querySelector("details").open = false;
  chartData = null;
}

async function predict(event) {
  if (isBusy) return;
  const retry = event?.currentTarget === byId("retryBtn");
  const symbol = selectedSymbol;
  clearResults();
  setBusy(true);
  byId("emptyState").hidden = true;
  byId("loading").hidden = false;
  byId("loadingText").textContent = "Reading recent market data and running the model.";
  if (window.matchMedia("(max-width: 860px)").matches) {
    byId("outlook").scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  const waitingNotice = setTimeout(() => {
    byId("loadingText").textContent = "Still working. The service may need a moment to wake up. Please keep this page open.";
  }, 15000);
  try {
    await apiConfigured;
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ symbol }),
      signal: AbortSignal.timeout(90000),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : `Forecast request returned HTTP ${response.status}.`);
      error.status = response.status;
      throw error;
    }
    const price = data.predicted_close;
    const confidence = data.confidence;
    if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(confidence) || confidence < 0 || confidence > 100 || data.symbol !== symbol) {
      const error = new Error("The forecast response was incomplete or invalid.");
      error.invalid = true;
      throw error;
    }
    const history = Array.isArray(data.history) ? data.history.filter((value) => Number.isFinite(value) && value > 0).slice(-30) : [];
    const last = history.at(-1);
    byId("prediction").textContent = currency.format(price);
    byId("predictionFoot").textContent = `${coinNames[symbol]} (${symbol}) · USD estimate`;
    byId("confidence").textContent = `${confidence.toFixed(1)}%`;
    byId("confidenceBar").style.width = `${confidence}%`;
    byId("lastClose").textContent = last ? currency.format(last) : "Unavailable";
    const change = byId("priceChange");
    change.hidden = !last;
    if (last) {
      const percent = (price - last) / last * 100;
      change.classList.toggle("negative", percent < 0);
      change.textContent = `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}% ${percent >= 0 ? "above" : "below"} the latest recorded close`;
    }
    byId("forecastUpdated").textContent = `Generated ${new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit", timeZoneName: "short" }).format(new Date())}`;
    byId("readout").hidden = false;
    chartData = [history, price, symbol];
    renderPriceChart(...chartData);
    successfulPrediction = true;
    markEngine(true);
    byId("forecastAnnouncement").textContent = `${coinNames[symbol]} forecast ready. Estimated next close: ${currency.format(price)}.`;
  } catch (error) {
    const timeout = ["TimeoutError", "AbortError"].includes(error.name);
    let message = "We couldn’t connect to the forecast service. Check your connection and try again.";
    if ([429, 503].includes(error.status)) message = "Market data is temporarily unavailable. Wait a moment, then try again or choose another coin.";
    else if (timeout) message = "The service is taking longer than expected. It may still be waking up. Please try again.";
    else if (error.status || error.invalid) message = "The forecast service couldn’t complete this request. Please try again.";
    byId("error").textContent = message;
    byId("errorDetails").textContent = `${error.message}\nService: ${apiBaseUrl}`;
    byId("errorBox").hidden = false;
  } finally {
    clearTimeout(waitingNotice);
    byId("loading").hidden = true;
    setBusy(false);
    if (retry) byId("predictBtn").focus({ preventScroll: true });
  }
}

function renderPriceChart(history, forecast, symbol) {
  const chart = byId("chartSection");
  if (history.length < 2) { chart.hidden = true; return; }
  chart.hidden = false;
  const svg = byId("priceChart");
  const width = Math.max(240, Math.floor(svg.getBoundingClientRect().width));
  chartWidth = width;
  const left = 60, right = width - 16, top = 20, bottom = 220;
  const historyRight = right - Math.min(90, width * 0.2);
  const minValue = Math.min(...history, forecast), maxValue = Math.max(...history, forecast);
  const padding = Math.max((maxValue - minValue) * 0.12, Math.abs(maxValue) * 0.015, 0.01);
  const min = minValue - padding, max = maxValue + padding;
  const yFor = (value) => bottom - (value - min) / (max - min) * (bottom - top);
  const points = history.map((value, index) => [left + (historyRight - left) * index / (history.length - 1), yFor(value)]);
  const linePath = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const [lastX, lastY] = points.at(-1);
  const formatter = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 });
  const grid = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4, y = top + ratio * (bottom - top);
    return `<line class="chart-grid" x1="${left}" y1="${y}" x2="${right}" y2="${y}"/><text class="chart-y-label" x="${left - 8}" y="${y + 4}" text-anchor="end">${formatter.format(max - ratio * (max - min))}</text>`;
  }).join("");
  svg.setAttribute("viewBox", `0 0 ${width} 260`);
  svg.innerHTML = `<defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#176b57" stop-opacity=".12"/><stop offset="100%" stop-color="#176b57" stop-opacity="0"/></linearGradient></defs>${grid}<path class="chart-area" d="${linePath} L${lastX},${bottom} L${left},${bottom} Z"/><path class="chart-history" d="${linePath}"/><path class="chart-forecast" d="M${lastX},${lastY} L${right},${yFor(forecast)}"/><circle class="chart-point" cx="${lastX}" cy="${lastY}" r="4"/><circle class="chart-point-forecast" cx="${right}" cy="${yFor(forecast)}" r="5"/><text class="chart-x-label" x="${left}" y="250">Recent history</text><text class="chart-x-label" x="${right}" y="250" text-anchor="end">Estimate</text>`;
  svg.setAttribute("aria-label", `${coinNames[symbol]}: ${history.length} recent closing prices. Latest recorded close ${currency.format(history.at(-1))}. Next-day estimate ${currency.format(forecast)}.`);
  byId("chartAsset").textContent = `${coinNames[symbol]} (${symbol}) · ${history.length} recent daily closes`;
}
