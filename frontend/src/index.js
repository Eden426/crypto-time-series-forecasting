// ----------------------
// API Configuration
// ----------------------
// Replace this URL with your deployed Render backend URL (without trailing slash)
const PRODUCTION_API_URL = "https://crypto-forecasting-backend.onrender.com";

const isLocalhost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:";

const API_BASE_URL = (
  window.CRYPTO_API_URL || (isLocalhost ? "http://127.0.0.1:8000" : PRODUCTION_API_URL)
).replace(/\/+$/, "");

// Selected cryptocurrency symbol
let selectedSymbol = null;

document.addEventListener("DOMContentLoaded", () => {
  setupCryptoButtons();
  setupPredictButton();
  hideLoading();
});

// ----------------------
// Crypto button handling
// ----------------------
function setupCryptoButtons() {
  const buttons = document.querySelectorAll(".crypto-btn");
  const predictBtn = document.getElementById("predictBtn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      selectedSymbol = btn.dataset.symbol;
      clearResults();
      predictBtn.disabled = false;
    });
  });
}

// ----------------------
// Predict button handling
// ----------------------
function setupPredictButton() {
  const predictBtn = document.getElementById("predictBtn");
  predictBtn.addEventListener("click", predict);
}

// ----------------------
// Prediction logic
// ----------------------
async function predict() {
  if (!selectedSymbol) {
    showError("Please select a cryptocurrency first.");
    return;
  }

  const predictionEl = document.getElementById("prediction");
  const confidenceEl = document.getElementById("confidence");
  const loadingEl = document.getElementById("loading");
  const predictBtn = document.getElementById("predictBtn");

  clearResults();
  showLoading();
  predictBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol: selectedSymbol }),
    });

    if (!response.ok) {
      throw new Error("Backend error");
    }

    const data = await response.json();

    const formattedPrice = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(parseFloat(data.predicted_close));

    predictionEl.textContent = data.predicted_close
      ? `$${formattedPrice}`
      : "—";

    confidenceEl.textContent = data.confidence
      ? `${Number(data.confidence).toFixed(1)}%`
      : "—";
  } catch (err) {
    showError("Failed to get prediction. Check backend server.");
    predictionEl.textContent = "Error";
  } finally {
    hideLoading();
    predictBtn.disabled = false;
  }
}

// ----------------------
// Helper functions
// ----------------------
function clearResults() {
  document.getElementById("prediction").textContent = "—";
  document.getElementById("confidence").textContent = "—";
  hideError();
}

function showLoading() {
  const el = document.getElementById("loading");
  el.style.display = "flex";
}

function hideLoading() {
  const el = document.getElementById("loading");
  el.style.display = "none";
}

function showError(message) {
  const el = document.getElementById("error");
  el.textContent = message;
  el.style.display = "block";

  setTimeout(hideError, 4000);
}

function hideError() {
  const el = document.getElementById("error");
  el.textContent = "";
  el.style.display = "none";
}
