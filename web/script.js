// Set your API base URL here:
const API_URL = localStorage.getItem("API_URL") || "http://127.0.0.1:8000";
document.getElementById("api-url").textContent = API_URL;

const form = document.getElementById("predict-form");
const resultCard = document.getElementById("result");
const resultText = document.getElementById("result-text");
const predSpan = document.getElementById("prediction");
const confSpan = document.getElementById("confidence");
const metricsPre = document.getElementById("metrics");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
  ].reduce((acc, id) => {
    const el = document.getElementById(id);
    acc[id] = Number(el.value);
    return acc;
  }, {});

  try {
    const resp = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();

    resultText.textContent = data.result;
    predSpan.textContent = data.prediction;
    confSpan.textContent = `${(data.confidence * 100).toFixed(2)}%`;
    resultCard.classList.remove("hidden");

    // load metrics lazily
    const m = await fetch(`${API_URL}/metrics`).then(r => r.json());
    metricsPre.textContent = JSON.stringify(m, null, 2);
  } catch (err) {
    alert("Prediction failed: " + err.message);
  }
});
