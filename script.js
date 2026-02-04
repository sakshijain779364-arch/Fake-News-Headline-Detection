const form = document.getElementById("headline-form");
const input = document.getElementById("headline-input");
const resultBox = document.getElementById("result");
const labelEl = document.getElementById("label");
const confEl = document.getElementById("confidence");
const errorBox = document.getElementById("error");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");

  const headline = input.value.trim();
  if (!headline) {
    showError("Please enter a headline.");
    return;
  }

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headline }),
    });

    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }

    labelEl.textContent = data.label;
    confEl.textContent = `${Math.round(data.confidence * 100)}%`;
    resultBox.classList.remove("hidden");
  } catch (err) {
    showError("Network error. Please try again.");
  }
});

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}