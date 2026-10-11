const rowSelect = document.querySelector('#row');
const featurePanel = document.querySelector('#features');
const featureCaption = document.querySelector('#feature-caption');
const predictionPanel = document.querySelector('#prediction');
const predictButton = document.querySelector('#predict');

async function loadSelectedFeatures() {
  if (!rowSelect || !featurePanel) return;
  featurePanel.textContent = 'Loading selected record…';
  featureCaption.textContent = `Record ${rowSelect.value}`;
  try {
    const response = await fetch(`/api/row/${rowSelect.value}`);
    const values = await response.json();
    if (!response.ok) throw new Error(values.detail || 'Could not load this record.');
    featurePanel.textContent = JSON.stringify(values, null, 2);
  } catch (error) {
    featurePanel.textContent = error.message;
  }
}

if (rowSelect) {
  rowSelect.addEventListener('change', loadSelectedFeatures);
  loadSelectedFeatures();
}

if (predictButton) predictButton.addEventListener('click', async () => {
  predictButton.disabled = true;
  predictButton.textContent = 'Classifying…';
  predictionPanel.textContent = 'Running the saved model pipeline…';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ row: Number(rowSelect.value) })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed.');
    const label = data.prediction;
    const probability = Number(data.phishing_probability);
    const percent = (probability * 100).toFixed(1);
    predictionPanel.innerHTML = '';
    const outcome = document.createElement('div');
    outcome.className = `result-result ${label}`;
    outcome.textContent = label === 'phishing' ? 'Phishing indicators detected' : 'Classified as legitimate';
    const description = document.createElement('div');
    description.className = 'result-details';
    description.textContent = `Estimated phishing probability · ${percent}%`;
    const track = document.createElement('div');
    track.className = 'probability-track';
    track.setAttribute('role', 'img');
    track.setAttribute('aria-label', `Estimated phishing probability ${percent} percent`);
    const fill = document.createElement('div');
    fill.className = 'probability-fill';
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    track.append(fill);
    const note = document.createElement('div');
    note.className = 'result-details';
    note.textContent = `Legitimate estimate · ${(Number(data.legitimate_probability) * 100).toFixed(1)}%`;
    predictionPanel.append(outcome, description, track, note);
  } catch (error) {
    predictionPanel.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.innerHTML = 'Classify record <span aria-hidden="true">→</span>';
  }
});
