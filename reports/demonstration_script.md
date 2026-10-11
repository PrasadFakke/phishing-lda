# Faculty demonstration sequence

1. Start the local app with `uvicorn app.main:app --reload` and open `http://127.0.0.1:8000`.
2. On Overview, explain the phishing classification goal and the structured feature dataset.
3. Open Dataset and point out class labels, class balance, and why `Index` is excluded.
4. Open Methodology and explain Fisher separation, shared covariance, the linear boundary, and shrinkage.
5. Open Evaluation to show held-out metrics, TN/FP/FN/TP, curves, and measured timing once training artifacts are available.
6. Open Demo, select a dataset row, show the predicted class and estimated probabilities, and explain that they are not certainty.
7. Close with limitations: static benchmark features, possible dataset shift, and the need to validate on newer unseen websites.
