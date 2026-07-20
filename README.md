# Intelligent Healthcare Diagnostic Assistant

**Introduction to AI — 13-Week Capstone Project**

## What this is

An AI-powered diagnostic assistant that takes a patient's symptoms and vitals and
runs them through **five different AI techniques** at once, then combines their
outputs into a single diagnosis, urgency level, and treatment plan. It exists to
demonstrate how the core paradigms taught across the course — agents, logic,
probabilistic reasoning, machine learning, and planning — can be wired together
into one working system rather than studied in isolation.

The pipeline:

| Module | Technique | Role |
|---|---|---|
| `agent.py` | Model/goal-based agent | Perceives the patient, coordinates all modules, decides the next action |
| `knowledge_base.py` | First-order logic (forward/backward chaining) | Encodes medical rules and infers facts from symptoms |
| `bayesian_net.py` | Naive Bayes | Computes P(disease \| symptoms) |
| `ml_classifier.py` | Random Forest / Gradient Boosting | Learns disease patterns from training data |
| `neural_network.py` | Keras DNN | Deep-learning-based diagnosis |
| `fuzzy_controller.py` | Fuzzy logic | Turns vitals into a severity score (0–100) |
| `planner.py` | STRIPS planning | Generates a step-by-step treatment plan for the diagnosis |

## Requirements

- Python 3.10+
- Packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`
  (full list in `Environment Setup`, which is a `requirements.txt`)

Install with:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## How to use it

Run the whole pipeline on the built-in sample patient:
```bash
python3 app.py
```

This will:
1. Build the agent and register all 5 diagnostic modules
2. Train the ML classifier and neural network
3. Run a full diagnostic cycle (perceive → reason → decide) on a sample patient
4. Print a diagnosis, confidence, urgency level, and recommendations
5. Generate a step-by-step treatment plan
6. Save a 4-panel results dashboard to `diagnostic_dashboard.png`

### Diagnosing your own patient

Edit the `patient` object near the bottom of `app.py`:
```python
patient = PatientPercept(
    patient_id="P-1042",
    symptoms=["fever", "cough", "fatigue"],   # any known symptoms
    age=34,
    temperature=38.9,                          # °C
    heart_rate=102,                             # bpm
    blood_pressure="128/84",
)
```
Then re-run `python3 app.py`.

## Output

- **Terminal report** — diagnosis, confidence, urgency, severity score, recommended next action, and the treatment plan steps
- **`diagnostic_dashboard.png`** — Bayesian posterior chart, ML classifier top-5, fuzzy severity gauge, and final agent decision, all in one image
# The Team
 - Brian Karaba Wachira
 - Enock Katui
 - Kelvin Mwarano
 - Joylyn Wanjiru
 - Gladys Thuo
 - Margaret Wambui Kiragu
