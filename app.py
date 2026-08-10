# ============================================================
# CAPSTONE MAIN APPLICATION
# Intelligent Healthcare Diagnostic Assistant
# Introduction to AI — 13-Week Capstone
# ============================================================

import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings('ignore')

# Import all modules
from modules.agent            import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base   import MedicalKnowledgeBase
from modules.bayesian_net     import SimpleBayesianDiagnostics
from modules.ml_classifier    import MLDiagnosticClassifier
from modules.neural_network   import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner          import TreatmentPlanner

# ── ANSI Colors ────────────────────────────────────────────
class C:
    HEADER = '\033[95m'; BLUE   = '\033[94m'
    GREEN  = '\033[92m'; YELLOW = '\033[93m'
    RED    = '\033[91m'; BOLD   = '\033[1m'
    END    = '\033[0m'

def banner():
    print(f"""
{C.BOLD}{C.BLUE}
╔══════════════════════════════════════════════════════════╗
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║
║         Introduction to AI — Capstone Project             ║
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy       ║
╚══════════════════════════════════════════════════════════╝
{C.END}""")

def section(title: str):
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")

def build_system() -> HealthcareDiagnosticAgent:
    """Instantiate and wire all AI modules"""
    section("🔧 Building AI System — Registering Modules")

    agent = HealthcareDiagnosticAgent()

    print("\n  Initializing modules...")
    modules = {
        'KnowledgeBase': MedicalKnowledgeBase(),
        'BayesianNet':   SimpleBayesianDiagnostics(),
        'MLClassifier':  MLDiagnosticClassifier(),
        'NeuralNetwork': NeuralDiagnosticModel(),
        'Fuzzy':         FuzzySeverityAssessor(),
    }

    for name, module in modules.items():
        agent.register_module(name, module)

    print(f"\n  {C.GREEN}✅ {len(modules)} modules online{C.END}")
    return agent, modules['Fuzzy']


def pretrain_models(modules_dict):
    """Train the data-driven modules once, up front, so the
    interactive diagnostic run below is fast."""
    section("📚 Pre-training Learned Modules")

    ml = modules_dict.get('MLClassifier')
    if ml is not None:
        ml.train(verbose=True)

    nn = modules_dict.get('NeuralNetwork')
    if nn is not None:
        nn.train(epochs=25, verbose=0)
        print(f"\n  {C.GREEN}✅ Neural network trained{C.END}")


def print_report(report: dict, fuzzy_result: dict):
    section(f"📋 Diagnostic Report — Patient {report['patient_id']}")

    urgency_color = {
        'CRITICAL': C.RED, 'HIGH': C.RED,
        'MEDIUM': C.YELLOW, 'LOW': C.GREEN
    }.get(report['urgency'], C.END)

    print(f"\n  Symptoms      : {', '.join(report['symptoms'])}")
    print(f"  Diagnosis     : {C.BOLD}{report['diagnosis']}{C.END}")
    print(f"  Confidence    : {report['confidence']:.1%}")
    print(f"  Urgency       : {urgency_color}{C.BOLD}{report['urgency']}{C.END}")
    print(f"  Severity Score: {fuzzy_result['severity_score']:.1f}/100 "
          f"({fuzzy_result['severity_label']})")
    print(f"  Next Action   : {report['next_action']}")

    print("\n  Recommendations:")
    for rec in report['recommendations']:
        print(f"    {rec}")


def print_treatment_plan(plan: dict):
    section("🗓️  Treatment Plan (STRIPS Planner)")
    if plan.get('error'):
        print(f"\n  {C.RED}⚠ {plan['error']}{C.END}")
        return
    print(f"\n  Diagnosis: {plan['diagnosis']}  |  Urgency: {plan['urgency']}")
    print(f"  {plan['steps']} steps to reach goal state:\n")
    for step in plan['plan']:
        print(f"    {step['step']}. {step['action']:<28}"
              f"({step['duration']})")


def plot_dashboard(report: dict, fuzzy_result: dict,
                    bayes_result: dict, ml_result: dict):
    """Summary dashboard combining several modules' outputs"""
    fig = plt.figure(figsize=(13, 7))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # 1. Bayesian posterior distribution
    ax1 = fig.add_subplot(gs[0, 0])
    diseases, probs = zip(*sorted(bayes_result['all_posteriors'].items(),
                                   key=lambda x: x[1], reverse=True))
    ax1.barh(diseases, probs, color='#3498db')
    ax1.set_title("Bayesian Posterior P(Disease|Symptoms)", fontweight='bold')
    ax1.set_xlabel("Probability")

    # 2. ML classifier top-5
    ax2 = fig.add_subplot(gs[0, 1])
    top5_names, top5_probs = zip(*ml_result['top5'])
    ax2.barh(top5_names, top5_probs, color='#2ecc71')
    ax2.set_title(f"ML Classifier Top-5 ({ml_result['model_used']})",
                  fontweight='bold')
    ax2.set_xlabel("Probability")

    # 3. Fuzzy severity gauge
    ax3 = fig.add_subplot(gs[1, 0])
    score = fuzzy_result['severity_score']
    ax3.barh(['Severity'], [score], color='#e74c3c' if score >= 60 else
              '#f39c12' if score >= 40 else '#2ecc71')
    ax3.set_xlim(0, 100)
    ax3.set_title(f"Fuzzy Severity: {fuzzy_result['severity_label']} "
                  f"({score:.1f}/100)", fontweight='bold')

    # 4. Final agent decision summary (text panel)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    summary_text = (
        f"Final Diagnosis: {report['diagnosis']}\n"
        f"Confidence: {report['confidence']:.1%}\n"
        f"Urgency: {report['urgency']}\n"
        f"Next Action: {report['next_action']}"
    )
    ax4.text(0.05, 0.5, summary_text, fontsize=12, va='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1'))
    ax4.set_title("Agent Decision", fontweight='bold')

    plt.suptitle("Intelligent Healthcare Diagnostic Assistant — Dashboard",
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("diagnostic_dashboard.png", dpi=150, bbox_inches='tight')
    print(f"\n  {C.GREEN}✅ Saved: diagnostic_dashboard.png{C.END}")


def main():
    banner()

    agent, fuzzy = build_system()
    pretrain_models(agent._modules)

    # ── Sample patient case ──────────────────────────────
    section("🧑‍⚕️ Running Diagnostic Cycle — Sample Patient")
    patient = PatientPercept(
        patient_id="P-1042",
        symptoms=["fever", "cough", "fatigue", "loss_of_smell", "headache"],
        age=34,
        temperature=38.9,
        heart_rate=102,
        blood_pressure="128/84",
    )

    report = agent.run(patient)

    fuzzy_result = fuzzy.assess(
        patient.temperature, patient.heart_rate, len(patient.symptoms))
    bayes_result = agent._modules['BayesianNet'].analyze(patient)
    ml_result    = agent._modules['MLClassifier'].analyze(patient)

    print_report(report, fuzzy_result)

    # ── Treatment planning ───────────────────────────────
    planner = TreatmentPlanner()
    plan = planner.create_treatment_plan(report['diagnosis'], report['urgency'])
    print_treatment_plan(plan)

    # ── Agent log & performance ──────────────────────────
    agent.print_log()
    section("📈 Agent Performance")
    perf = agent.get_performance()
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # ── Visualization dashboard ──────────────────────────
    section("📊 Generating Dashboard")
    plot_dashboard(report, fuzzy_result, bayes_result, ml_result)

    section("✅ Run Complete")


if __name__ == "__main__":
    main()
