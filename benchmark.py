# benchmark.py
import logging
import warnings

# 1. Onderdruk alle standaard Python-waarschuwingen over Streamlit
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")

# 2. Zet het logniveau van Streamlit op ERROR (zodat WARNINGS verborgen blijven)
logging.getLogger("streamlit").setLevel(logging.ERROR)

import sys
import os
import time
import datetime
from pathlib import Path

# Zorg dat we de lokale map kunnen bereiken voor imports
sys.path.append(str(Path(__file__).parent))

from ai_router import ask_ai, get_session_state
from ollama import Client as OllamaClient

# --- 1. DYNAMISCHE OLLAMA DETECTIE ---
def get_local_ollama_models():
    """Haalt alle lokaal geïnstalleerde Ollama modellen op."""
    try:
        client = OllamaClient()
        response = client.list()
        models = []
        if hasattr(response, 'models'):
            models = [m.model for m in response.models]
        elif isinstance(response, dict) and 'models' in response:
            models = [m.get('model', m.get('name')) for m in response['models']]
        return sorted(list(set(models)))
    except Exception:
        # Als Ollama niet draait of niet bereikbaar is
        return []

# --- 2. TERMINAL VISUELE VOORTGANGSBALK ---
def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    """Toont een visuele voortgangsbalk in de console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # Print op dezelfde regel met \r
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print() # Volgende regel bij voltooien

# --- 3. BENCHMARK RUNNER ---
def run_test(name, provider, model_name, prompt):
    ss = get_session_state()
    ss["ai_provider"] = provider
    ss["model_name"] = model_name  # Zorg dat de router het juiste model pakt

    messages = [
        {"role": "system", "content": "Je bent een benchmark-assistent. Antwoord kort en bondig."},
        {"role": "user", "content": prompt},
    ]

    start = time.time()
    try:
        answer = ask_ai(messages)
        duration = time.time() - start

        # Veilige check voor het geval het antwoord geen string is
        if isinstance(answer, dict):
            answer_str = str(answer.get("message", {}).get("content", answer))
        else:
            answer_str = str(answer)

        tokens_out = len(answer_str) // 4
        tps = tokens_out / duration if duration > 0 else 0

        return {
            "name": name,
            "provider": provider,
            "model": model_name,
            "success": True,
            "error": "",
            "duration_sec": round(duration, 2),
            "tokens_out_est": tokens_out,
            "tokens_per_sec": round(tps, 2),
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "name": name,
            "provider": provider,
            "model": model_name,
            "success": False,
            "error": str(e),
            "duration_sec": round(duration, 2),
            "tokens_out_est": 0,
            "tokens_per_sec": 0,
        }

# --- 4. RAPPORT GENERATIE (HTML) ---
def generate_html(results, output_html):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="utf-8">
<title>AI Benchmark Report</title>
</head>
<body style="background:#111;color:#eee;font-family:system-ui;margin:2rem;">
<h1>AI Benchmark Report</h1>
<p style="color:#aaa;font-size:0.95rem;">Gegenereerd op: <strong>{timestamp}</strong></p>
<table style="border-collapse:collapse;width:100%;margin-top:1.5rem;">
<tr style="background:#222;">
<th style="border:1px solid #444;padding:8px;text-align:left;">Test Scenario</th>
<th style="border:1px solid #444;padding:8px;text-align:left;">Provider</th>
<th style="border:1px solid #444;padding:8px;text-align:left;">Model</th>
<th style="border:1px solid #444;padding:8px;text-align:center;">Status</th>
<th style="border:1px solid #444;padding:8px;text-align:right;">Duur (s)</th>
<th style="border:1px solid #444;padding:8px;text-align:right;">Tokens (geschat)</th>
<th style="border:1px solid #444;padding:8px;text-align:right;">Tokens/sec</th>
<th style="border:1px solid #444;padding:8px;text-align:left;">Foutmelding</th>
</tr>
"""
    for r in results:
        cls = "background:#132a13;" if r["success"] else "background:#3b0f0f;"
        html += f"<tr style='{cls}'>"
        html += f"<td style='border:1px solid #444;padding:6px;'>{r['name']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;'>{r['provider']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;'>{r['model']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;text-align:center;'>{'✔' if r['success'] else '✘'}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;text-align:right;'>{r['duration_sec']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;text-align:right;'>{r['tokens_out_est']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;text-align:right;'>{r['tokens_per_sec']}</td>"
        html += f"<td style='border:1px solid #444;padding:6px;color:#ff9999;'>{r['error'] if r['error'] else '-'}</td>"
        html += "</tr>"

    html += """</table>
</body>
</html>"""
    output_html.write_text(html, encoding="utf-8")

# --- 5. MAIN FLOW ---
def main():
    print("🔍 Lokale Ollama modellen zoeken...")
    local_models = get_local_ollama_models()

    if local_models:
        print(f"   Gevonden lokale modellen: {', '.join(local_models)}")
    else:
        print("   Geen actieve Ollama modellen gevonden (staat Ollama aan?).")

    # Basistests die we sowieso willen draaien
    base_tests = [
        ("Informatica Geschiedenis", "Gemini", "gemini-2.5-flash-lite", "Schrijf 500 woorden over de geschiedenis van de informatica."),
        ("Spoorwegen Nederland", "Gemini", "gemini-2.5-flash-lite", "Vat de geschiedenis van de Nederlandse spoorwegen samen in 3 alinea's."),
    ]

    # Dynamisch alle gevonden Ollama modellen toevoegen aan de testlijst!
    test_queue = []
    for name, provider, model, prompt in base_tests:
        test_queue.append((name, provider, model, prompt))

    for local_model in local_models:
        test_queue.append(
            (f"Lokale Test - {local_model}", "Lokaal", local_model, "Schrijf een kort gedicht over kunstmatige intelligentie.")
        )

    total_tests = len(test_queue)
    if total_tests == 0:
        print("❌ Geen tests om uit te voeren.")
        return

    print(f"\n🚀 Start benchmark ({total_tests} tests in wachtrij)...")
    results = []

    for idx, (name, provider, model, prompt) in enumerate(test_queue, 1):
        # Update de visuele voortgangsbalk
        print_progress_bar(idx - 1, total_tests, prefix='Voortgang:', suffix=f'Volgende: {model}', length=40)

        # Voer de daadwerkelijke test uit
        res = run_test(name, provider, model, prompt)
        results.append(res)

    # Toon 100% voltooid
    print_progress_bar(total_tests, total_tests, prefix='Voortgang:', suffix='Voltooid!       ', length=40)

    # Rapport wegschrijven
    out_dir = Path("benchmark_reports")
    out_dir.mkdir(exist_ok=True)
    out_html = out_dir / "benchmark_report.html"
    generate_html(results, out_html)

    # Korte samenvatting in de console
    print("\n📊 RECENTE RESULTATEN:")
    for r in results:
        status = "✔" if r["success"] else "✘"
        print(f"  [{status}] {r['model']} ({r['provider']}) -> {r['duration_sec']}s | {r['tokens_per_sec']} t/s")

    print(f"\n✨ HTML-rapport opgeslagen in: {out_html.resolve()}")

if __name__ == "__main__":
    main()