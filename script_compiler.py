# script_compiler.py
# -------------------------------------------------
# Plaats dit bestand in de map met al je .py‑scripts.
# Run:   python merge_to_txt.py
# -------------------------------------------------

from pathlib import Path

def main():
    # 1️⃣ De map waarin dit script zelf staat
    here = Path(__file__).parent

    # 2️⃣ Alle *.py‑bestanden in die map (exclusief dit script)
    py_files = sorted(p for p in here.glob("*.py") if p.name != Path(__file__).name)

    # 3️⃣ Doel‑bestand (kan je aanpassen)
    target = here / "all_scripts.txt"

    # 4️⃣ Schrijf de inhoud met een nette scheiding
    with target.open("w", encoding="utf-8") as out:
        out.write(f"# Samengevoegde scripts – {len(py_files)} bestanden\n")
        out.write(f"# Map: {here.resolve()}\n")
        out.write("-" * 40 + "\n\n")

        for p in py_files:
            out.write(f"# ---- {p.name} ----\n")
            out.write(p.read_text(encoding="utf-8"))
            out.write("\n\n")          # lege regel tussen scripts

    print(f"✅  {len(py_files)} scripts samengevoegd → {target.name}")

if __name__ == "__main__":
    main()