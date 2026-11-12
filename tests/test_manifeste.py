"""Versioned artefacts (model_final/, data/ caches) match their provenance manifest.

Detects any silent regeneration. To print the fingerprints to copy into
model_final/MANIFEST.json after a deliberate retraining:
    python tests/test_manifeste.py
"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTE = REPO / "model_final" / "MANIFEST.json"
TEXTE = {".csv", ".json"}


def empreinte(chemin: Path) -> str:
    contenu = chemin.read_bytes()
    if chemin.suffix in TEXTE:      # neutralise Git's CRLF conversion on Windows
        contenu = contenu.replace(b"\r\n", b"\n")
    return hashlib.sha256(contenu).hexdigest()


def fichiers_versionnes():
    sortie = subprocess.run(["git", "ls-files", "model_final", "data"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    return sorted(f for f in sortie.split() if not f.endswith("MANIFEST.json"))


def test_artefacts_conformes_au_manifeste():
    attendu = json.loads(MANIFESTE.read_text(encoding="utf-8"))["empreintes_sha256"]
    differents = [f for f, h in attendu.items() if empreinte(REPO / f) != h]
    assert not differents, f"modified since the manifest: {differents}"


def test_tout_artefact_versionne_est_dans_le_manifeste():
    attendu = json.loads(MANIFESTE.read_text(encoding="utf-8"))["empreintes_sha256"]
    manquants = sorted(set(fichiers_versionnes()) - set(attendu))
    assert not manquants, f"missing from the manifest: {manquants}"


if __name__ == "__main__":
    print(json.dumps({f: empreinte(REPO / f) for f in fichiers_versionnes()}, indent=2))
