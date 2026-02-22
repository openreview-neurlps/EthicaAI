"""Update metadata for existing Zenodo draft #18728438 to v4.0.0."""
from dotenv import load_dotenv
import os, requests, json

load_dotenv(".env")
token = os.getenv("ZENODO_ACCESS_TOKEN")
DRAFT_ID = 18728438

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

metadata = {
    "metadata": {
        "title": "Beyond Homo Economicus: Computational Instantiation and Empirical Analysis of Amartya Sen's Meta-Ranking Theory in Multi-Agent Social Dilemmas",
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "<p>This study formalizes Amartya Sen's Meta-Ranking theory within a "
            "Multi-Agent Reinforcement Learning (MARL) framework, demonstrating that "
            "<strong>Bounded Commitment</strong>&mdash;dynamic moral commitment preserving "
            "moral residue even under resource crisis&mdash;is the key mechanism for "
            "resolving social dilemmas.</p>"
            "<p><strong>v4 Updates (2026-02-22):</strong> Phase W critique defense: "
            "(W1) Explicit SUTVA analysis via exposure mapping with ATE decomposition into "
            "direct and spillover effects across 3 environments; "
            "(W2) Bounded Commitment spectrum &mdash; 4-model comparison with Moran ESS and "
            "&epsilon;-sensitivity Pareto analysis; "
            "(W3) Integrity-constrained U_meta variants resisting reward hacking under "
            "50% adversarial populations; "
            "(W4) Cross-domain behavioral fingerprint transfer protocol across 4 environment "
            "pairs. Total 42 analysis modules generating 88 figures from a single command.</p>"
            "<p>Code: https://github.com/Yesol-Pilot/EthicaAI</p>"
        ),
        "creators": [{"name": "Heo, Yesol", "affiliation": "Independent Researcher"}],
        "keywords": [
            "Meta-Ranking",
            "Social Value Orientation",
            "Causal Inference",
            "AI Alignment",
            "Amartya Sen",
            "Multi-Agent Reinforcement Learning",
            "Evolutionary Stability",
            "SUTVA",
            "Bounded Commitment",
            "Behavioral Transfer",
            "Reward Hacking Defense",
            "Public Goods Game",
        ],
        "access_right": "open",
        "license": "cc-by-4.0",
        "publication_date": "2026-02-22",
        "version": "4.0.0",
        "language": "eng",
        "notes": "NeurIPS 2026 submission (Main Track) + AIES 2026 extended version planned.",
    }
}

print(f"Updating metadata for draft #{DRAFT_ID}...")
resp = requests.put(
    f"https://zenodo.org/api/deposit/depositions/{DRAFT_ID}",
    data=json.dumps(metadata),
    headers=headers,
)

if resp.status_code == 200:
    result = resp.json()
    print(f"  Title: {result['title']}")
    print(f"  Version: {result['metadata'].get('version', 'N/A')}")
    print(f"  Keywords: {len(result['metadata'].get('keywords', []))} items")
    print(f"  Files: {len(result.get('files', []))} file(s)")
    for f in result.get("files", []):
        size_mb = f["size"] / (1024 * 1024)
        print(f"    - {f['filename']} ({size_mb:.1f} MB)")
    print(f"\n  Preview: https://zenodo.org/deposit/{DRAFT_ID}")
    print(f"\n  METADATA UPDATE COMPLETE")
else:
    print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
