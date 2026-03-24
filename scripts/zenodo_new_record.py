"""Create a brand new Zenodo record (v1), upload PDF, publish."""
import requests, os, json
from dotenv import load_dotenv

load_dotenv(os.path.join("d:", os.sep, "00.test", "PAPER", "EthicaAI", ".env"))
token = os.getenv("ZENODO_ACCESS_TOKEN")
h = {"Authorization": "Bearer " + token}
pdf_path = os.path.join("d:", os.sep, "00.test", "PAPER", "EthicaAI", "paper", "unified_paper.pdf")

# Step 1: Create blank deposit
print("=== Step 1: Create new record ===")
r = requests.post(
    "https://zenodo.org/api/records",
    headers={**h, "Content-Type": "application/json"},
    data=json.dumps({}),
)
print("Status:", r.status_code)
if r.status_code not in (200, 201):
    print("Error:", r.text[:500])
    exit(1)

rec = r.json()
new_id = rec["id"]
print("New record ID:", new_id)

# Step 2: Upload PDF (3-step: init, content, commit)
print("\n=== Step 2: Upload PDF ===")
fname = "EthicaAI_NeurIPS2026.pdf"

r_init = requests.post(
    "https://zenodo.org/api/records/" + str(new_id) + "/draft/files",
    headers={**h, "Content-Type": "application/json"},
    data=json.dumps([{"key": fname}]),
)
print("Init:", r_init.status_code)

with open(pdf_path, "rb") as fp:
    r_content = requests.put(
        "https://zenodo.org/api/records/" + str(new_id) + "/draft/files/" + fname + "/content",
        data=fp,
        headers={**h, "Content-Type": "application/octet-stream"},
    )
    print("Upload:", r_content.status_code)

r_commit = requests.post(
    "https://zenodo.org/api/records/" + str(new_id) + "/draft/files/" + fname + "/commit",
    headers=h,
)
print("Commit:", r_commit.status_code)

# Step 3: Set metadata
print("\n=== Step 3: Set metadata ===")
metadata = {
    "metadata": {
        "title": "From Situational to Unconditional: The Spectrum of Moral Commitment Required for Multi-Agent Survival in Non-linear Social Dilemmas",
        "resource_type": {"id": "publication-preprint"},
        "publication_date": "2026-03-02",
        "publisher": "Zenodo",
        "description": (
            "<p>NeurIPS 2026 submission.</p>"
            "<p>We establish the <strong>Moral Commitment Spectrum</strong>: "
            "a systematic relationship between environmental severity and the minimum "
            "moral commitment required for multi-agent system survival in non-linear "
            "Public Goods Games with tipping-point dynamics.</p>"
            "<p>Key findings:</p>"
            "<ul>"
            "<li>Situational commitment achieves ESS in linear environments</li>"
            "<li>All evaluated RL algorithms fail in non-linear environments (Nash Trap)</li>"
            "<li>Only unconditional crisis commitment prevents collapse</li>"
            "</ul>"
            "<p>Code: <a href='https://github.com/openreview-neurlps/EthicaAI'>GitHub</a></p>"
        ),
        "creators": [
            {"person_or_org": {"type": "personal", "given_name": "Anonymous", "family_name": "Author"}}
        ],
        "rights": [{"id": "cc-by-4.0"}],
        "subjects": [
            {"subject": "multi-agent reinforcement learning"},
            {"subject": "cooperation"},
            {"subject": "public goods game"},
            {"subject": "moral commitment"},
            {"subject": "NeurIPS 2026"},
        ],
    }
}
rm = requests.put(
    "https://zenodo.org/api/records/" + str(new_id) + "/draft",
    data=json.dumps(metadata),
    headers={**h, "Content-Type": "application/json"},
)
print("Metadata:", rm.status_code)
if rm.status_code != 200:
    print("Error:", rm.text[:500])

# Step 4: Publish
print("\n=== Step 4: Publish ===")
rp = requests.post(
    "https://zenodo.org/api/records/" + str(new_id) + "/draft/actions/publish",
    headers=h,
)
print("Publish:", rp.status_code)
if rp.ok:
    pub = rp.json()
    doi = pub.get("pids", {}).get("doi", {}).get("identifier", "?")
    print("DOI:", doi)
    print("URL: https://zenodo.org/records/" + str(new_id))
else:
    print("Error:", rp.text[:500])

print("\nDone! Please verify at https://zenodo.org/records/" + str(new_id))
