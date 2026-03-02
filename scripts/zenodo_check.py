"""Check all Zenodo versions for record 18637742."""
import requests, os
from dotenv import load_dotenv
load_dotenv(os.path.join("d:", os.sep, "00.test", "PAPER", "EthicaAI", ".env"))
t = os.getenv("ZENODO_ACCESS_TOKEN")
h = {"Authorization": "Bearer " + t}

# Get original record to find parent concept
r = requests.get("https://zenodo.org/api/records/18637742", headers=h)
d = r.json()
parent_id = d.get("parent", {}).get("id", "?")
print("Concept/Parent ID:", parent_id)

# List all versions under this concept
rv = requests.get(
    "https://zenodo.org/api/records",
    headers=h,
    params={"q": "parent.id:" + str(parent_id), "size": 50, "sort": "version", "allversions": True},
)
if rv.ok:
    hits = rv.json().get("hits", {}).get("hits", [])
    print("Total versions:", len(hits))
    print()
    for i, hit in enumerate(hits):
        vid = hit.get("id", "?")
        created = hit.get("created", "?")[:10]
        title = hit.get("metadata", {}).get("title", "?")[:70]
        is_latest = hit.get("versions", {}).get("is_latest", False)
        nfiles = len(hit.get("files", []))
        doi = hit.get("pids", {}).get("doi", {}).get("identifier", "?")
        marker = " <-- LATEST" if is_latest else ""
        print(f"  Version {i+1} (id={vid}, {created}): {doi}{marker}")
        print(f"    Title: {title}")
        print(f"    Files: {nfiles}")
        print()
else:
    print("Error:", rv.status_code, rv.text[:300])
