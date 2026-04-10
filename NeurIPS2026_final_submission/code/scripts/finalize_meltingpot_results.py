#!/usr/bin/env python3
"""Fetch remote Melting Pot shard outputs and finalize merged paper snippets."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import textwrap
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
CODE_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = CODE_DIR / "outputs" / "meltingpot"
PAPER_ROOT = SCRIPT_PATH.parents[4]

sys.path.insert(0, str(PAPER_ROOT))
import monitor_experiments as monitor  # noqa: E402


DEFAULT_MAC_SNAPSHOT = RESULTS_DIR / "meltingpot_final_results_mac_head.json"
DEFAULT_SERVER_SNAPSHOT = RESULTS_DIR / "meltingpot_final_results_server_tail_snapshot.json"
DEFAULT_MERGED = RESULTS_DIR / "meltingpot_final_results_merged.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize Melting Pot results after Mac/server shards complete."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow syncing/merging even if the shard counts are not yet complete.",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only fetch the remote shard JSON files into local snapshot paths.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Do not fetch from remote; use existing local snapshot files.",
    )
    parser.add_argument(
        "--mac-output",
        default=str(DEFAULT_MAC_SNAPSHOT),
        help="Local snapshot path for the Mac head shard JSON.",
    )
    parser.add_argument(
        "--server-output",
        default=str(DEFAULT_SERVER_SNAPSHOT),
        help="Local snapshot path for the Linux server tail shard JSON.",
    )
    parser.add_argument(
        "--merged-output",
        default=str(DEFAULT_MERGED),
        help="Local merged result JSON path.",
    )
    return parser.parse_args()


def require_ready(report: dict[str, object], allow_partial: bool) -> None:
    mac = report["mac"]
    server = report["server"]
    mac_done = mac["results_count"] >= mac["target_results"]
    server_done = server["results_count"] >= server["target_results"]
    if allow_partial or (mac_done and server_done):
        return
    raise RuntimeError(
        "Melting Pot shards are not complete yet: "
        f"Mac {mac['results_count']}/{mac['target_results']}, "
        f"server {server['results_count']}/{server['target_results']}."
    )


def run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PAPER_ROOT.parents[0],
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def read_remote_text(
    credential: monitor.SshCredential,
    remote_path: str,
    *,
    openssh_alias_required: bool = False,
) -> str:
    script = textwrap.dedent(
        """
        import base64
        import json
        import pathlib

        path = pathlib.Path(__REMOTE_PATH__)
        if not path.exists():
            raise FileNotFoundError(str(path))
        data = path.read_text(encoding="utf-8", errors="replace")
        print(json.dumps({
            "path": str(path),
            "content_b64": base64.b64encode(data.encode("utf-8")).decode("ascii"),
        }))
        """
    ).replace("__REMOTE_PATH__", json.dumps(remote_path))

    if openssh_alias_required:
        alias_meta = monitor.inspect_local_ssh_alias(credential.alias)
        if alias_meta and alias_meta.get("configured") and credential.alias:
            payload = monitor.exec_remote_python_openssh(credential.alias, script, timeout=120)
            return base64.b64decode(payload["content_b64"]).decode("utf-8")

    if credential.label == "ysh-server" and credential.proxy_jump:
        ssh_args: list[str] = ["-J", credential.proxy_jump]
        if credential.port != 22:
            ssh_args.extend(["-p", str(credential.port)])
        payload = monitor.exec_remote_python_openssh(
            f"{credential.user}@{credential.host}",
            script,
            timeout=120,
            ssh_args=ssh_args,
        )
        return base64.b64decode(payload["content_b64"]).decode("utf-8")

    payload = monitor.exec_remote_python(
        credential,
        script,
        attempts=2,
        connect_timeout=15,
        auth_timeout=20,
        banner_timeout=20,
        exec_timeout=120,
    )
    return base64.b64decode(payload["content_b64"]).decode("utf-8")


def write_snapshot(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sync_remote_snapshots(report: dict[str, object], mac_output: Path, server_output: Path) -> None:
    mac_credential = monitor.load_ssh_credential("mx-macbuild-mac-studio")
    server_credential = monitor.load_ssh_credential("ysh-server")

    mac_text = read_remote_text(
        mac_credential,
        report["mac"]["results_path"],
    )
    server_text = read_remote_text(
        server_credential,
        report["server"]["results_path"],
        openssh_alias_required=True,
    )

    json.loads(mac_text)
    json.loads(server_text)
    write_snapshot(mac_output, mac_text)
    write_snapshot(server_output, server_text)


def merge_and_integrate(mac_output: Path, server_output: Path, merged_output: Path) -> tuple[str, str]:
    merge_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "merge_meltingpot_results.py"),
        str(mac_output),
        str(server_output),
        "--output",
        str(merged_output),
    ]
    merge_proc = run_subprocess(merge_cmd)
    if merge_proc.returncode != 0:
        raise RuntimeError(merge_proc.stderr.strip() or merge_proc.stdout.strip() or "merge failed")

    integrate_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "integrate_meltingpot_results.py"),
        "--cleanup",
        str(merged_output),
    ]
    integrate_proc = run_subprocess(integrate_cmd)
    if integrate_proc.returncode != 0:
        raise RuntimeError(integrate_proc.stderr.strip() or integrate_proc.stdout.strip() or "integration failed")

    return merge_proc.stdout.strip(), integrate_proc.stdout.strip()


def summarize(report: dict[str, object]) -> str:
    return (
        f"Mac {report['mac']['results_count']}/{report['mac']['target_results']}, "
        f"server {report['server']['results_count']}/{report['server']['target_results']}, "
        f"overall ETA {monitor.format_eta(report['overall_eta_kst'])}"
    )


def main() -> None:
    args = parse_args()
    report = monitor.collect_report()
    mac_output = Path(args.mac_output)
    server_output = Path(args.server_output)
    merged_output = Path(args.merged_output)

    require_ready(report, args.allow_partial)

    if not args.skip_fetch:
        sync_remote_snapshots(report, mac_output, server_output)

    if args.sync_only:
        print("Synced remote shard snapshots.")
        print(f"- status: {summarize(report)}")
        print(f"- mac_snapshot: {mac_output}")
        print(f"- server_snapshot: {server_output}")
        return

    merge_stdout, integrate_stdout = merge_and_integrate(mac_output, server_output, merged_output)
    print("Melting Pot finalization succeeded.")
    print(f"- status: {summarize(report)}")
    print(f"- mac_snapshot: {mac_output}")
    print(f"- server_snapshot: {server_output}")
    print(f"- merged_output: {merged_output}")
    if merge_stdout:
        print("- merge:")
        print(merge_stdout)
    if integrate_stdout:
        print("- integrate:")
        print(integrate_stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
