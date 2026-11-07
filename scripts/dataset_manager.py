"""Dataset manager CLI.

Examples (PowerShell, from the repository root, venv active):
    python -m scripts.dataset_manager list
    python -m scripts.dataset_manager inspect --dataset iu_xray --check-remote
    python -m scripts.dataset_manager download --dataset iu_xray --metadata-only
    python -m scripts.dataset_manager download --dataset iu_xray --yes
    python -m scripts.dataset_manager import --dataset nih_cxr14 --source-dir D:\\data\\nih
    python -m scripts.dataset_manager verify  --dataset iu_xray    # archives you placed yourself
    python -m scripts.dataset_manager prepare --dataset iu_xray    # verifies + extracts placed archives
    python -m scripts.dataset_manager validate --dataset iu_xray
    python -m scripts.dataset_manager activate --dataset iu_xray
    python -m scripts.dataset_manager status
    python -m scripts.dataset_manager cleanup --dataset iu_xray
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.dataset_ops import ConfirmationRequired, DatasetOpError, DatasetOps
from app.services.downloader import DownloadError
from app.services.torrent import TorrentError, read_torrent


def _progress():
    last = {"t": 0.0}

    def cb(name: str, done: int, total: int | None):
        now = time.monotonic()
        if now - last["t"] < 0.5 and (total is None or done < total):
            return
        last["t"] = now
        pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
        print(f"\r  {name}: {done / 1e6:9.1f} MB {pct}", end="", flush=True)
        if total and done >= total:
            print()

    return cb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.dataset_manager", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="Storage folder for downloads (default: settings data_root)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("status")
    p = sub.add_parser("inspect"); p.add_argument("--dataset", required=True)
    p.add_argument("--check-remote", action="store_true", help="HEAD request for current remote sizes")
    p = sub.add_parser("download"); p.add_argument("--dataset", required=True)
    p.add_argument("--metadata-only", action="store_true", help="Only the small report/metadata files")
    p.add_argument("--yes", action="store_true", help="Confirm a large download")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("import", help="Register a folder you downloaded yourself (gated datasets)")
    p.add_argument("--dataset", required=True); p.add_argument("--source-dir", type=Path, required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dataset", required=True)
    p.add_argument("--source-dir", type=Path, help="Import this folder first")
    p.add_argument("--field-map", type=Path, help="YAML {source column: normalized field} for the generic builder")
    p = sub.add_parser("validate"); p.add_argument("--dataset", required=True)
    p.add_argument("--no-image-check", action="store_true", help="Only check that image files exist")
    p.add_argument("--max-images", type=int, help="Open at most N images")
    p = sub.add_parser("verify", help="Check archives in the download folder (exact size, torrent piece hashes)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--torrent", type=Path, action="append", default=[],
                   help="An intact .torrent for one of the archives (repeatable); overrides the bundled one")
    p = sub.add_parser("activate"); p.add_argument("--dataset", required=True)
    p = sub.add_parser("cleanup"); p.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)

    ops = DatasetOps(get_settings(), args.root)
    try:
        if args.cmd == "list":
            for spec in ops.registry.values():
                auto = "automatic download" if spec.automated_download_allowed else f"{spec.strategy} access"
                size = f", ~{spec.documented_total_mb:,.0f} MB documented" if spec.documented_total_mb else ""
                print(f"{spec.id:15} {spec.display_name} [{auto}{size}]\n{'':15} {spec.official_source}")
        elif args.cmd == "status":
            for ds in ops.registry:
                st = ops.status(ds)
                print(f"{ds:15} download={st['download_state']:<14} prepared={st['prepared']!s:<5} "
                      f"validated={st['validation_ok']!s:<5} active={st['active']}"
                      + (f"  last_error={st['last_error']}" if st["last_error"] else ""))
        elif args.cmd == "inspect":
            print(json.dumps(ops.inspect(args.dataset, args.check_remote), indent=2))
        elif args.cmd == "download":
            try:
                res = ops.download(args.dataset, args.metadata_only, args.yes, args.dry_run, _progress())
            except ConfirmationRequired as exc:
                print(f"Confirmation needed: {exc}")
                if not sys.stdin.isatty():
                    return 2
                if input("Download now? [y/N] ").strip().lower() != "y":
                    print("Cancelled.")
                    return 1
                res = ops.download(args.dataset, args.metadata_only, True, False, _progress())
            print(json.dumps(res, indent=2))
        elif args.cmd == "import":
            print(json.dumps(ops.import_local(args.dataset, args.source_dir), indent=2))
        elif args.cmd == "prepare":
            options = {"field_map": str(args.field_map)} if args.field_map else {}
            meta = ops.prepare(args.dataset, args.source_dir, options)
            print(json.dumps({k: v for k, v in meta.items() if k != "build_issues"}, indent=2))
            if meta["build_issues"]:
                print(f"{len(meta['build_issues'])} build issue(s); first ones:")
                for issue in meta["build_issues"][:10]:
                    print(f"  - {issue['kind']}: {issue.get('case_id') or ''} {issue['detail']}")
            print(f"Next: python -m scripts.dataset_manager validate --dataset {args.dataset}")
        elif args.cmd == "validate":
            rep = ops.validate(args.dataset, not args.no_image_check, args.max_images)
            print(json.dumps({k: v for k, v in rep.items() if k != "issues"}, indent=2))
            print(f"{len(rep['issues'])} issue(s) recorded in {rep['report_files'][0]}")
            return 0 if rep["ok"] else 3
        elif args.cmd == "verify":
            overrides = {}
            for tp in args.torrent:
                overrides[read_torrent(tp).name] = tp
            all_ok = True
            for f in ops.spec(args.dataset).files:
                res = ops.check_archive(args.dataset, f.name, overrides.get(f.name))
                all_ok &= res["ok"]
                print(f"{f.name}: {'OK' if res['ok'] else 'FAILED'} - {res['detail']}")
            return 0 if all_ok else 3
        elif args.cmd == "activate":
            print(json.dumps(ops.activate(args.dataset), indent=2))
        elif args.cmd == "cleanup":
            print("\n".join(ops.cleanup_instructions(args.dataset)))
    except (DatasetOpError, DownloadError, TorrentError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
