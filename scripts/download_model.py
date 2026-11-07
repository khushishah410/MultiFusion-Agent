"""Download TorchXRayVision weights into the project's model folder, after showing the size.

    python -m scripts.download_model               # weights from settings (densenet121-res224-all)
    python -m scripts.download_model --weights densenet121-res224-nih --yes
    python -m scripts.download_model --list
"""
from __future__ import annotations

import argparse
import sys

from app.core.config import get_settings
from app.services.downloader import DownloadError, download_file, remote_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights")
    ap.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    ap.add_argument("--list", action="store_true", help="List weight names known to torchxrayvision")
    args = ap.parse_args(argv)
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("torchxrayvision is not installed: pip install -r requirements-model.txt", file=sys.stderr)
        return 1
    if args.list:
        for name in sorted(k for k in xrv.models.model_urls if k.startswith(("densenet", "resnet"))):
            print(name)
        return 0
    s = get_settings()
    weights = args.weights or s.model_weights
    entry = xrv.models.model_urls.get(weights)
    if entry is None:
        print(f"Unknown weights {weights!r}; use --list.", file=sys.stderr)
        return 1
    url = entry["weights_url"]
    dest = s.model_path_abs / url.rsplit("/", 1)[-1]
    try:
        size = remote_size(url)
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Weights:  {weights}\nSource:   {url}\nSize:     {size / 1e6:.1f} MB" if size else f"Source: {url}")
    print(f"Target:   {dest}\nLicense:  TorchXRayVision is Apache-2.0; see the repository for the weights' terms.")
    if not args.yes and (not sys.stdin.isatty() or input("Download? [y/N] ").strip().lower() != "y"):
        print("Not downloaded.")
        return 1
    try:
        res = download_file(url, dest, expected_size=size)
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"{'Already present' if res.skipped_existing else 'Downloaded'}: {res.bytes} bytes, sha256 {res.sha256}")
    if weights != s.model_weights:
        print(f"Set MULTIFUSION_MODEL_WEIGHTS={weights} to use these weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
