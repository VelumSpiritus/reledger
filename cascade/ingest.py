"""Cascade level 0 — ingest: bytes in, features out.

Downloads, extracts, reads every file, bridges to (N, 8) features.
The largest file in the cascade: all fetchers and the full IO layer.

CC0 1.0 Universal — public domain.
"""
import sys
import os
import argparse
from collections import defaultdict

import numpy as np

from lib.bridge import bridge_signal, bridge_text, collect_open_class, collect_all_tokens


def load_sources(data_dir):
    """Load every file in a directory. Every file is bytes.

    Read the bytes, bridge them as signal → (N, 8).
    Text files also get the text bridge for semantic return.
    The engine compresses from there.
    """
    sources = defaultdict(list)
    vocab = {}
    all_toks = {}
    raw_texts = {}

    all_files = []
    for entry in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, entry)
        if os.path.isdir(fpath):
            for sub in sorted(os.listdir(fpath)):
                subpath = os.path.join(fpath, sub)
                if os.path.isfile(subpath):
                    all_files.append((entry, sub, subpath))
        elif os.path.isfile(fpath):
            all_files.append((entry, entry, fpath))

    if not all_files:
        return sources, vocab

    for fi, (group, fname, fpath) in enumerate(all_files):
        if fname.endswith('.mat'):
            try:
                import scipy.io
                d = scipy.io.loadmat(fpath)
                key = [k for k in d.keys() if not k.startswith('__')][0]
                arr = d[key]
                if hasattr(arr[0, 0], 'dtype') and hasattr(arr[0, 0].dtype, 'names') and arr[0, 0].dtype.names:
                    s = arr[0, 0]
                    if 'Y' in s.dtype.names:
                        sig = s['Y'][0, 6]['Data'].flatten()
                    else:
                        sig = s[list(s.dtype.names)[0]].flatten()
                else:
                    sig = arr.flatten()
                feats = bridge_signal(sig)
                if len(feats) > 0:
                    sources[group].append((fname, feats))
                    continue
            except Exception:
                pass

        try:
            raw_bytes = open(fpath, 'rb').read()
        except Exception:
            continue
        if len(raw_bytes) < 4:
            continue

        sig = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64)
        sig_feats = bridge_signal(sig)

        text = raw_bytes.decode('utf-8', errors='ignore')
        txt_feats = bridge_text(text) if text.strip() else np.empty((0, 8))
        if text.strip():
            vocab[(group, fname)] = collect_open_class(text)
            all_toks[(group, fname)] = collect_all_tokens(text)
            raw_texts[(group, fname)] = text

        parts = []
        if len(sig_feats) > 0:
            parts.append(sig_feats)
        if len(txt_feats) > 0:
            parts.append(txt_feats)
        if parts:
            feats = np.vstack(parts)
            sources[group].append((fname, feats))

        if (fi + 1) % 100 == 0:
            print(f"  loaded {fi+1}/{len(all_files)}...", flush=True)

    print(f"  {len(all_files)} files")
    return sources, vocab, all_toks, raw_texts


def ingest(data_dir):
    """Level 0: load every file, bridge to features, return live."""
    print("Loading sources...")
    sources, vocab, all_toks, raw_texts = load_sources(data_dir)
    if not sources:
        return None

    source_names = sorted(sources.keys())
    print(f"  Sources: {len(source_names)}")
    for s in source_names:
        print(f"    {s}: {len(sources[s])} files")

    all_raw = []
    file_ranges = []
    file_vocab = []
    file_all_tokens = []
    file_raw_texts = []
    file_sources = []
    file_names = []

    for src in source_names:
        for fname, feats in sources[src]:
            start = len(all_raw)
            all_raw.extend(feats.tolist())
            file_ranges.append((start, len(all_raw), src, fname))
            file_vocab.append(vocab.get((src, fname), []))
            file_all_tokens.append(all_toks.get((src, fname), []))
            file_raw_texts.append(raw_texts.get((src, fname), ""))
            file_sources.append(src)
            file_names.append(fname)

    return {
        "source_names": source_names,
        "all_raw": np.array(all_raw),
        "file_ranges": file_ranges,
        "file_vocab": file_vocab,
        "file_all_tokens": file_all_tokens,
        "file_raw_texts": file_raw_texts,
        "file_sources": file_sources,
        "file_names": file_names,
        "data_dir": data_dir,
        "vocab": vocab,
    }


# ── Source fetchers ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def fetch_source(source, dest):
    """Fetch data from any endpoint into dest."""
    import tempfile

    if source.startswith('file:///'):
        local = source[len('file:///'):]
        if os.path.isdir(local):
            return local
        if os.path.isfile(local):
            os.makedirs(dest, exist_ok=True)
            return _extract_or_place(local, dest)
        raise FileNotFoundError(f"Local path not found: {local}")

    resolved = os.path.abspath(source)
    if os.path.isdir(resolved):
        return resolved
    if os.path.isfile(resolved):
        os.makedirs(dest, exist_ok=True)
        return _extract_or_place(resolved, dest)

    os.makedirs(dest, exist_ok=True)
    from urllib.parse import urlparse
    url_path = urlparse(source).path
    _, ext = os.path.splitext(url_path)
    suffix = ext if ext else '_download'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name

    try:
        if source.startswith('s3://'):
            _fetch_s3(source, tmp_path)
        elif source.startswith('gs://'):
            _fetch_gcs(source, tmp_path)
        elif source.startswith(('http://', 'https://')):
            import urllib.request
            import ssl
            import shutil
            print(f"Downloading {source} ...")
            try:
                urllib.request.urlretrieve(source, tmp_path)
            except urllib.error.URLError:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                with urllib.request.urlopen(source, context=ctx) as resp, \
                     open(tmp_path, 'wb') as out:
                    shutil.copyfileobj(resp, out)
        else:
            raise ValueError(
                f"Not a local path and unknown scheme: {source}\n"
                "  Expected: http(s)://, s3://, gs://, file:///, "
                "or a path that exists on disk")

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"  Fetched {size_mb:.1f} MB")
        return _extract_or_place(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _extract_or_place(filepath, dest):
    """Extract an archive into dest, or place a plain file there."""
    import shutil
    name = os.path.basename(filepath).lower()

    if name.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(dest)
        print(f"  Extracted zip -> {dest}")
    elif any(name.endswith(ext)
             for ext in ('.tar.gz', '.tgz', '.tar.bz2', '.tar')):
        import tarfile
        with tarfile.open(filepath, 'r:*') as tf:
            tf.extractall(dest)
        print(f"  Extracted tar -> {dest}")
    else:
        fname = os.path.basename(filepath)
        shutil.copy2(filepath, os.path.join(dest, fname))
        print(f"  Placed {fname} in {dest}")

    _extract_nested(dest)
    return dest


def _extract_nested(directory):
    """Recursively extract any archives found inside a directory."""
    import zipfile
    import tarfile

    for root, dirs, files in os.walk(directory):
        for f in files:
            fpath = os.path.join(root, f)
            fl = f.lower()
            try:
                if fl.endswith('.zip') and zipfile.is_zipfile(fpath):
                    with zipfile.ZipFile(fpath, 'r') as zf:
                        zf.extractall(root)
                    os.unlink(fpath)
                    print(f"  Extracted nested {f}")
                    _extract_nested(root)
                    return
                elif any(fl.endswith(e) for e in
                         ('.tar.gz', '.tgz', '.tar.bz2', '.tar')):
                    with tarfile.open(fpath, 'r:*') as tf:
                        tf.extractall(root)
                    os.unlink(fpath)
                    print(f"  Extracted nested {f}")
                    _extract_nested(root)
                    return
            except Exception:
                pass


def _fetch_s3(uri, dest_path):
    """Download from S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError("s3:// requires boto3: pip install boto3")
    parts = uri[5:].split('/', 1)
    bucket, key = parts[0], parts[1] if len(parts) > 1 else ''
    print(f"Fetching s3://{bucket}/{key} ...")
    boto3.client('s3').download_file(bucket, key, dest_path)


def _fetch_gcs(uri, dest_path):
    """Download from Google Cloud Storage."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "gs:// requires google-cloud-storage: "
            "pip install google-cloud-storage")
    parts = uri[5:].split('/', 1)
    bucket_name, blob_name = parts[0], parts[1] if len(parts) > 1 else ''
    print(f"Fetching gs://{bucket_name}/{blob_name} ...")
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(dest_path)


def resolve_data_dir(source):
    """Resolve a source argument to a data directory path."""
    import hashlib
    from urllib.parse import urlparse
    parsed = urlparse(source)
    if parsed.scheme in ('http', 'https', 's3', 'gs'):
        slug = parsed.netloc + parsed.path
    else:
        slug = os.path.splitext(os.path.basename(source))[0] or source
    slug = slug.strip('/').replace('/', '_').replace('.', '_')
    if len(slug) > 60:
        slug = slug[:40] + '_' + hashlib.md5(slug.encode()).hexdigest()[:8]
    dest = os.path.join(DATA_DIR, slug)
    dest = fetch_source(source, dest)

    subdirs = [d for d in os.listdir(dest)
               if os.path.isdir(os.path.join(dest, d))]
    if len(subdirs) == 1:
        dest = os.path.join(dest, subdirs[0])

    return dest


def main():
    parser = argparse.ArgumentParser(
        description='Reledger — discover structural genres, re-file by what the data says')
    parser.add_argument('data_dir', nargs='?', default=None,
                        help='Directory containing data files')
    parser.add_argument('--temporal', action='store_true',
                        help='Enable drift detection '
                             '(files must be in time order)')
    parser.add_argument('--download', metavar='SOURCE',
                        help='Fetch data and run. Accepts: http(s):// URL, '
                             's3://bucket/key, gs://bucket/key, '
                             'local path, or file:/// URI')
    parser.add_argument('--output', metavar='DIR',
                        help='Write library to this directory '
                             '(default: library/ beside the data)')
    args = parser.parse_args()

    from cascade.cascade.cascade.reledger import reledger_run

    if args.download:
        data_dir = resolve_data_dir(args.download)
        reledger_run(data_dir, temporal=args.temporal,
                     output_dir=args.output)
    elif args.data_dir:
        reledger_run(args.data_dir, temporal=args.temporal,
                     output_dir=args.output)
    else:
        parser.error('either data_dir or --download is required')


if __name__ == "__main__":
    main()
