#!/usr/bin/env python3
"""
01_download_unifoil_raw.py
==========================
Step 1 of the project. Downloads ONLY the raw UniFoil files the project uses, straight out of the
Hugging Face archives, without changing them. Nothing is parsed, converted or calculated.

A zip keeps a table of contents at its end, so the script reads that table with HTTP range
requests and then fetches only the files it needs, instead of the whole 63-267 GB archives.

Source    https://huggingface.co/datasets/rkanchi/UniFoil   (licence CC BY-SA 4.0)
Revision  6724bb9691e866655446543aa3ee16cf78f26b5f  (the 26-27 July 2026 re-upload)

WHAT IS TAKEN (the transitional NLF cases: 63,373 cases on 4,618 airfoils) and where it is used
  transi/airfoil_data_from_simulations.zip   whole archive, unchanged (343 MB)
      <airfoil>_case_<k>_analysis.csv        Mach, AoA, Re, CL, CD             -> 02 preprocessing
      <airfoil>_case_<k>_convergence.pkl     solver history (totalR, ...)      -> 02 convergence check
  transi/output_transi.zip                   per case, these six files (12 GB of 267 GB):
      transiLoc.dat                          solver's transition location      -> 02, 03
      nfactor_ts.dat                         N-factor curves (TS waves)        -> 02, 03
      bledge_info.dat                        boundary-layer edge data          -> 02, 03 (separation point)
      nfactor_cf.dat, allSideOut.dat, tscf_info.dat   checked in 00_raw_data_dictionary.py only
  transi/output_aero_cutout1-6.zip           per case: the *_slices.dat surface file (6 GB)
                                             Cp and skin friction round the airfoil -> 02, 04 drag split

WHAT IS LEFT OUT, BECAUSE THE PROJECT DOES NOT USE IT
  the surface and volume CGNS files and the boundary-layer profile files of the transitional
  cases, the fully turbulent NLF twins (nlf_turb_cutout, about 726 GB of volume files) and the
  30,000 fully turbulent airfoils (ft_turb_cutout_1-4). --list shows what each archive holds.

HOW EACH FILE IS CHECKED
  A zip stores the size and a CRC-32 checksum of every file inside it. Every file taken out is
  checked against both, so it is byte-for-byte the file the dataset authors put in the archive.
  The whole archive is checked against the SHA-256 checksum Hugging Face publishes.

HOW THE FILES ARE STORED
  About 440,000 small files are very slow to copy to Google Drive one by one, so they are kept,
  with their original names and contents, in shard zips: raw/<archive name>/shard_000.zip, ...
  Unzipping a shard gives the original files exactly. manifest_files.csv.gz lists every file and
  its shard; SOURCE.txt records the revision, the licence and what was taken.

USAGE (Colab)
  !pip install -q -U "huggingface_hub[hf_xet]" requests
  !python 01_download_unifoil_raw.py --list          # only shows what would be taken
  !python 01_download_unifoil_raw.py --backup /content/drive/MyDrive/unifoil_project/raw_data
  About 17 GB is downloaded. If it stops (for example Colab disconnects), run the same command
  again: finished shards are kept, and with --backup they are restored from Drive.
"""


import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import struct
import sys
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

# ---- settings ---------------------------------------------------------------
REPO_ID = "rkanchi/UniFoil"
REVISION = "6724bb9691e866655446543aa3ee16cf78f26b5f"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"

WHOLE = {   # small archive kept exactly as it is
    "transi": ["transi/airfoil_data_from_simulations.zip"],
}
PARTIAL = {  # big archives: only the needed files are taken out
    "transi": ["transi/output_transi.zip"] + [f"transi/output_aero_cutout{i}.zip" for i in range(1, 7)],
}
TRANSI_FILES = ("transiLoc.dat", "nfactor_ts.dat", "nfactor_cf.dat",
                "bledge_info.dat", "allSideOut.dat", "tscf_info.dat")
SHARD_BYTES = 400 * 10**6      # up to 400 MB of files per shard
SHARD_FILES = 20000            # ... or 20,000 files
GAP = 128 * 1024               # merge neighbouring files closer than this into one request
MAX_RANGE = 8 * 1024 * 1024    # largest single range request
SPACE_MARGIN = 2 * 10**9
HEADERS = {"User-Agent": "Mozilla/5.0 UniFoil-research/1.0"}
if os.environ.get("HF_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
CASE_RE = re.compile(r"airfoil_\d+\w*?case_\d+")


def human(n):
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{int(n)} B"


def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stem(rel):
    return posixpath.basename(rel)[:-4]          # "output_transi.zip" -> "output_transi"


def safe_copy(src, dst):
    """copy through a temporary name, so a half-finished copy never looks complete"""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".copying"
    shutil.copyfile(src, tmp)
    if os.path.getsize(tmp) != os.path.getsize(src):
        os.remove(tmp)
        raise IOError(f"copy of {os.path.basename(src)} was incomplete")
    os.replace(tmp, dst)


# =============================================================================
# REMOTE ACCESS (HTTP range requests, as in build_final_dataset.py)
# =============================================================================
class Remote:
    """Thread-safe range access to one archive at the pinned revision."""

    def __init__(self, rel):
        self.rel, self.url = rel, f"{BASE_URL}/{rel}"
        self._tls = threading.local()
        self._lock = threading.Lock()
        self.final, self.size = self._resolve()

    def _session(self):
        s = getattr(self._tls, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update(HEADERS)
            self._tls.s = s
        return s

    def _resolve(self):
        r = self._session().get(self.url, headers={"Range": "bytes=0-0"}, stream=True,
                                timeout=120, allow_redirects=True)
        code, final, cr = r.status_code, r.url, r.headers.get("Content-Range", "")
        r.close()
        if code != 206 or "/" not in cr:
            raise IOError(f"cannot open {self.rel} (HTTP {code})")
        return final, int(cr.split("/")[-1])

    def fetch(self, start, length):
        end = min(start + length, self.size) - 1
        wait = 1.0
        for _ in range(8):
            try:
                r = self._session().get(self.final, timeout=180,
                                        headers={"Range": f"bytes={start}-{end}"})
                if r.status_code == 206:
                    return r.content
                if r.status_code in (403, 410):              # signed link expired
                    with self._lock:
                        self.final, _ = self._resolve()
                    continue
                if r.status_code == 429:
                    wait = max(wait, float(r.headers.get("Retry-After", wait)))
            except requests.RequestException:
                pass
            time.sleep(wait)
            wait = min(wait * 2, 60)
        raise IOError(f"range request failed: {self.rel} at byte {start}")


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a Remote, used only to read the zip's table of contents."""

    BLOCK = 1 << 20

    def __init__(self, remote):
        self.r, self.pos, self.b0, self.buf = remote, 0, 0, b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.r.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.r.size - self.pos
        if n <= 0 or self.pos >= self.r.size:
            return b""
        if not (self.b0 <= self.pos and self.pos + n <= self.b0 + len(self.buf)):
            self.b0, self.buf = self.pos, self.r.fetch(self.pos, max(n, self.BLOCK))
        out = self.buf[self.pos - self.b0: self.pos - self.b0 + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


# =============================================================================
# TABLE OF CONTENTS
# =============================================================================
INDEX_FIELDS = ["name", "offset", "csize", "usize", "method", "crc", "date_time"]


def load_or_build_index(rel, remote, dest, backup):
    """name, offset, sizes, compression method and CRC-32 of every file in the archive"""
    base = f"index/{stem(rel)}"
    for root in [dest] + ([backup] if backup else []):
        meta_p, csv_p = os.path.join(root, base + ".json"), os.path.join(root, base + ".csv.gz")
        if os.path.exists(meta_p) and os.path.exists(csv_p):
            meta = json.load(open(meta_p))
            if meta.get("archive_size") == remote.size and meta.get("revision") == REVISION:
                with gzip.open(csv_p, "rt", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                for r in rows:
                    for k in ("offset", "csize", "usize", "method", "crc"):
                        r[k] = int(r[k])
                # keep a copy in both places
                for other in [dest] + ([backup] if backup else []):
                    if other != root and not os.path.exists(os.path.join(other, base + ".csv.gz")):
                        safe_copy(meta_p, os.path.join(other, base + ".json"))
                        safe_copy(csv_p, os.path.join(other, base + ".csv.gz"))
                return rows
    print(f"    reading the table of contents of {rel} ...", flush=True)
    rows = []
    with zipfile.ZipFile(RangeFile(remote)) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            rows.append(dict(name=i.filename, offset=i.header_offset, csize=i.compress_size,
                             usize=i.file_size, method=i.compress_type, crc=i.CRC,
                             date_time="%04d-%02d-%02d %02d:%02d:%02d" % i.date_time))
    os.makedirs(os.path.join(dest, "index"), exist_ok=True)
    csv_p, meta_p = os.path.join(dest, base + ".csv.gz"), os.path.join(dest, base + ".json")
    with gzip.open(csv_p, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    json.dump({"archive": rel, "archive_size": remote.size, "revision": REVISION,
               "files": len(rows), "read_utc": now_utc()}, open(meta_p, "w"), indent=1)
    if backup:
        for p in (csv_p, meta_p):
            safe_copy(p, os.path.join(backup, os.path.relpath(p, dest)))
    return rows


def kind_of(name):
    """file type label for the summary, e.g. '<case>_slices.dat' or 'transiLoc.dat'"""
    base = CASE_RE.sub("<case>", posixpath.basename(name))
    return re.sub(r"\d+", "#", base)


def select(rel, rows):
    """which files to take out of this archive, by NAME only"""
    base = posixpath.basename(rel)
    if base == "output_transi.zip":
        return [r for r in rows if posixpath.basename(r["name"]) in TRANSI_FILES], \
            "names " + ", ".join(TRANSI_FILES)
    if base.startswith("output_aero_cutout"):
        return [r for r in rows if r["name"].endswith("_slices.dat")], "names ending _slices.dat"
    return [], "not used"


def summarise(rel, rows, chosen, rule, dest, backup=None):
    kinds = {}
    chosen_names = {r["name"] for r in chosen}
    for r in rows:
        k = kinds.setdefault(kind_of(r["name"]), [0, 0, 0])
        k[0] += 1
        k[1] += r["usize"]
        k[2] += r["name"] in chosen_names
    total = sum(r["usize"] for r in rows)
    taken = sum(r["usize"] for r in chosen)
    print(f"\n  {rel}: {len(rows):,} files, {human(total)} uncompressed")
    print(f"    taking {len(chosen):,} files, {human(taken)}  ({rule})")
    print(f"    leaving out {human(total - taken)}")
    for k, (n, size, sel) in sorted(kinds.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"      {'TAKE' if sel else '    '} {n:>8,} x {k:<40} {human(size):>10}")
    if len(kinds) > 12:
        print(f"      ... and {len(kinds) - 12} more kinds of file")
    os.makedirs(os.path.join(dest, "index"), exist_ok=True)
    path = os.path.join(dest, "index", f"contents_{stem(rel)}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "files", "uncompressed_bytes", "files_taken"])
        for k, (n, size, sel) in sorted(kinds.items(), key=lambda kv: -kv[1][1]):
            w.writerow([k, n, size, sel])
    if backup:
        safe_copy(path, os.path.join(backup, "index", os.path.basename(path)))
    return total, taken


# =============================================================================
# TAKING FILES OUT (raw bytes, checked against the archive's CRC-32)
# =============================================================================
def coalesce(rows):
    rows = sorted(rows, key=lambda r: r["offset"])
    groups, cur = [], None
    for r in rows:
        start = r["offset"]
        end = start + 30 + len(r["name"].encode()) + 1024 + r["csize"]
        if cur and start - cur[1] <= GAP and max(end, cur[1]) - cur[0] <= MAX_RANGE:
            cur[1] = max(cur[1], end)
            cur[2].append(r)
        else:
            if cur:
                groups.append(cur)
            cur = [start, end, [r]]
    if cur:
        groups.append(cur)
    return groups


def unpack_and_check(data, r):
    if r["method"] == 8:
        raw = zlib.decompress(data, -15)
    elif r["method"] == 0:
        raw = data
    else:
        raise IOError(f"unsupported compression method {r['method']} in {r['name']}")
    if len(raw) != r["usize"]:
        raise IOError(f"size mismatch in {r['name']}")
    if (zlib.crc32(raw) & 0xFFFFFFFF) != r["crc"]:
        raise IOError(f"CRC-32 mismatch in {r['name']}")
    return raw


def member_from_blob(blob, base, r):
    rel = r["offset"] - base
    if rel + 30 > len(blob):
        raise ValueError("short")
    sig = struct.unpack("<I", blob[rel:rel + 4])[0]
    if sig != 0x04034B50:
        raise IOError(f"bad local header for {r['name']}")
    nlen, elen = struct.unpack("<HH", blob[rel + 26:rel + 30])
    ds = rel + 30 + nlen + elen
    if ds + r["csize"] > len(blob):
        raise ValueError("short")
    return unpack_and_check(blob[ds:ds + r["csize"]], r)


def member_alone(remote, r):
    head = remote.fetch(r["offset"], 30)
    nlen, elen = struct.unpack("<HH", head[26:30])
    return unpack_and_check(remote.fetch(r["offset"] + 30 + nlen + elen, r["csize"]), r)


def fetch_group(remote, group):
    start, end, rows = group
    blob = remote.fetch(start, end - start)
    out = {}
    for r in rows:
        try:
            out[r["name"]] = member_from_blob(blob, start, r)
        except ValueError:
            out[r["name"]] = member_alone(remote, r)
    return out


def plan_shards(chosen):
    shards, cur, size = [], [], 0
    for r in sorted(chosen, key=lambda r: r["offset"]):
        if cur and (size + r["usize"] > SHARD_BYTES or len(cur) >= SHARD_FILES):
            shards.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += r["usize"]
    if cur:
        shards.append(cur)
    return shards


def shard_done(marker, rows):
    if not os.path.exists(marker):
        return False
    m = json.load(open(marker))
    return (m.get("files") == len(rows) and m.get("first") == rows[0]["name"]
            and m.get("last") == rows[-1]["name"])


def run_shard(remote, rel, sid, rows, dest, backup, workers):
    folder = os.path.join(dest, "raw", stem(rel))
    zpath = os.path.join(folder, f"shard_{sid:03d}.zip")
    marker = zpath[:-4] + ".json"
    listing = zpath[:-4] + "_files.csv"
    if os.path.exists(zpath) and shard_done(marker, rows):
        return "kept"
    if backup:
        bz = os.path.join(backup, os.path.relpath(zpath, dest))
        if os.path.exists(bz) and shard_done(bz[:-4] + ".json", rows):
            for p in (bz, bz[:-4] + ".json", bz[:-4] + "_files.csv"):
                safe_copy(p, os.path.join(dest, os.path.relpath(p, backup)))
            return "restored from backup"

    got = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_group, remote, g) for g in coalesce(rows)]
        for f in as_completed(futs):
            got.update(f.result())

    os.makedirs(folder, exist_ok=True)
    tmp = zpath + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r in rows:
            dt_ = tuple(int(v) for v in re.split(r"[- :]", r["date_time"]))
            z.writestr(zipfile.ZipInfo(r["name"], date_time=dt_), got[r["name"]])
    os.replace(tmp, zpath)
    with open(listing, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["archive", "name", "bytes", "crc32", "shard"])
        for r in rows:
            w.writerow([rel, r["name"], r["usize"], f"{r['crc']:08x}", os.path.basename(zpath)])
    json.dump({"archive": rel, "files": len(rows), "bytes": sum(r["usize"] for r in rows),
               "first": rows[0]["name"], "last": rows[-1]["name"], "written_utc": now_utc()},
              open(marker, "w"), indent=1)
    if backup:
        for p in (zpath, listing, marker):
            safe_copy(p, os.path.join(backup, os.path.relpath(p, dest)))
    return "downloaded"


# =============================================================================
# WHOLE ARCHIVES
# =============================================================================
def published_sha256(paths):
    """SHA-256 that Hugging Face publishes for each file (empty if unavailable)"""
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(REPO_ID, paths, repo_type="dataset", revision=REVISION)
        return {f.path: (f.lfs.sha256 if getattr(f, "lfs", None) else "") for f in info}
    except Exception as e:
        print(f"    (could not read published checksums: {type(e).__name__}; sizes will still be checked)")
        return {}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def download_whole(rel, dest, backup, expected_sha):
    target = os.path.join(dest, "archives", rel)
    marker = target + ".json"
    remote = Remote(rel)
    if os.path.exists(target) and os.path.exists(marker) and os.path.getsize(target) == remote.size:
        if backup:   # make sure Drive has it too (an earlier run may have been without --backup)
            bt = os.path.join(backup, "archives", rel)
            if not (os.path.exists(bt) and os.path.exists(bt + ".json") and os.path.getsize(bt) == remote.size):
                safe_copy(target, bt)
                safe_copy(marker, bt + ".json")
                return "kept, and copied to the backup folder"
        return "kept"
    if backup:
        bt = os.path.join(backup, "archives", rel)
        if os.path.exists(bt) and os.path.exists(bt + ".json") and os.path.getsize(bt) == remote.size:
            safe_copy(bt, target)
            safe_copy(bt + ".json", marker)
            return "restored from backup"
    os.makedirs(os.path.dirname(target), exist_ok=True)
    part = target + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > remote.size:                             # leftover from something else: start again
        os.remove(part)
        have = 0
    if have < remote.size:
        s = requests.Session()
        s.headers.update(HEADERS)
        with s.get(remote.url, headers={"Range": f"bytes={have}-"} if have else {},
                   stream=True, timeout=3600, allow_redirects=True) as r:
            if have and r.status_code == 200:          # server did not resume: start again
                have = 0
            r.raise_for_status()
            with open(part, "ab" if have else "wb") as fh:
                for block in r.iter_content(1 << 23):
                    fh.write(block)
    if os.path.getsize(part) != remote.size:
        raise IOError(f"{rel} is incomplete - run the script again to resume")
    sha = sha256_of(part)
    if expected_sha and sha != expected_sha:
        os.remove(part)
        raise IOError(f"{rel}: SHA-256 does not match the published one - run again")
    os.replace(part, target)
    json.dump({"archive": rel, "bytes": remote.size, "sha256": sha,
               "sha256_published": expected_sha or "not available",
               "revision": REVISION, "downloaded_utc": now_utc()}, open(marker, "w"), indent=1)
    if backup:
        safe_copy(target, os.path.join(backup, "archives", rel))
        safe_copy(marker, os.path.join(backup, "archives", rel + ".json"))
    return "downloaded"


# =============================================================================
# RECORDS (always about everything taken so far, local folder and backup together)
# =============================================================================
def all_taken_rows(dest, backup):
    """every file taken so far, from the shard file lists in the local folder and the backup folder"""
    seen, rows = set(), []
    for root in [dest] + ([backup] if backup else []):
        for folder, _, names in sorted(os.walk(os.path.join(root, "raw"))):
            for f in sorted(names):
                if not f.endswith("_files.csv"):
                    continue
                with open(os.path.join(folder, f), newline="") as fh:
                    for r in csv.DictReader(fh):
                        key = (r["archive"], r["name"])
                        if key not in seen:
                            seen.add(key)
                            rows.append(r)
    return rows


def archive_totals(rel, dest, backup):
    """(files, uncompressed bytes) of a whole archive, from its saved table of contents"""
    for root in [dest] + ([backup] if backup else []):
        p = os.path.join(root, "index", stem(rel) + ".csv.gz")
        if os.path.exists(p):
            n = total = 0
            with gzip.open(p, "rt", newline="") as fh:
                for r in csv.DictReader(fh):
                    n += 1
                    total += int(r["usize"])
            return n, total
    return None, None


def earlier_rules(dest, backup):
    """the selection rule of each archive, as an earlier run wrote it into SOURCE.txt"""
    rules = {}
    for root in ([backup] if backup else []) + [dest]:          # the local copy wins
        p = os.path.join(root, "SOURCE.txt")
        if os.path.exists(p):
            with open(p) as fh:
                for line in fh:
                    m = re.match(r"\s+(\S+\.zip): [\d,]+ of [\d,]+ files, .+? of .+? \((.*)\)\s*$", line)
                    if m:
                        rules[m.group(1)] = m.group(2)
    return rules


def write_records(dest, backup, summary):
    rows = all_taken_rows(dest, backup)
    man = os.path.join(dest, "manifest_files.csv.gz")
    with gzip.open(man, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["archive", "name", "bytes", "crc32", "shard"])
        w.writeheader()
        w.writerows(rows)

    per_archive = {}
    for r in rows:
        n, b = per_archive.get(r["archive"], (0, 0))
        per_archive[r["archive"]] = (n + 1, b + int(r["bytes"]))
    rules = earlier_rules(dest, backup)
    order = [rel for part in PARTIAL.values() for rel in part]
    archives = sorted(per_archive, key=lambda a: (order.index(a) if a in order else len(order), a))

    lines = ["UniFoil raw files for the project, taken unchanged from Hugging Face",
             f"Repository : https://huggingface.co/datasets/{REPO_ID}",
             f"Revision   : {REVISION}",
             "Licence    : CC BY-SA 4.0",
             "Paper      : Kanchi, R.S., Melanson, B., Somasekharan, N., Pan, S. and He, S. (2025) "
             "'UniFoil: a universal dataset of airfoils in transitional and turbulent regimes for "
             "subsonic and transonic flows', NeurIPS 2025 Datasets and Benchmarks Track.",
             f"Written    : {now_utc()}", "",
             "Whole archives (unchanged, SHA-256 in archives/*.json):"]
    lines += [f"  {rel}" for part in WHOLE.values() for rel in part
              if any(os.path.exists(os.path.join(root, "archives", rel))
                     for root in [dest] + ([backup] if backup else []))]
    lines += ["", "Files taken out of the big archives (checked against each archive's CRC-32):"]
    for rel in archives:
        n_taken, taken = per_archive[rel]
        if rel in summary:
            n_all, total, rule = summary[rel][0], summary[rel][1], summary[rel][4]
        else:
            (n_all, total), rule = archive_totals(rel, dest, backup), rules.get(rel, "see index/contents_*.csv")
        all_txt = f"{n_all:,}" if n_all is not None else "?"
        total_txt = human(total) if total is not None else "?"
        lines.append(f"  {rel}: {n_taken:,} of {all_txt} files, {human(taken)} of {total_txt} ({rule})")
    lines += ["", f"Every taken file is listed in manifest_files.csv.gz ({len(rows):,} files).",
              "What each archive contains, and what was left out: index/contents_*.csv"]
    src = os.path.join(dest, "SOURCE.txt")
    open(src, "w").write("\n".join(lines) + "\n")
    if backup:
        for p in (man, src):
            safe_copy(p, os.path.join(backup, os.path.basename(p)))
    print("\n" + "\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Download only the raw UniFoil files the project uses.")
    p.add_argument("--dest", default="/content/unifoil_raw" if os.path.isdir("/content") else "unifoil_raw")
    p.add_argument("--backup", help="second folder (e.g. on Google Drive) that keeps a copy of every finished part")
    p.add_argument("--list", action="store_true", help="only show what each archive contains")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    dest = os.path.abspath(os.path.expanduser(args.dest))
    backup = os.path.abspath(os.path.expanduser(args.backup)) if args.backup else None
    os.makedirs(dest, exist_ok=True)
    if backup and backup.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        sys.exit("Google Drive is not mounted. In Colab run: from google.colab import drive; "
                 "drive.mount('/content/drive')")

    print(f"UniFoil, revision {REVISION}")
    print(f"saving to {dest}" + (f", backup copy in {backup}" if backup else ""))

    # ---- 1. tables of contents ---------------------------------------------------------
    print("\n[1] tables of contents")
    plan, summary, remotes = {}, {}, {}
    for rel in PARTIAL["transi"]:
        remotes[rel] = Remote(rel)
        rows = load_or_build_index(rel, remotes[rel], dest, backup)
        plan[rel], rule = select(rel, rows)
        total, taken = summarise(rel, rows, plan[rel], rule, dest, backup)
        summary[rel] = (len(rows), total, len(plan[rel]), taken, rule)
    whole_size = sum(Remote(rel).size for rel in WHOLE["transi"])
    need = sum(v[3] for v in summary.values()) + whole_size
    print(f"\nto download: about {human(need)} (files taken out, before compression in the "
          f"shards, plus {human(whole_size)} of whole archive)")
    if args.list:
        print("--list: nothing downloaded. The summaries are saved in", os.path.join(dest, "index"))
        return
    empty = [rel for rel in plan if not plan[rel]]
    if empty:
        sys.exit("No files matched the selection in: " + ", ".join(empty))
    free = shutil.disk_usage(dest).free
    if free < need + SPACE_MARGIN:
        sys.exit(f"Not enough space in {dest}: about {human(need)} needed, {human(free)} free.")
    if backup:
        os.makedirs(backup, exist_ok=True)
        if shutil.disk_usage(backup).free < need:
            print("WARNING: the backup folder may not have enough free space; some copies may fail.")

    # ---- 2. whole archive ----------------------------------------------------------------
    print("\n[2] whole archive")
    shas = published_sha256(WHOLE["transi"])
    for rel in WHOLE["transi"]:
        print(f"    {rel}: {download_whole(rel, dest, backup, shas.get(rel, ''))}", flush=True)

    # ---- 3. files taken out of the big archives -----------------------------------------
    print("\n[3] taking the needed files out of the big archives")
    for rel, chosen in plan.items():
        shards = plan_shards(chosen)
        print(f"  {rel}: {len(chosen):,} files in {len(shards)} shards")
        for sid, rows in enumerate(shards):
            t0 = time.time()
            try:
                state = run_shard(remotes[rel], rel, sid, rows, dest, backup, args.workers)
            except Exception as e:
                print(f"\n    shard_{sid:03d} stopped: {type(e).__name__}: {str(e)[:150]}")
                print("    Everything finished so far is saved. Run the same command again to resume.")
                return
            print(f"    shard_{sid:03d}: {len(rows):,} files, {state} ({time.time() - t0:.0f} s)", flush=True)

    write_records(dest, backup, summary)
    print("\nDone. No file was changed: every file matches the source checksum.")


if __name__ == "__main__":
    main()
