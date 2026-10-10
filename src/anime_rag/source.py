"""Access to the hal-utokyo/Manga109-s dataset on the Hugging Face Hub.

The dataset is distributed as a single large zip file. Instead of downloading
the whole archive, we open it through ``HfFileSystem`` (which supports HTTP
range reads) and read only the entries we need, so indexing a handful of
books does not require pulling the full ~3GB+ archive to disk.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, HfFileSystem

REPO_ID = "hal-utokyo/Manga109-s"
REPO_TYPE = "dataset"


@dataclass(frozen=True)
class MangaZip:
    zf: zipfile.ZipFile
    root: str  # common top-level directory inside the zip, e.g. "Manga109s_released_2026_05_21/"


def _find_zip_filename() -> str:
    api = HfApi()
    info = api.dataset_info(REPO_ID)
    candidates = [s.rfilename for s in info.siblings if s.rfilename.endswith(".zip")]
    if not candidates:
        raise RuntimeError(f"No .zip file found in {REPO_ID}")
    return sorted(candidates)[-1]


def open_zip() -> MangaZip:
    fs = HfFileSystem()
    zip_filename = _find_zip_filename()
    path = f"datasets/{REPO_ID}/{zip_filename}"
    f = fs.open(path, "rb")
    zf = zipfile.ZipFile(f)
    first_name = zf.namelist()[0]
    root = first_name.split("/")[0] + "/"
    return MangaZip(zf=zf, root=root)


def list_books(mz: MangaZip) -> list[str]:
    prefix = f"{mz.root}annotations/"
    books = []
    for name in mz.zf.namelist():
        if name.startswith(prefix) and name.endswith(".xml"):
            books.append(name[len(prefix) : -len(".xml")])
    return sorted(books)


def book_is_extracted(book: str, dest_dir: Path) -> bool:
    ann = dest_dir / "annotations" / f"{book}.xml"
    img_dir = dest_dir / "images" / book
    return ann.exists() and img_dir.exists() and any(img_dir.iterdir())


def extract_book(mz: MangaZip, book: str, dest_dir: Path) -> None:
    """Extract one book's annotation XML and page images into dest_dir."""
    if book_is_extracted(book, dest_dir):
        return

    ann_dest = dest_dir / "annotations" / f"{book}.xml"
    ann_dest.parent.mkdir(parents=True, exist_ok=True)
    ann_src = f"{mz.root}annotations/{book}.xml"
    ann_dest.write_bytes(mz.zf.read(ann_src))

    img_prefix = f"{mz.root}images/{book}/"
    img_dest_dir = dest_dir / "images" / book
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    for name in mz.zf.namelist():
        if name.startswith(img_prefix) and not name.endswith("/"):
            page_name = name[len(img_prefix) :]
            (img_dest_dir / page_name).write_bytes(mz.zf.read(name))
