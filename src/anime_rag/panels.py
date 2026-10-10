"""Build panel-level documents (image crop + associated dialogue text) from a
parsed Manga109-s Book, which are the unit of retrieval for the RAG index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .annotations import Book, Page


@dataclass(frozen=True)
class PanelDoc:
    id: str
    book: str
    page_index: int
    bbox: tuple[int, int, int, int]
    text: str
    characters: tuple[str, ...]
    image_path: Path


def _texts_in_frame(page: Page, frame_bbox) -> list[str]:
    hits = [t for t in page.texts if frame_bbox.contains_point(*t.bbox.center())]
    hits.sort(key=lambda t: (t.bbox.ymin, -t.bbox.xmin))
    return [t.text for t in hits if t.text]


def _characters_in_frame(page: Page, frame_bbox, characters: dict[str, str]) -> list[str]:
    char_ids = []
    for f in page.faces:
        if frame_bbox.contains_point(*f.bbox.center()) and f.character_id not in char_ids:
            char_ids.append(f.character_id)
    for b in page.bodies:
        if frame_bbox.contains_point(*b.bbox.center()) and b.character_id not in char_ids:
            char_ids.append(b.character_id)
    return [characters.get(cid, cid) for cid in char_ids]


def build_panels(book: Book, images_dir: Path, crops_dir: Path) -> list[PanelDoc]:
    """Crop one image per frame and pair it with the dialogue text and
    character names found inside that frame. Crops are cached on disk."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    panels: list[PanelDoc] = []

    for page in book.pages:
        if not page.frames:
            continue
        page_image_path = images_dir / f"{page.index:03d}.jpg"
        if not page_image_path.exists():
            continue
        page_image = None  # lazy-loaded only if a crop is missing

        for frame in page.frames:
            panel_id = f"{book.title}_{page.index:03d}_{frame.id}"
            crop_path = crops_dir / f"{panel_id}.jpg"
            if not crop_path.exists():
                if page_image is None:
                    page_image = Image.open(page_image_path).convert("RGB")
                box = (frame.bbox.xmin, frame.bbox.ymin, frame.bbox.xmax, frame.bbox.ymax)
                box = tuple(max(0, v) for v in box)
                page_image.crop(box).save(crop_path, "JPEG", quality=90)

            text = " / ".join(_texts_in_frame(page, frame.bbox))
            characters = tuple(_characters_in_frame(page, frame.bbox, book.characters))

            panels.append(
                PanelDoc(
                    id=panel_id,
                    book=book.title,
                    page_index=page.index,
                    bbox=(frame.bbox.xmin, frame.bbox.ymin, frame.bbox.xmax, frame.bbox.ymax),
                    text=text,
                    characters=characters,
                    image_path=crop_path,
                )
            )

    return panels
