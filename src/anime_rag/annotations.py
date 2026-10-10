"""Parser for Manga109-s per-book annotation XML files."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def contains_point(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)


@dataclass(frozen=True)
class TextBox:
    id: str
    bbox: BBox
    text: str


@dataclass(frozen=True)
class Frame:
    id: str
    bbox: BBox


@dataclass(frozen=True)
class Face:
    id: str
    bbox: BBox
    character_id: str


@dataclass(frozen=True)
class Body:
    id: str
    bbox: BBox
    character_id: str


@dataclass(frozen=True)
class Page:
    index: int
    width: int
    height: int
    frames: list[Frame] = field(default_factory=list)
    texts: list[TextBox] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    bodies: list[Body] = field(default_factory=list)


@dataclass(frozen=True)
class Book:
    title: str
    characters: dict[str, str]  # character id -> name
    pages: list[Page]


def _bbox(el: ET.Element) -> BBox:
    return BBox(
        xmin=int(el.get("xmin")),
        ymin=int(el.get("ymin")),
        xmax=int(el.get("xmax")),
        ymax=int(el.get("ymax")),
    )


def parse_book(xml_path: Path) -> Book:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    title = root.get("title", xml_path.stem)

    characters = {
        char.get("id"): char.get("name")
        for char in root.findall("./characters/character")
    }

    pages: list[Page] = []
    for page_el in root.findall("./pages/page"):
        frames = [
            Frame(id=el.get("id"), bbox=_bbox(el)) for el in page_el.findall("frame")
        ]
        texts = [
            TextBox(id=el.get("id"), bbox=_bbox(el), text=el.text or "")
            for el in page_el.findall("text")
        ]
        faces = [
            Face(id=el.get("id"), bbox=_bbox(el), character_id=el.get("character"))
            for el in page_el.findall("face")
        ]
        bodies = [
            Body(id=el.get("id"), bbox=_bbox(el), character_id=el.get("character"))
            for el in page_el.findall("body")
        ]
        pages.append(
            Page(
                index=int(page_el.get("index")),
                width=int(page_el.get("width")),
                height=int(page_el.get("height")),
                frames=frames,
                texts=texts,
                faces=faces,
                bodies=bodies,
            )
        )

    return Book(title=title, characters=characters, pages=pages)
