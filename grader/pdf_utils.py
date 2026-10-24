"""PDF → PNG conversion utilities."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_OK = True
except ImportError:
    _HEIF_OK = False


@dataclass
class PageImage:
    index: int
    pil_image: Image.Image
    png_bytes: bytes
    width: int
    height: int
    saved_path: Path | None = None


@dataclass
class PageStrip:
    """One horizontal slice of a page, plus the y-bounds needed to remap
    Gemini's strip-local coordinates back into full-page space.
    """
    page_index: int
    strip_index: int
    png_bytes: bytes
    y_start_norm: float
    y_end_norm: float


def partition_page(
    page: PageImage, n: int, overlap_frac: float = 0.05
) -> list[PageStrip]:
    """Split a page into n horizontal strips with optional overlap.

    Each strip's y_start_norm/y_end_norm describe its position within the
    original page in [0, 1] — used to remap question coords after Gemini
    grades the strip on its own.
    """
    if n <= 1:
        return [
            PageStrip(
                page_index=page.index,
                strip_index=0,
                png_bytes=page.png_bytes,
                y_start_norm=0.0,
                y_end_norm=1.0,
            )
        ]

    H = page.height
    base_h = 1.0 / n
    strips: list[PageStrip] = []
    for i in range(n):
        y0 = 0.0 if i == 0 else max(0.0, i * base_h - overlap_frac)
        y1 = 1.0 if i == n - 1 else min(1.0, (i + 1) * base_h + overlap_frac)
        top_px = int(y0 * H)
        bot_px = int(y1 * H)
        cropped = page.pil_image.crop((0, top_px, page.width, bot_px))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        strips.append(
            PageStrip(
                page_index=page.index,
                strip_index=i,
                png_bytes=buf.getvalue(),
                y_start_norm=top_px / H,
                y_end_norm=bot_px / H,
            )
        )
    return strips


def pdf_to_pages(pdf_path: str | Path, dpi: int = 200) -> list[PageImage]:
    """Convert a PDF to a list of PageImage objects (in memory)."""
    pdf_path = Path(pdf_path)
    pil_pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")

    pages: list[PageImage] = []
    for i, img in enumerate(pil_pages):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        pages.append(
            PageImage(
                index=i,
                pil_image=img,
                png_bytes=buf.getvalue(),
                width=img.width,
                height=img.height,
            )
        )
    return pages


def save_pages(pages: list[PageImage], out_dir: str | Path, prefix: str = "page") -> list[Path]:
    """Save pages to disk as `<prefix>_<n>_original.png` (1-indexed in filename)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for p in pages:
        path = out_dir / f"{prefix}_{p.index + 1}_original.png"
        p.pil_image.save(path, "PNG")
        p.saved_path = path
        paths.append(path)
    return paths


def pages_to_png_bytes(pages: list[PageImage]) -> list[bytes]:
    return [p.png_bytes for p in pages]


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}


def image_to_pages(image_path: str | Path) -> list[PageImage]:
    """Load a single PNG/JPG/HEIC as a one-page PageImage list."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"} and not _HEIF_OK:
        raise RuntimeError(
            f"HEIC/HEIF support requires `pillow-heif`. "
            f"Install with: pip install pillow-heif"
        )
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect phone orientation
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return [
        PageImage(
            index=0,
            pil_image=img,
            png_bytes=buf.getvalue(),
            width=img.width,
            height=img.height,
        )
    ]


def load_any(path: str | Path, dpi: int = 200) -> list[PageImage]:
    """Load a PDF, PNG, or JPG into a list of PageImage objects."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_pages(p, dpi=dpi)
    if suffix in IMAGE_SUFFIXES:
        return image_to_pages(p)
    raise ValueError(f"Unsupported file type: {suffix} ({p})")
