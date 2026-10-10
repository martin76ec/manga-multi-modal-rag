"""Rich/inquirer-based interactive REPL for `anime-rag ask`.

Menu-driven instead of the old text-prefix convention ('agent:', 'image:')
-- that parsing was a real source of bugs (e.g. case-sensitivity silently
routing a query to the wrong mode). Colors + spinners via rich, inline
panel thumbnails via rich-pixels (true-color ANSI half-blocks -- works in
any modern terminal, no special graphics-protocol support needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import inquirer
from PIL import Image, ImageOps
from rich.console import Console
from rich.text import Text
from rich_pixels import Pixels

from .agent import AgentResult, MrklIrcotAgent
from .embedder import ClipJapaneseEmbedder
from .indexing import Index
from .llm import ChatLLM
from .search import SearchResult, Searcher

# rich-pixels only offers two renderers (half-cell, 2x vertical resolution
# via unicode half-blocks, or full-cell at half that) -- no quadrant/higher
# resolution option. Manga line art + screentone dithering blurs into gray
# mush at small sizes, so legibility comes from going bigger and boosting
# contrast before downscaling, not from the renderer itself.
THUMBNAIL_SIZE = (64, 64)

MODE_SEARCH_TEXT = "Search by text"
MODE_SEARCH_IMAGE = "Search by image"
MODE_AGENT = "Ask the agent (MRKL+IRCoT, uses HF Inference)"
MODE_QUIT = "Quit"

console = Console()


def _source_page_path(meta: dict) -> Path:
    data_dir = Path(meta["image_path"]).parents[2]
    return data_dir / "images" / meta["book"] / f"{meta['page_index']:03d}.jpg"


def _render_thumbnail(image_path: Path) -> Pixels | None:
    try:
        img = Image.open(image_path).convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=2)
        img.thumbnail(THUMBNAIL_SIZE)
        return Pixels.from_image(img)
    except Exception:
        return None


def _print_result(rank: int, r: SearchResult, translation: str | None) -> None:
    m = r.meta
    chars = ", ".join(m["characters"]) if m["characters"] else "-"
    text = m["text"] or "-"
    text_score = f"{r.text_score:.3f}" if r.text_score is not None else "-"

    header = Text()
    header.append(f"#{rank} ", style="bold white")
    header.append(f"[{r.score:.3f}] ", style="bold green")
    header.append(f"{m['book']} ", style="bold cyan")
    header.append(f"p.{m['page_index']:03d} ", style="cyan")
    header.append(f"({m['id']})", style="dim")
    console.print(header)

    console.print(f"    img={r.image_score:.3f} text={text_score}  chars: [yellow]{chars}[/yellow]")
    console.print(f"    [dim]text (ja):[/dim] {text}")
    if translation:
        console.print(f"    [dim]text (en):[/dim] [italic]{translation}[/italic]")
    console.print(f"    [dim]page:[/dim] {_source_page_path(m)}")

    pixels = _render_thumbnail(Path(m["image_path"]))
    if pixels is not None:
        console.print(pixels)
    console.print()


def _print_results(results: list[SearchResult], translations: list[str | None]) -> None:
    if not results:
        console.print("[red]no results[/red]\n")
        return
    for rank, (r, t) in enumerate(zip(results, translations)):
        _print_result(rank, r, t)


def _print_agent_result(result: AgentResult, meta_by_id: dict) -> None:
    for i, step in enumerate(result.steps):
        console.rule(f"step {i}", style="dim")
        console.print(step.reply.strip())
        if step.query:
            console.print(f"\n[bold]Observation[/bold] (search: [italic]{step.query!r}[/italic]):")
            console.print(step.observation)
        console.print()

    console.rule(style="green")
    console.print(f"[bold green]Final Answer:[/bold green] {result.answer}")

    if result.seen_panel_ids:
        console.print("\n[bold]Panels consulted:[/bold]")
        for pid in result.seen_panel_ids:
            m = meta_by_id.get(pid)
            if not m:
                continue
            console.print(f"  [cyan]{m['book']}[/cyan] p.{m['page_index']:03d} ({pid})")
            pixels = _render_thumbnail(Path(m["image_path"]))
            if pixels is not None:
                console.print(pixels)
    console.print()


def run(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    with console.status("[bold green]Loading index and models..."):
        index = Index.load_all(index_dir)
        meta_by_id = {m["id"]: m for m in index.meta}
        embedder = ClipJapaneseEmbedder()
        searcher = Searcher(index, embedder)
    console.print(f"[bold green]Loaded[/bold green] {len(index)} panels from {index_dir}\n")

    agent: MrklIrcotAgent | None = None  # lazy: only hits the HF Inference API if the agent mode is used

    while True:
        try:
            answer = inquirer.prompt(
                [
                    inquirer.List(
                        "mode",
                        message="What do you want to do?",
                        choices=[MODE_SEARCH_TEXT, MODE_SEARCH_IMAGE, MODE_AGENT, MODE_QUIT],
                    )
                ]
            )
        except KeyboardInterrupt:
            break
        if answer is None or answer["mode"] == MODE_QUIT:
            break
        mode = answer["mode"]

        if mode == MODE_SEARCH_TEXT:
            q = inquirer.prompt([inquirer.Text("query", message="Query (English or Japanese)")])
            if q is None or not q["query"].strip():
                continue
            query = q["query"].strip()
            with console.status("[bold green]Searching..."):
                results = searcher.search_text(
                    query, top_k=args.top_k, w_img=args.w_img, w_text=args.w_text, translate=not args.no_translate
                )
                translations: list[str | None] = [None] * len(results)
                if not args.no_translate:
                    translations = searcher.translate_result_texts(results)
            if results and results[0].query_ja:
                console.print(f'[dim](translated to Japanese: "{results[0].query_ja}")[/dim]\n')
            _print_results(results, translations)

        elif mode == MODE_SEARCH_IMAGE:
            q = inquirer.prompt([inquirer.Text("path", message="Path to image file")])
            if q is None or not q["path"].strip():
                continue
            image_path = Path(q["path"].strip())
            if not image_path.exists():
                console.print(f"[red]no such file: {image_path}[/red]\n")
                continue
            with console.status("[bold green]Searching..."):
                results = searcher.search_image(image_path, top_k=args.top_k, w_img=args.w_img, w_text=args.w_text)
            _print_results(results, [None] * len(results))

        elif mode == MODE_AGENT:
            q = inquirer.prompt([inquirer.Text("question", message="Question for the agent")])
            if q is None or not q["question"].strip():
                continue
            if agent is None:
                agent = MrklIrcotAgent(
                    searcher, ChatLLM(model=args.model), max_steps=args.max_steps, top_k=args.agent_top_k
                )
            try:
                with console.status("[bold green]Running agent (calls the HF Inference API, can take a while)..."):
                    result = agent.answer(q["question"].strip())
            except Exception as e:
                console.print(f"[red]agent failed: {e}[/red]\n")
                continue
            _print_agent_result(result, meta_by_id)
