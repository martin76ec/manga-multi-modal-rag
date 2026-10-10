from __future__ import annotations

import argparse
from pathlib import Path

from . import source
from .agent import AgentResult, MrklIrcotAgent
from .annotations import parse_book
from .embedder import ClipJapaneseEmbedder
from .indexing import Index, build_index
from .llm import DEFAULT_MODEL, ChatLLM
from .panels import build_panels
from .search import SearchResult, Searcher

DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INDEX_DIR = Path("data/index")


def cmd_books(args: argparse.Namespace) -> None:
    mz = source.open_zip()
    books = source.list_books(mz)
    for book in books[: args.limit] if args.limit else books:
        print(book)
    print(f"\n{len(books)} books total")


def cmd_index(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    mz = None
    if args.all:
        mz = source.open_zip()
        books = source.list_books(mz)
        print(f"--all: indexing all {len(books)} books in the dataset")
    else:
        books = args.books

    to_fetch = [b for b in books if not source.book_is_extracted(b, data_dir)]
    if to_fetch:
        if mz is None:
            mz = source.open_zip()
        for book in to_fetch:
            print(f"extracting {book} from the dataset zip...")
            source.extract_book(mz, book, data_dir)

    embedder: ClipJapaneseEmbedder | None = None
    for i, book in enumerate(books):
        book_index_dir = index_dir / book
        if book_index_dir.exists() and not args.force:
            print(f"[{i + 1}/{len(books)}] {book}: already indexed, skipping (use --force to rebuild)")
            continue

        xml_path = data_dir / "annotations" / f"{book}.xml"
        book_obj = parse_book(xml_path)
        images_dir = data_dir / "images" / book
        crops_dir = data_dir / "crops" / book
        panels = build_panels(book_obj, images_dir, crops_dir)
        print(f"[{i + 1}/{len(books)}] {book}: {len(panels)} panels")

        if embedder is None:
            embedder = ClipJapaneseEmbedder()
        idx = build_index(panels, embedder, image_batch_size=args.batch_size, text_batch_size=args.batch_size * 2)
        idx.save(book_index_dir)
        print(f"[{i + 1}/{len(books)}] {book}: saved index to {book_index_dir}")


def cmd_crops(args: argparse.Namespace) -> None:
    """(Re)generate panel crop images without touching the index. Useful
    after pulling an embeddings-only index (see pull-index) onto a machine
    that has no data/raw/ yet -- no GPU/embedding work happens here."""
    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    books = args.books
    if not books:
        if not index_dir.exists():
            print(f"no books given and no index found under {index_dir}")
            return
        books = sorted(d.name for d in index_dir.iterdir() if d.is_dir())
        if not books:
            print(f"no books given and no book indices found under {index_dir}")
            return

    to_fetch = [b for b in books if not source.book_is_extracted(b, data_dir)]
    if to_fetch:
        mz = source.open_zip()
        for book in to_fetch:
            print(f"extracting {book} from the dataset zip...")
            source.extract_book(mz, book, data_dir)

    for book in books:
        xml_path = data_dir / "annotations" / f"{book}.xml"
        book_obj = parse_book(xml_path)
        images_dir = data_dir / "images" / book
        crops_dir = data_dir / "crops" / book
        panels = build_panels(book_obj, images_dir, crops_dir)
        print(f"{book}: {len(panels)} crops ready under {crops_dir}")


def cmd_push_index(args: argparse.Namespace) -> None:
    """Upload the built index (embeddings + metadata only -- never
    data/raw/crops, which lives in a separate directory tree) to a HF
    dataset repo, so it can be pulled onto another machine without
    recomputing embeddings there."""
    from huggingface_hub import HfApi

    index_dir = Path(args.index_dir)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        print(f"no index found under {index_dir}")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=str(index_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="update panel index (embeddings + metadata only, no images)",
    )
    visibility = "public" if args.public else "private"
    print(f"pushed {index_dir} ({visibility}) to https://huggingface.co/datasets/{args.repo_id}")


def cmd_pull_index(args: argparse.Namespace) -> None:
    from huggingface_hub import snapshot_download

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(repo_id=args.repo_id, repo_type="dataset", local_dir=str(index_dir))
    books = sorted(d.name for d in Path(local_path).iterdir() if d.is_dir())
    print(f"pulled {len(books)} book indices from https://huggingface.co/datasets/{args.repo_id} into {local_path}")
    print("note: this is embeddings + metadata only -- run `anime-rag crops` to regenerate panel images locally")


def _print_results(results: list[SearchResult], translations: list[str | None]) -> None:
    for rank, r in enumerate(results):
        m = r.meta
        chars = ", ".join(m["characters"]) if m["characters"] else "-"
        text = m["text"] or "-"
        text_score = f"{r.text_score:.3f}" if r.text_score is not None else "-"
        print(f"#{rank} [{r.score:.3f}] {m['book']} p.{m['page_index']:03d} ({m['id']})")
        print(f"    img={r.image_score:.3f} text={text_score}  chars: {chars}")
        print(f"    text (ja): {text}")
        if translations[rank]:
            print(f"    text (en): {translations[rank]}")
        print(f"    crop:  {m['image_path']}")
        print()


def _save_crops(results: list[SearchResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(results):
        src = Path(r.meta["image_path"])
        dst = out_dir / f"{rank:02d}_{src.name}"
        dst.write_bytes(src.read_bytes())
    print(f"saved {len(results)} crops to {out_dir}")


def cmd_search(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    index = Index.load_all(index_dir)
    embedder = ClipJapaneseEmbedder()
    searcher = Searcher(index, embedder)

    if args.image:
        results = searcher.search_image(Path(args.image), top_k=args.top_k, w_img=args.w_img, w_text=args.w_text)
    else:
        results = searcher.search_text(
            args.query, top_k=args.top_k, w_img=args.w_img, w_text=args.w_text, translate=not args.no_translate
        )
        if results and results[0].query_ja:
            print(f'(translated query to Japanese: "{results[0].query_ja}")\n')

    translations: list[str | None] = [None] * len(results)
    if not args.no_translate:
        translations = searcher.translate_result_texts(results)

    _print_results(results, translations)

    if args.save_crops:
        _save_crops(results, Path(args.save_crops))


def cmd_ask(args: argparse.Namespace) -> None:
    """Interactive REPL: load the index and models once, then let you type
    one question after another instead of re-paying model load time per
    query like a one-shot `anime-rag search` call would."""
    index_dir = Path(args.index_dir)
    index = Index.load_all(index_dir)
    meta_by_id = {m["id"]: m for m in index.meta}
    print(f"Loaded {len(index)} panels from {index_dir}.")
    embedder = ClipJapaneseEmbedder()
    searcher = Searcher(index, embedder)
    agent: MrklIrcotAgent | None = None  # lazy: only hits the HF Inference API if 'agent:' is used

    print("Type a question (English or Japanese) and press enter -- plain retrieval by default.")
    print("Prefix with 'image:' and a path to search by image, e.g. image:data/raw/crops/ARMS/ARMS_001_xxx.jpg")
    print("Prefix with 'agent:' to route through the MRKL+IRCoT agent (uses the HF Inference API) instead.")
    print("Empty line or 'quit' to exit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        if query.lower().startswith("agent:"):
            question = query[len("agent:") :].strip()
            if not question:
                print("usage: agent: <question>\n")
                continue
            if agent is None:
                agent = MrklIrcotAgent(
                    searcher, ChatLLM(model=args.model), max_steps=args.max_steps, top_k=args.agent_top_k
                )
            print("(running agent -- this calls the HF Inference API and can take a while)", flush=True)
            try:
                result = agent.answer(question)
            except Exception as e:
                print(f"agent failed: {e}\n")
                continue
            _print_agent_result(result, meta_by_id)
            print()
            continue

        if query.lower().startswith("image:"):
            image_path = Path(query[len("image:") :].strip())
            if not image_path.exists():
                print(f"no such file: {image_path}\n")
                continue
            results = searcher.search_image(image_path, top_k=args.top_k, w_img=args.w_img, w_text=args.w_text)
        else:
            results = searcher.search_text(
                query, top_k=args.top_k, w_img=args.w_img, w_text=args.w_text, translate=not args.no_translate
            )
            if results and results[0].query_ja:
                print(f'(translated to Japanese: "{results[0].query_ja}")\n')

        translations: list[str | None] = [None] * len(results)
        if not args.no_translate:
            translations = searcher.translate_result_texts(results)

        _print_results(results, translations)


def _print_agent_result(result: AgentResult, meta_by_id: dict) -> None:
    for i, step in enumerate(result.steps):
        print(f"--- step {i} ---")
        print(step.reply.strip())
        if step.query:
            print(f"\nObservation (search: {step.query!r}):")
            print(step.observation)
        print()

    print("=" * 40)
    print("Final Answer:", result.answer)

    if result.seen_panel_ids:
        print("\nPanels consulted:")
        for pid in result.seen_panel_ids:
            m = meta_by_id.get(pid)
            if m:
                print(f"  - {m['book']} p.{m['page_index']:03d} ({pid}): {m['image_path']}")


def cmd_agent(args: argparse.Namespace) -> None:
    index = Index.load_all(Path(args.index_dir))
    meta_by_id = {m["id"]: m for m in index.meta}
    embedder = ClipJapaneseEmbedder()
    searcher = Searcher(index, embedder)
    llm = ChatLLM(model=args.model)
    agent = MrklIrcotAgent(searcher, llm, max_steps=args.max_steps, top_k=args.top_k)

    result = agent.answer(args.question)
    _print_agent_result(result, meta_by_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anime-rag", description="Multimodal RAG search over Manga109-s")
    sub = parser.add_subparsers(dest="command", required=True)

    p_books = sub.add_parser("books", help="list books available in the dataset")
    p_books.add_argument("--limit", type=int, default=0)
    p_books.set_defaults(func=cmd_books)

    p_index = sub.add_parser("index", help="extract and embed books into the panel index")
    p_index.add_argument("books", nargs="*", help="book titles, as shown by `anime-rag books` (omit with --all)")
    p_index.add_argument("--all", action="store_true", help="index every book in the dataset")
    p_index.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_index.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_index.add_argument("--force", action="store_true", help="re-index even if already present")
    p_index.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="image embedding batch size (text uses 2x this). Raise this a lot on a big GPU, e.g. 256+ on an H200",
    )
    p_index.set_defaults(func=cmd_index)

    p_crops = sub.add_parser(
        "crops", help="(re)generate panel crop images for books without touching the index (no GPU needed)"
    )
    p_crops.add_argument(
        "books", nargs="*", help="book titles; omit to regenerate crops for every book under --index-dir"
    )
    p_crops.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p_crops.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_crops.set_defaults(func=cmd_crops)

    p_push = sub.add_parser(
        "push-index", help="upload the built index (embeddings + metadata, no images) to a HF dataset repo"
    )
    p_push.add_argument("repo_id", help="e.g. yourname/manga109-rag-index")
    p_push.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_push.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    p_push.set_defaults(func=cmd_push_index)

    p_pull = sub.add_parser("pull-index", help="download a previously pushed index from a HF dataset repo")
    p_pull.add_argument("repo_id", help="e.g. yourname/manga109-rag-index")
    p_pull.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_pull.set_defaults(func=cmd_pull_index)

    p_search = sub.add_parser("search", help="search the panel index with text or an image")
    p_search.add_argument("query", nargs="?", default=None, help="text query")
    p_search.add_argument("--image", default=None, help="path to an image to use as the query instead of text")
    p_search.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--w-img", type=float, default=0.6, help="weight for the image-embedding score")
    p_search.add_argument("--w-text", type=float, default=0.4, help="weight for the text-embedding score")
    p_search.add_argument("--save-crops", default=None, help="directory to copy result crops into")
    p_search.add_argument(
        "--no-translate",
        action="store_true",
        help="don't auto-translate an English query to Japanese or results to English",
    )
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="interactively ask questions against the panel index")
    p_ask.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--w-img", type=float, default=0.6, help="weight for the image-embedding score")
    p_ask.add_argument("--w-text", type=float, default=0.4, help="weight for the text-embedding score")
    p_ask.add_argument(
        "--no-translate",
        action="store_true",
        help="don't auto-translate an English query to Japanese or results to English",
    )
    p_ask.add_argument("--model", default=DEFAULT_MODEL, help="HF Inference chat model id, for 'agent:' queries")
    p_ask.add_argument("--max-steps", type=int, default=4, help="max retrieval rounds for 'agent:' queries")
    p_ask.add_argument("--agent-top-k", type=int, default=3, help="panels retrieved per round for 'agent:' queries")
    p_ask.set_defaults(func=cmd_ask)

    p_agent = sub.add_parser(
        "agent", help="answer a question via a MRKL+IRCoT retrieval agent (uses the HF Inference API)"
    )
    p_agent.add_argument("question", help="question to answer")
    p_agent.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    p_agent.add_argument("--model", default=DEFAULT_MODEL, help="HF Inference chat model id")
    p_agent.add_argument("--max-steps", type=int, default=4, help="max retrieval rounds before forcing an answer")
    p_agent.add_argument("--top-k", type=int, default=3, help="panels retrieved per search round")
    p_agent.set_defaults(func=cmd_agent)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "search" and not args.query and not args.image:
        parser.error("search requires either a text query or --image")
    if args.command == "index" and not args.all and not args.books:
        parser.error("index requires book titles, or --all")
    args.func(args)
