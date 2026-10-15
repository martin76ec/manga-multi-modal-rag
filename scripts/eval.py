"""Lightweight, RAGAS-style evaluation against the already-built index.

No `ragas` package: it hard-imports langchain_community internals at
import time (confirmed broken against current langchain_community, and
pulls in ~92 packages including the full LangChain stack this project
deliberately avoids for the agent -- see README). Hit-rate/MRR cover the
"RAGAS context precision/recall" idea for retrieval with zero LLM calls;
the small LLM-judge scorer below covers faithfulness/relevancy the same
way RAGAS would, using our own ChatLLM instead.

Nothing here touches data/index/ -- hyperparameters are all applied at
query time (w_img/w_text in search.py, max_steps/top_k in agent.py), so
this never re-embeds or re-indexes anything. Requires AisazuNihaIrarenai,
DollGun, and HanzaiKousyouninMinegishiEitarou specifically indexed (the
books the eval cases below reference), not just any subset.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anime_rag.agent import MrklIrcotAgent
from anime_rag.embedder import ClipJapaneseEmbedder
from anime_rag.indexing import Index
from anime_rag.llm import ChatLLM
from anime_rag.search import Searcher

INDEX_DIR = Path("data/index")


@dataclass
class EvalCase:
    query: str
    expected_ids: list[str]
    verified_by: str


# Ground truth reused from queries already manually verified during this
# project's development (by viewing the actual panel image, or by the
# agent correctly describing a visual detail the dialogue didn't mention),
# not a freshly built benchmark.
#
# "police officer"'s original ground truth (AisazuNihaIrarenai_057, pixel-
# verified) was established with only 2 of 87 books indexed. Running this
# eval for real against the full index showed it's no longer the top
# match: there are now multiple panels with literal "警察官"/"警察だ"
# dialogue that outrank it. Those are genuinely correct too (confirmed by
# their transcribed text, not re-verified pixel by pixel), so they're
# added as additional valid answers rather than treating the corpus growth
# as a regression.
RETRIEVAL_CASES = [
    EvalCase(
        query="police officer",
        expected_ids=[
            "AisazuNihaIrarenai_057_000015bd",  # pixel-verified: panel reads "ちょっと警察の方!!"
            "HanzaiKousyouninMinegishiEitarou_054_0001f1a6",  # text-verified: literally "...警察官..."
            "HanzaiKousyouninMinegishiEitarou_094_0001fa0d",  # text-verified: literally "警察だ！"
        ],
        verified_by="mixed: one pixel-verified, two text-verified (see comment above)",
    ),
    EvalCase(
        query="someone holding a gun",
        expected_ids=["DollGun_079_00011f25"],
        verified_by="visual: panel shows a smoking handgun",
    ),
]

AGENT_QUESTION = "What happens when a police officer shows up at someone's house?"
AGENT_EXPECTED_IDS = [
    "AisazuNihaIrarenai_057_000015bd",
    "AisazuNihaIrarenai_057_000015e7",
    "AisazuNihaIrarenai_057_000015d3",
]

JUDGE_PROMPT = """You are grading a RAG system's answer. Given the question, the cited evidence, and the answer, rate two things from 1 (worst) to 5 (best):

faithfulness: is every claim in the answer actually supported by the evidence, with nothing invented or contradicted?
relevancy: does the answer actually address the question asked?

Respond with EXACTLY two lines, nothing else:
faithfulness: <1-5>
relevancy: <1-5>
"""


def hit_rate_and_mrr(searcher: Searcher, cases: list[EvalCase], top_k: int, w_img: float, w_text: float) -> tuple[float, float]:
    hits = 0
    reciprocal_ranks = []
    for case in cases:
        results = searcher.search_text(case.query, top_k=top_k, w_img=w_img, w_text=w_text)
        ids = [r.meta["id"] for r in results]
        rank = next((i + 1 for i, id_ in enumerate(ids) if id_ in case.expected_ids), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        hits += 1 if rank else 0
    n = len(cases)
    return hits / n, sum(reciprocal_ranks) / n


def sweep_retrieval(searcher: Searcher) -> None:
    print(f"Retrieval sweep over {len(RETRIEVAL_CASES)} verified cases (top_k=5):")
    for case in RETRIEVAL_CASES:
        print(f"  - {case.query!r} -> {case.expected_ids[0]} ({case.verified_by})")
    print(f"\n{'w_img':>6} {'w_text':>7} {'hit-rate':>9} {'MRR':>6}")
    for w_img in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        w_text = round(1 - w_img, 1)
        hit_rate, mrr = hit_rate_and_mrr(searcher, RETRIEVAL_CASES, top_k=5, w_img=w_img, w_text=w_text)
        print(f"{w_img:>6.1f} {w_text:>7.1f} {hit_rate:>9.2f} {mrr:>6.2f}")


def judge_answer(llm: ChatLLM, question: str, evidence: str, answer: str) -> tuple[str | None, str | None]:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer: {answer}"},
    ]
    reply = llm.chat(messages, max_tokens=30)
    faith, rel = None, None
    for line in reply.splitlines():
        low = line.lower()
        if "faithfulness" in low:
            faith = line.split(":")[-1].strip()
        if "relevancy" in low:
            rel = line.split(":")[-1].strip()
    return faith, rel


def sweep_agent(searcher: Searcher) -> None:
    print(f"\nAgent sweep, question: {AGENT_QUESTION!r}")
    judge_llm = ChatLLM()
    for top_k, max_steps in [(1, 1), (3, 4)]:
        agent = MrklIrcotAgent(searcher, ChatLLM(), max_steps=max_steps, top_k=top_k)
        result = agent.answer(AGENT_QUESTION)
        cited_ids = {r.meta["id"] for r, _ in result.seen_panels}
        hit = bool(cited_ids & set(AGENT_EXPECTED_IDS))
        evidence = "\n".join(
            f"- {r.meta['book']} p.{r.meta['page_index']}: {t or r.meta['text']}" for r, t in result.seen_panels
        )
        faith, rel = judge_answer(judge_llm, AGENT_QUESTION, evidence or "(none)", result.answer)
        print(f"\ntop_k={top_k} max_steps={max_steps}: hit={hit} rounds_used={len(result.steps)}")
        print(f"  faithfulness={faith} relevancy={rel}")
        print(f"  answer: {result.answer[:160]}")


def main() -> None:
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        print(f"no index found under {INDEX_DIR}; run `make index-all` or `anime-rag index` first")
        return

    index = Index.load_all(INDEX_DIR)
    embedder = ClipJapaneseEmbedder()
    searcher = Searcher(index, embedder)

    sweep_retrieval(searcher)
    sweep_agent(searcher)


if __name__ == "__main__":
    main()
