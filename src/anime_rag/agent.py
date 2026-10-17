"""A MRKL-style agent (explicit Thought / Action / Observation / Final
Answer transcript, single retrieval tool) combined with IRCoT-style
retrieval: each search query is the agent's own latest reasoning step
rather than a fixed query, so multi-hop questions get resolved over several
retrieval rounds grounded in manga panels instead of one shot.

MRKL (Karpas et al. 2022) contributes the explicit tool-call scaffold and
stopping condition ("Final Answer:"). IRCoT (Trivedi et al. 2022)
contributes the retrieval strategy: instead of the agent picking an
arbitrary next query, the system prompt ties each Action Input to "what you
still need to find out" given the evidence gathered so far, i.e. the next
step of chain-of-thought reasoning becomes the next retrieval query.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from PIL import Image

from .llm import ChatLLM
from .search import SearchResult, Searcher

# Resize/recompress crops before sending them to the (vision-capable) LLM --
# full-size crops would bloat the request and, since every past Observation
# stays in the conversation history, compound across retrieval rounds.
IMAGE_MAX_DIM = 384
IMAGE_JPEG_QUALITY = 80

SYSTEM_PROMPT = """You are a research agent that answers questions about a manga story using ONLY a search tool over manga panels. You know nothing else about the story -- never use outside knowledge, and never invent characters, names, or events that are not in the search results.

Tool:
- search_manga[query]: searches manga panels by MEANING (semantic search, not keyword/boolean search) over both the dialogue text and the panel artwork. Give it a short natural-language phrase or question, in English. Do not use quotes, OR, AND, or multiple alternatives.

Each Observation gives you, for every matching panel, its dialogue text AND the actual panel image. Look at the image, not just the text -- translated dialogue can be imprecise or say nothing about what's visually happening (who's present, what they're doing or holding, the setting), so judge each panel by what you can actually see in it, not only its caption.

Respond with EXACTLY one of these two blocks, nothing else, no preamble:

Thought: <what you still need to find out, based on the question and the evidence gathered so far>
Action: search_manga
Action Input: <one short natural-language phrase, the next thing to look for>

or, once the evidence is enough to answer:

Thought: I now know the answer.
Final Answer: <answer in plain English, citing book title and page number for each claim>

If the evidence so far doesn't support a confident answer, say so in the Final Answer instead of guessing.
"""

FORCE_FINAL_MESSAGE = (
    "You have used all your searches. Answer now with a Final Answer based only on the evidence gathered so far."
)
REPROMPT_MESSAGE = "Reply using exactly the Thought/Action/Action Input format, or a Thought/Final Answer."

ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)")
FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


@dataclass
class AgentStep:
    reply: str
    query: str | None
    observation: str | None
    panels: list[tuple[SearchResult, str | None]] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    seen_panels: list[tuple[SearchResult, str | None]]


class MrklIrcotAgent:
    def __init__(
        self,
        searcher: Searcher,
        llm: ChatLLM | None = None,
        max_steps: int = 4,
        top_k: int = 3,
        ircot: bool = True,
    ):
        self.searcher = searcher
        self.llm = llm or ChatLLM()
        self.max_steps = max_steps
        self.top_k = top_k
        # ircot=False is an ablation, not a real mode to ship: every round
        # re-searches the literal original question instead of the agent's
        # own latest reasoning step, isolating whether IRCoT's query
        # reformulation is actually what multi-hop questions benefit from,
        # versus just "ask more than once" (see scripts/eval.py sweep).
        self.ircot = ircot

    @staticmethod
    def _encode_image(path: Path) -> str:
        img = Image.open(path).convert("RGB")
        img.thumbnail((IMAGE_MAX_DIM, IMAGE_MAX_DIM))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=IMAGE_JPEG_QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _observe(self, query: str) -> tuple[list[dict], str, list[tuple[SearchResult, str | None]]]:
        """Returns (multimodal content blocks for the LLM, plain-text
        summary for the CLI transcript, the ranked (SearchResult,
        translation) pairs -- same shape `search`/`ask` show, so the agent's
        transcript can carry the exact same detail: scores, characters,
        dialogue, page path)."""
        results = self.searcher.search_text(query, top_k=self.top_k)
        translations = self.searcher.translate_result_texts(results)

        if not results:
            return [{"type": "text", "text": "(no matching panels found)"}], "(no matching panels found)", []

        lines: list[str] = []
        panels: list[tuple[SearchResult, str | None]] = []
        content: list[dict] = [
            {"type": "text", "text": f"Observation for search_manga[{query!r}] -- {len(results)} panel(s):"}
        ]
        for idx, (r, en) in enumerate(zip(results, translations)):
            m = r.meta
            chars = ", ".join(m["characters"]) if m["characters"] else "unknown characters"
            dialogue = en or m["text"] or "(no dialogue)"
            label = f"{m['book']} p.{m['page_index']:03d} [{chars}]: {dialogue}"
            lines.append(f"- {label}")
            content.append({"type": "text", "text": f"Panel {idx + 1} -- {label}"})
            content.append({"type": "image_url", "image_url": {"url": self._encode_image(Path(m["image_path"]))}})
            panels.append((r, en))

        observation_text = "\n".join(lines)
        return content, observation_text, panels

    def answer(self, question: str) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        steps: list[AgentStep] = []
        seen_panels: list[tuple[SearchResult, str | None]] = []
        seen_ids: set[str] = set()

        for i in range(self.max_steps + 1):
            if i == self.max_steps:
                messages.append({"role": "user", "content": FORCE_FINAL_MESSAGE})

            reply = self.llm.chat(messages, max_tokens=250, stop=["Observation:"])
            messages.append({"role": "assistant", "content": reply})

            final_match = FINAL_ANSWER_RE.search(reply)
            if final_match:
                steps.append(AgentStep(reply=reply, query=None, observation=None))
                return AgentResult(answer=final_match.group(1).strip(), steps=steps, seen_panels=seen_panels)

            action_match = ACTION_INPUT_RE.search(reply)
            if not action_match:
                steps.append(AgentStep(reply=reply, query=None, observation=None))
                messages.append({"role": "user", "content": REPROMPT_MESSAGE})
                continue

            query = question if not self.ircot else action_match.group(1).strip().strip('"')
            content, observation_text, panels = self._observe(query)
            for r, t in panels:
                pid = r.meta["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    seen_panels.append((r, t))
            steps.append(AgentStep(reply=reply, query=query, observation=observation_text, panels=panels))
            messages.append({"role": "user", "content": content})

        return AgentResult(
            answer="(agent did not produce a final answer within the step budget)",
            steps=steps,
            seen_panels=seen_panels,
        )
