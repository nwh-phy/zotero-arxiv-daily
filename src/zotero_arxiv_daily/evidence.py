"""Select bounded, labelled excerpts without treating them as a full-paper review."""
import re

import tiktoken

EVIDENCE_VERSION = 1

_TEX_HEADING = re.compile(r"\\(?:sub)*section\*?(?:\[[^\]]*\])?\{([^{}]+)\}")
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PLAIN_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
    r"(abstract|introduction|methods?|materials and methods|experimental(?: methods| setup)?|"
    r"results?(?: and discussion)?|discussion|conclusions?|summary|references|acknowledg(?:e)?ments?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _priority(heading):
    heading = heading.lower()
    if any(word in heading for word in ("reference", "bibliograph", "acknowledg")):
        return None
    if any(word in heading for word in ("conclusion", "summary")):
        return 0
    if any(word in heading for word in ("result", "discussion")):
        return 1
    if any(word in heading for word in ("method", "experiment", "material", "model", "theor")):
        return 2
    if "abstract" in heading:
        return 3
    return 4


def select_excerpts(text: str, token_budget: int = 2600) -> list[dict[str, str]]:
    if not text or token_budget <= 0:
        return []
    # Drop TeX preamble and comments before finding section boundaries.
    if r"\begin{document}" in text or _TEX_HEADING.search(text):
        text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1]
    text = text.split(r"\end{document}", 1)[0]
    matches = sorted(
        [*_TEX_HEADING.finditer(text), *_MARKDOWN_HEADING.finditer(text), *_PLAIN_HEADING.finditer(text)],
        key=lambda match: match.start(),
    )
    sections = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        priority = _priority(heading)
        if priority is not None and body:
            sections.append((priority, match.start(), heading, body))
    if not matches:
        # Without trustworthy section boundaries, provide both ends of the body.
        midpoint = len(text) // 2
        sections = [(4, 0, "未识别章节：正文开头", text[:midpoint]),
                    (4, midpoint, "未识别章节：正文末尾", text[midpoint:])]
    elif not sections:
        return []
    sections.sort(key=lambda section: (section[0], section[1]))
    # Several results subsections must not displace the methods section entirely.
    chosen, priorities = [], set()
    for section in sections:
        if section[0] not in priorities:
            chosen.append(section)
            priorities.add(section[0])
    chosen = chosen[:4]
    for section in sections:
        if len(chosen) >= 4:
            break
        if section not in chosen:
            chosen.append(section)
    sections = sorted(chosen, key=lambda section: (section[0], section[1]))
    encoding = tiktoken.get_encoding("o200k_base")
    budget = max(1, token_budget // len(sections))
    excerpts = []
    for _, offset, heading, body in sections:
        tokens = encoding.encode(body)
        # Retain the end as well as the start when a section is long.
        if len(tokens) > budget:
            start_budget = budget * 2 // 3
            body = encoding.decode(tokens[:start_budget]) + "\n[中间内容省略]\n" + encoding.decode(tokens[-(budget-start_budget):])
        if body.strip():
            excerpts.append({"id": f"E{len(excerpts) + 1}", "section": heading,
                             "offset": str(offset), "text": body})
    return excerpts


def build_summary_prompt(paper, language: str) -> str:
    encoding = tiktoken.get_encoding("o200k_base")
    def clip(text, count):
        return encoding.decode(encoding.encode(text or '')[:count])
    title = clip(paper.title, 200)
    abstract = clip(paper.abstract, 700)
    paper.evidence_excerpts = select_excerpts(paper.full_text or "")
    prompt = (f"Generate a summary in {language} using only the supplied evidence.\n"
              f"Title: {title}\n\nAbstract [A]: {abstract}\n\n")
    if paper.evidence_excerpts:
        prompt += ("证据范围：摘要及以下正文片段，未提供完整论文。"
                   "全文事实必须使用[全文片段确认:E编号]指明来源，例如[全文片段确认:E1]。\n")
        supplied = []
        for excerpt in paper.evidence_excerpts:
            heading = clip(excerpt['section'], 50)
            header = f"\n[{excerpt['id']}] Section: {heading}\n"
            remaining = 4000 - len(encoding.encode(prompt + header)) - 2
            if remaining <= 0:
                break
            excerpt = {**excerpt, 'text': clip(excerpt['text'], remaining)}
            prompt += header + excerpt['text'] + '\n'
            supplied.append(excerpt)
        paper.evidence_excerpts = supplied
    else:
        prompt += "证据范围：仅提供摘要，未提供可用正文片段。只能使用[摘要确认]，不得声称全文片段确认。\n"
    return prompt
