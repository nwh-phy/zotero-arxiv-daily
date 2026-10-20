from dataclasses import dataclass, field
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')


def _request_llm(openai_client: OpenAI, llm_params: dict, messages: list[dict]) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")
    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in generation_kwargs:
            generation_kwargs["max_output_tokens"] = max_tokens
        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion' or 'response'."
    )


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    published: Optional[str] = None
    journal_ref: Optional[str] = None
    recommendation_theme: Optional[str] = None
    related_papers: list[str] = field(default_factory=list)
    selection_score: Optional[float] = None
    evidence_excerpts: list[dict[str, str]] = field(default_factory=list)
    relevance_score: Optional[int] = None
    relevance_reason: Optional[str] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        instructions = llm_params.get('summary_instructions')
        if instructions:
            instructions = instructions.replace('[全文片段确认]', '[全文片段确认:E编号]')
        from .evidence import build_summary_prompt
        if not self.full_text and not self.abstract:
            raise ValueError("Neither full text nor abstract is provided")
        prompt = build_summary_prompt(self, lang)
        instructions = instructions or f"Summarize the core idea in one sentence in {lang}. Use only the supplied evidence."

        tldr = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": instructions,
                },
                {"role": "user", "content": prompt},
            ],
        )
        min_chars = llm_params.get('summary_min_chars')
        max_chars = llm_params.get('summary_max_chars')
        if min_chars is not None and max_chars is not None:
            issues = self.summary_validation_issues(tldr, llm_params)
            if issues:
                # One bounded repair using the original evidence, never truncation.
                allowed = '、'.join(f'[全文片段确认:{e["id"]}]' for e in self.evidence_excerpts)
                feedback = (
                    '上一版校验失败：' + '；'.join(issues) + '。\n'
                    f'请重写为总计{min_chars}～{max_chars}个汉字，目标180～200字，证据标签计入字数。'
                    '五项正文各约25～35个汉字，用问题、方法、结论、创新、参考价值及冒号标示。'
                    '每项均须有证据标记，或明确写尚无法确认；推测仅可用[推测]。'
                    '只依据原始论文文本，不得增加事实或机制。'
                    + ('允许的全文标记仅为：' + allowed + '。不得使用其他编号或不带编号的全文标记。'
                       if allowed else '未提供全文：删除全文片段确认，只能使用[摘要确认]。')
                )
                tldr = _request_llm(openai_client, llm_params, [
                    {'role': 'system', 'content': instructions},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': tldr or '（上一版未返回摘要文本）'},
                    {'role': 'user', 'content': feedback},
                ])
            issues = self.summary_validation_issues(tldr, llm_params)
            if issues:
                # Emit only deterministic constraint diagnostics, never model text.
                logger.warning('Summary constraints not met: {}', '；'.join(issues))
                raise ValueError('Summary still violates the requested format or length')
        if not tldr or not tldr.strip():
            raise ValueError('Empty summary from LLM')
        if not self.summary_is_acceptable(tldr, llm_params):
            raise ValueError('Summary cites unavailable evidence')
        return tldr
    
    def summary_is_acceptable(self, text: str, llm_params: dict) -> bool:
        return not self.summary_validation_issues(text, llm_params)

    def summary_validation_issues(self, text: str, llm_params: dict) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return ['摘要为空']
        issues = []
        ids = {excerpt['id'] for excerpt in self.evidence_excerpts}
        cited = re.findall(r'\[全文片段确认:(E\d+)\]', text)
        remainder = re.sub(r'\[全文片段确认:E\d+\]', '', text)
        if '全文片段确认' in remainder or any(ref not in ids for ref in cited):
            issues.append('全文标记格式或编号无效')
        minimum = llm_params.get('summary_min_chars')
        maximum = llm_params.get('summary_max_chars')
        if minimum is not None and maximum is not None:
            count = sum('\u4e00' <= c <= '\u9fff' for c in text)
            fields = list(re.finditer(r'(问题|方法|结论|创新|参考价值)\s*[:：]', text))
            if not minimum <= count <= maximum:
                issues.append(f'实际{count}个汉字，要求{minimum}～{maximum}个')
            if (len(fields) != 5
                    or {match.group(1) for match in fields} != {'问题', '方法', '结论', '创新', '参考价值'}):
                issues.append('五项字段缺失或重复')
            for i, match in enumerate(fields):
                end = fields[i+1].start() if i+1 < len(fields) else len(text)
                body = text[match.end():end]
                if not body.strip() or not (
                    '[摘要确认]' in body or re.search(r'\[全文片段确认:E\d+\]', body)
                    or '[推测]' in body or '尚无法确认' in body
                ):
                    issues.append(f'{match.group(1)}缺少证据标记或尚无法确认说明')
            if '[摘要确认]' not in text and not cited:
                issues.append('缺少可核对的事实证据标记')
        return issues

    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Summary request failed for {self.url} ({type(e).__name__})")
            if llm_params.get('strict_summary', False):
                raise RuntimeError("Chinese summary generation failed; email was not sent.") from None
            tldr = self.abstract or "Failed to generate TLDR. Neither full text nor abstract is provided"
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
    # Collections selected by include_path; None means every path is in scope.
    theme_paths: Optional[list[str]] = None
