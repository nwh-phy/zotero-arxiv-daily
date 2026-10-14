import tarfile
import re
import glob
import math
import smtplib
import posixpath
from collections import Counter
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from loguru import logger
import datetime
from omegaconf import DictConfig
import pymupdf
import pymupdf.layout
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.layout.activate()

import pymupdf4llm  # noqa: E402

_TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bm25_pick(query: str, candidates: dict[str, str], k1: float = 1.5, b: float = 0.75) -> str:
    """Return the candidate key whose content best matches *query* by BM25."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return next(iter(candidates))

    doc_tokens = {name: _tokenize(content) for name, content in candidates.items()}
    N = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens.values()) / max(N, 1)

    df: Counter[str] = Counter()
    for tokens in doc_tokens.values():
        df.update(set(tokens))

    best_name, best_score = None, -1.0
    for name, tokens in doc_tokens.items():
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for q in query_tokens:
            n_q = df.get(q, 0)
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)
            f_q = tf.get(q, 0)
            score += idf * (f_q * (k1 + 1)) / (f_q + k1 * (1 - b + b * dl / max(avgdl, 1)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def extract_tex_code_from_tar(file_path:str, paper_id:str, paper_title:str | None = None) -> dict[str,str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None
 
    tex_files = [f.name for f in tar.getmembers() if f.isfile() and f.name.endswith('.tex')]
    if sum(tar.getmember(name).size for name in tex_files) > 32 * 1024 * 1024:
        tar.close()
        raise ValueError('TeX source exceeds the bounded extraction size')
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None
    
    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file) :
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl','')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")

    file_contents = {}
    doc_block_candidates: list[str] = []
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8',errors='ignore')
        content = re.sub(r'(?<!\\)%[^\n]*', '', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content) and not any(w in t for w in ['example', 'sample', 'template']):
            doc_block_candidates.append(t)
        file_contents[t] = content

    if main_tex is None:
        if len(doc_block_candidates) == 1:
            main_tex = doc_block_candidates[0]
            logger.debug(f"Choose {main_tex} as main tex file of {paper_id}")
        elif len(doc_block_candidates) > 1:
            if paper_title:
                main_tex = _bm25_pick(paper_title, {c: file_contents[c] for c in doc_block_candidates})
                logger.debug(f"Multiple document blocks found in {paper_id}; BM25 selected {main_tex} from {doc_block_candidates}")
            else:
                main_tex = doc_block_candidates[0]
                logger.debug(f"Multiple document blocks found in {paper_id}; no title provided, using first candidate {main_tex}")

    if main_tex is not None:
        normalized = {posixpath.normpath(name): content for name, content in file_contents.items()}
        root = posixpath.dirname(posixpath.normpath(main_tex))
        include_pattern = re.compile(r'\\(?:input|include)\s*(?:\{([^}]+)\}|([^\s%{}]+))')
        def expand(filename, stack=()):
            if filename in stack or len(stack) >= 16:
                raise ValueError('Cyclic or excessively nested TeX includes')
            text = normalized[filename]
            expanded_size = len(text)
            def replace(match):
                nonlocal expanded_size
                name = (match.group(1) or match.group(2)).strip()
                name = name if name.endswith('.tex') else name + '.tex'
                choices = [posixpath.normpath(posixpath.join(root, name)),
                           posixpath.normpath(posixpath.join(posixpath.dirname(filename), name)),
                           posixpath.normpath(name)]
                target = next((choice for choice in choices if choice in normalized), None)
                content = expand(target, (*stack, filename)) if target else '[TeX include unavailable]'
                expanded_size += len(content) - len(match.group(0))
                if expanded_size > 32 * 1024 * 1024:
                    raise ValueError('Expanded TeX source exceeds the bounded extraction size')
                return content
            return include_pattern.sub(replace, text)
        try:
            file_contents['all'] = expand(posixpath.normpath(main_tex))
        finally:
            tar.close()
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None
        
    tar.close()
    return file_contents

def extract_markdown_from_pdf(file_path:str) -> str:
    return pymupdf4llm.to_markdown(file_path,use_ocr=False,header=False,footer=False,ignore_code=True)

def glob_match(path:str, pattern:str) -> bool:
    re_pattern = glob.translate(pattern,recursive=True)
    return re.match(re_pattern, path) is not None

class EmailDeliveryError(OSError):
    """A sanitized SMTP failure with an explicit acceptance-uncertainty flag."""
    def __init__(self, phase, error_type, *, uncertain=False):
        super().__init__(f"SMTP {phase} failed ({error_type}); "
                         + ("acceptance is uncertain" if uncertain else "message was not accepted"))
        self.uncertain = uncertain


def send_email(config: DictConfig, html: str, subject: str | None = None, message_id: str | None = None):
    from email.mime.multipart import MIMEMultipart
    from email.utils import format_datetime, make_msgid
    from zoneinfo import ZoneInfo
    import ssl
    from .construct_email import html_to_text

    sender, receiver = config.email.sender, config.email.receiver
    def _format_addr(value):
        name, addr = parseaddr(value)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai'))
    msg = MIMEMultipart('alternative')
    msg['From'] = _format_addr(f'科研文献周报 <{sender}>')
    msg['To'] = _format_addr(receiver)
    msg['Subject'] = Header(subject or f'科研文献简报 {now:%Y/%m/%d}', 'utf-8').encode()
    msg['Date'] = format_datetime(now)
    msg['Message-ID'] = message_id or make_msgid(domain='zotero-arxiv-daily.local')
    msg.attach(MIMEText(html_to_text(html), 'plain', 'utf-8'))
    msg.attach(MIMEText(html, 'html', 'utf-8'))
    server = None
    phase = 'connection or TLS'
    try:
        port = int(config.email.smtp_port)
        if port == 465:
            server = smtplib.SMTP_SSL(config.email.smtp_server, port, timeout=30, context=ssl.create_default_context())
        else:
            try:
                server = smtplib.SMTP(config.email.smtp_server, port, timeout=30)
                server.starttls(context=ssl.create_default_context())
            except Exception:
                if server is not None:
                    try:
                        server.close()
                    except Exception:
                        pass
                server = smtplib.SMTP_SSL(config.email.smtp_server, port, timeout=30, context=ssl.create_default_context())
        phase = 'authentication'
        server.login(sender, config.email.sender_password)
        phase = 'submission'
        refused = server.sendmail(sender, [receiver], msg.as_string())
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)
    except Exception as exc:
        rejected = isinstance(exc, (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused,
                                    smtplib.SMTPDataError, smtplib.SMTPHeloError, smtplib.SMTPNotSupportedError))
        raise EmailDeliveryError(phase, type(exc).__name__, uncertain=phase == 'submission' and not rejected) from None
    finally:
        if server is not None:
            try:
                server.quit()
            except Exception:
                # A QUIT/disconnect failure cannot undo successful DATA acceptance.
                try:
                    server.close()
                except Exception:
                    pass
    return msg['Message-ID']
