# analyzer/utils/vba_deobfuscator.py
from __future__ import annotations
import re
from base64 import b64decode
from typing import List, Iterable, Tuple

_CHR_CALL_ANY = re.compile(r"(?i)ChrW?\s*\(\s*([^)]+)\s*\)")
_CHR_CHAIN_ANY = re.compile(r'(?is)(?:ChrW?\s*\(\s*[^)]+\s*\)\s*(?:&|\+)\s*){1,}ChrW?\s*\(\s*[^)]+\s*\)')

_ARRAY_NUMS = re.compile(r"(?is)Array\(\s*((?:&H[0-9A-F]+|\d+)(?:\s*,\s*(?:&H[0-9A-F]+|\d+))*)\s*\)")
_XOR_NEAR = re.compile(r"(?i)\bXor\s+(\d{1,3})\b")

_MID_LIT = re.compile(r'(?is)Mid\(\s*"([^"]+)"\s*,\s*(\d+)\s*(?:,\s*(\d+))?\s*\)')
_LEFT_LIT = re.compile(r'(?is)Left\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)')
_RIGHT_LIT = re.compile(r'(?is)Right\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)')
_REPLACE_LIT = re.compile(r'(?is)Replace\(\s*"([^"]+)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\)')

_STR_LIT = re.compile(r'"([^"]+)"')

_B64 = re.compile(r'"([A-Za-z0-9+/=]{16,})"')

_JOIN_ARRAY = re.compile(r'(?is)Join\(\s*Array\(\s*((?:"[^"]*"(?:\s*,\s*"[^"]*")*))\s*\)\s*,\s*"([^"]*)"\s*\)')
_JOIN_SPLIT = re.compile(r'(?is)Join\(\s*Split\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*,\s*"([^"]*)"\s*\)')

_ASC_LIT = re.compile(r'(?i)AscW?\s*\(\s*"([^"])"\s*\)')

_STRREV_LIT = re.compile(r'(?is)StrReverse\(\s*"([^"]+)"\s*\)')

_FRAG_HTTP = re.compile(r'(?i)h\s*[\W_]*\s*t\s*[\W_]*\s*t\s*[\W_]*\s*p\s*[\W_]*\s*s?')
_FRAG_SCHEME_SEP = re.compile(r'(?i)\[\s*:\s*\]|\(\s*:\s*\)|\s*:\s*')
_FRAG_SLASH2 = re.compile(r'(?i)\/\s*\/')
_FRAG_DOT = re.compile(r'(?i)\[\s*\.\s*\]|\(\s*\.\s*\)|\{\s*\.\s*\}|\s*dot\s*|\s*\(\s*dot\s*\)|\s*\\x2e\s*')
_FRAG_WS = re.compile(r'\s+')

_RELAX_URL = re.compile(r'(?i)\b(?:https?|hxxps?)\b[^\s"\']{0,8}\/\/[^\s\'"<>]{3,}', re.M)
_RELAX_DOMAIN = re.compile(r'(?i)\b[a-z0-9][a-z0-9\-]{1,63}(?:\s*(?:\.|\[\s*\.\s*\]|\(\s*\.\s*\)|\s*dot\s*)\s*[a-z0-9\-]{1,63}){1,}\b')

def _to_int(tok: str) -> int:
    tok = tok.strip()
    if tok.lower().startswith("&h"):
        return int(tok[2:], 16)
    return int(tok, 10)

def _safe_chr(n: int) -> str:
    try:
        if 0 <= n <= 0x10FFFF:
            return chr(n)
    except Exception:
        pass
    return ""

def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in items or []:
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _pre_normalize_asc_literals(code: str) -> str:
    def repl(m: re.Match) -> str:
        ch = m.group(1)
        return str(ord(ch))
    return _ASC_LIT.sub(repl, code or "")

def _eval_int_expr(expr: str) -> int | None:
    if not expr:
        return None
    toks = re.findall(r'&H[0-9A-F]+|\d+|[+\-]|(?i)Xor', expr, flags=re.I)
    if not toks:
        return None
    try:
        val = _to_int(toks[0])
        i = 1
        while i < len(toks):
            op = toks[i].lower()
            rhs = _to_int(toks[i+1])
            if op == '+':
                val = (val + rhs) & 0xFFFFFFFF
            elif op == '-':
                val = (val - rhs) & 0xFFFFFFFF
            elif op == 'xor':
                val = (val ^ rhs) & 0xFFFFFFFF
            else:
                return None
            i += 2
        return int(val)
    except Exception:
        return None

def _normalize_ioc_variants(s: str) -> str:
    if not s:
        return s
    t = s

    # 공백·잡문자 섞인 스킴 → http/https
    t = _FRAG_HTTP.sub(lambda m: ('https' if 's' in m.group(0).lower()[-2:] else 'http'), t)

    # 콜론 우회 → ':'
    t = _FRAG_SCHEME_SEP.sub(':', t)

    # 슬래시 2개
    t = _FRAG_SLASH2.sub('//', t)

    # [.] (.) {dot} dot → .
    t = _FRAG_DOT.sub('.', t)

    # hxxp/hxxps → http/https
    t = re.sub(r'(?i)\bh\s*xx\s*p\b', 'http', t)
    t = re.sub(r'(?i)\bh\s*xx\s*ps\b', 'https', t)

    # http[:]// / https[:]//
    t = re.sub(r'(?i)https?\s*\[:\]\s*//', lambda m: m.group(0).lower().replace('[:]', ':'), t)

    # hxp/hxxtp 등 일부 변형
    t = re.sub(r'(?i)h\s*x{1,2}\s*tp', 'http', t)
    t = re.sub(r'(?i)h\s*x{1,2}\s*tps', 'https', t)

    # 과도한 공백 제거
    t = _FRAG_WS.sub('', t)

    # 'http:////' 같은 중복 슬래시 축소
    t = re.sub(r'(?i)https?:/{3,}', lambda m: m.group(0)[:8], t)  # 'http://' 또는 'https://'

    return t

def recover_from_chr_chain(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    t = _pre_normalize_asc_literals(text)

    for m in _CHR_CHAIN_ANY.finditer(t):
        seg = m.group(0)
        chars: List[str] = []
        for x in _CHR_CALL_ANY.finditer(seg):
            expr = x.group(1)
            val = _eval_int_expr(expr)
            if val is None:
                try:
                    val = _to_int(expr)
                except Exception:
                    val = None
            chars.append(_safe_chr((val or 0) & 0xFFFF))
        s = "".join(chars)
        if len(s) >= 4:
            out.append(_normalize_ioc_variants(s))
    return _dedup_keep_order(out)

def recover_from_array(text: str) -> List[str]:
    out: List[str] = []
    for m in _ARRAY_NUMS.finditer(text or ""):
        nums = [_to_int(t) for t in m.group(1).split(",")]
        wnd = (text[max(0, m.start()-200): m.end()+200] or "")
        kx = _XOR_NEAR.search(wnd)
        key = int(kx.group(1)) if kx else None
        if key is not None:
            nums = [(n ^ key) & 0xFF for n in nums]
        s = "".join(_safe_chr(n & 0xFF) for n in nums)
        if len(s) >= 4:
            out.append(_normalize_ioc_variants(s))
    return _dedup_keep_order(out)

def recover_from_base64(text: str) -> List[str]:
    out: List[str] = []
    for m in _B64.finditer(text or ""):
        b64 = m.group(1)
        if len(b64) < 16:
            continue
        try:
            data = b64decode(b64, validate=False)
            s = data.decode("utf-8", "ignore")
            if len(s) >= 4:
                out.append(_normalize_ioc_variants(s))
        except Exception:
            continue
    return _dedup_keep_order(out)

def recover_from_literal_mid_left_right_replace(text: str) -> List[str]:
    out: List[str] = []

    for m in _MID_LIT.finditer(text or ""):
        s, start_s, length_s = m.groups()
        start = max(1, int(start_s))
        length = int(length_s) if length_s else len(s) - (start - 1)
        out.append(_normalize_ioc_variants(s[start-1:start-1+max(0, length)]))

    for m in _LEFT_LIT.finditer(text or ""):
        s, n_s = m.groups()
        out.append(_normalize_ioc_variants(s[:max(0, int(n_s))]))

    for m in _RIGHT_LIT.finditer(text or ""):
        s, n_s = m.groups()
        out.append(_normalize_ioc_variants(s[-max(0, int(n_s)):]))

    for m in _REPLACE_LIT.finditer(text or ""):
        s, old, new = m.groups()
        out.append(_normalize_ioc_variants(s.replace(old, new)))

    return _dedup_keep_order(out)

def recover_from_string_concat(text: str) -> List[str]:
    out: List[str] = []
    pat = re.compile(r'(?is)(?:"[^"]+"\s*(?:&|\+)\s*){1,}"[^"]+"')
    for m in pat.finditer(text or ""):
        seg = m.group(0)
        lits = _STR_LIT.findall(seg)
        if lits:
            out.append(_normalize_ioc_variants("".join(lits)))
    return _dedup_keep_order(out)

def recover_from_join_array(text: str) -> List[str]:
    out: List[str] = []
    for m in _JOIN_ARRAY.finditer(text or ""):
        arr_raw, sep = m.groups()
        parts = [p.strip()[1:-1] for p in re.findall(r'"[^"]*"', arr_raw)]
        out.append(_normalize_ioc_variants(sep.join(parts)))
    return _dedup_keep_order(out)

def recover_from_join_split(text: str) -> List[str]:
    out: List[str] = []
    for m in _JOIN_SPLIT.finditer(text or ""):
        src, sep_in, sep_out = m.groups()
        try:
            parts = src.split(sep_in)
            out.append(_normalize_ioc_variants(sep_out.join(parts)))
        except Exception:
            continue
    return _dedup_keep_order(out)

def recover_from_strreverse_literals(text: str) -> List[str]:
    out: List[str] = []
    for m in _STRREV_LIT.finditer(text or ""):
        s = m.group(1)[::-1]
        if len(s) >= 4:
            out.append(_normalize_ioc_variants(s))
    return _dedup_keep_order(out)

def _strengthen_url_candidates(strings: Iterable[str], promote_domain_to_url: bool = True) -> Tuple[List[str], List[str]]:
    normed = [_normalize_ioc_variants(s) for s in (strings or []) if s]
    urls: List[str] = []
    for s in normed:
        urls += _RELAX_URL.findall(s)
    urls = _dedup_keep_order(urls)

    if promote_domain_to_url:
        doms = []
        for s in normed:
            doms += _RELAX_DOMAIN.findall(s)
        doms = _dedup_keep_order(doms)
        for d in doms:
            dd = _normalize_ioc_variants(d)
            if '.' in dd and not any(dd.startswith(p) for p in ('http://', 'https://')):
                urls.append('http://' + dd)
        urls = _dedup_keep_order(urls)

    return (normed, urls)

def deobfuscate_vba_strings(blobs: List[str]) -> List[str]:
    recovered: List[str] = []
    for t in (blobs or []):
        recovered += recover_from_chr_chain(t)
        recovered += recover_from_array(t)
        recovered += recover_from_base64(t)
        recovered += recover_from_literal_mid_left_right_replace(t)
        recovered += recover_from_string_concat(t)
        recovered += recover_from_join_array(t)
        recovered += recover_from_join_split(t)
        recovered += recover_from_strreverse_literals(t)

    normed, url_candidates = _strengthen_url_candidates(recovered, promote_domain_to_url=True)

    final = _dedup_keep_order(list(normed) + list(url_candidates))
    return final
