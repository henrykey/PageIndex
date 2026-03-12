from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
from threading import RLock
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from .pageindex.page_index import page_index_main
from .pageindex.page_index_md import md_to_tree
from .pageindex.utils import ConfigLoader, clear_runtime_llm_config, set_runtime_llm_config

logger = logging.getLogger(__name__)

PAGEINDEX_VLM_CALL_TIMEOUT_CAP_SEC = int(os.getenv("PAGEINDEX_VLM_CALL_TIMEOUT_CAP_SEC", "1200"))
PAGEINDEX_VLM_MAX_RETRIES_CAP = int(os.getenv("PAGEINDEX_VLM_MAX_RETRIES_CAP", "5"))
PAGEINDEX_VLM_MAX_TOKENS_CAP = int(os.getenv("PAGEINDEX_VLM_MAX_TOKENS_CAP", "16384"))

mcp = FastMCP(name="pageindex", mask_error_details=True)

# In-memory index store:
# {tenant_id: {doc_key: [entry, ...]}}
# entry = {doc_id, version, page_no, title, snippet, text}
_INDEX_STORE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_INDEX_LOCK = RLock()


@mcp.tool("health_check")
async def health_check() -> str:
    with _INDEX_LOCK:
        tenant_count = len(_INDEX_STORE)
        doc_count = sum(len(v) for v in _INDEX_STORE.values())
    return json.dumps(
        {
            "ok": True,
            "service": "pageindex",
            "version": "0.2.0",
            "tenants": tenant_count,
            "indexed_docs": doc_count,
        },
        ensure_ascii=False,
    )


@mcp.tool("build_index_from_pdf")
async def build_index_from_pdf(
    pdf_path: Optional[str] = None,
    file_data: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    version: Optional[int] = None,
) -> str:
    try:
        res = await _build_index_from_pdf_internal(
            pdf_path=pdf_path,
            file_data=file_data,
            options=options,
            vlm_config=vlm_config,
            tenant_id=tenant_id,
            doc_id=doc_id,
            version=version,
        )
        return json.dumps(res, ensure_ascii=False)
    except Exception as exc:
        logger.exception("build_index_from_pdf failed")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@mcp.tool("build_index_from_markdown")
async def build_index_from_markdown(
    md_path: Optional[str] = None,
    md_content: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    version: Optional[int] = None,
) -> str:
    try:
        res = await _build_index_from_markdown_internal(
            md_path=md_path,
            md_content=md_content,
            options=options,
            vlm_config=vlm_config,
            tenant_id=tenant_id,
            doc_id=doc_id,
            version=version,
        )
        return json.dumps(res, ensure_ascii=False)
    except Exception as exc:
        logger.exception("build_index_from_markdown failed")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# Compatibility tool for Java default name: pageindex.build
@mcp.tool("build")
async def build(
    tenant_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    version: Optional[int] = None,
    pages: Optional[List[int]] = None,
    index_payload: Optional[Dict[str, Any]] = None,
    pdf_path: Optional[str] = None,
    file_data: Optional[str] = None,
    md_path: Optional[str] = None,
    md_content: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    vlm_config: Optional[Dict[str, Any]] = None,
) -> str:
    try:
        if index_payload:
            stored = _store_payload(
                index_payload=index_payload,
                tenant_id=tenant_id,
                doc_id=doc_id,
                version=version,
            )
            return json.dumps(
                {
                    "ok": True,
                    "mode": "payload",
                    "tenant_id": tenant_id or "default",
                    "doc_id": doc_id or "unknown",
                    "version": version,
                    "indexed_entries": stored,
                    "pages": pages or [],
                },
                ensure_ascii=False,
            )

        if md_content is not None or md_path:
            res = await _build_index_from_markdown_internal(
                md_path=md_path,
                md_content=md_content,
                options=options,
                vlm_config=vlm_config,
                tenant_id=tenant_id,
                doc_id=doc_id,
                version=version,
            )
            res["mode"] = "markdown"
            res["pages"] = pages or []
            return json.dumps(res, ensure_ascii=False)

        if file_data or pdf_path:
            res = await _build_index_from_pdf_internal(
                pdf_path=pdf_path,
                file_data=file_data,
                options=options,
                vlm_config=vlm_config,
                tenant_id=tenant_id,
                doc_id=doc_id,
                version=version,
            )
            res["mode"] = "pdf"
            res["pages"] = pages or []
            return json.dumps(res, ensure_ascii=False)

        # Keep compatibility with existing Java build call args
        # even when source content is not provided yet.
        return json.dumps(
            {
                "ok": True,
                "mode": "noop",
                "tenant_id": tenant_id or "default",
                "doc_id": doc_id or "unknown",
                "version": version,
                "pages": pages or [],
                "indexed_entries": 0,
                "warning": "No source payload provided (pdf/md/payload). Build accepted as no-op.",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("build failed")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


# Compatibility tool for Java default name: pageindex.search
@mcp.tool("search")
async def search(
    tenant_id: Optional[str] = None,
    query: str = "",
    top_k: int = 10,
    doc_id: Optional[str] = None,
    version: Optional[int] = None,
    index_payload: Optional[Dict[str, Any]] = None,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> str:
    try:
        tenant_key = str(tenant_id or "default")
        q = (query or "").strip()
        if not q:
            return json.dumps({"hits": [], "total": 0}, ensure_ascii=False)

        query_tokens = _tokenize(q)
        if not query_tokens:
            return json.dumps({"hits": [], "total": 0}, ensure_ascii=False)

        rows: List[Dict[str, Any]] = []
        payload_rows = entries
        if payload_rows is None and isinstance(index_payload, dict):
            payload_rows = index_payload.get("entries") or index_payload.get("hits") or index_payload.get("rows")

        if isinstance(payload_rows, list):
            for row in payload_rows:
                if not isinstance(row, dict):
                    continue
                row_doc_id = str(row.get("doc_id") or row.get("document_id") or doc_id or "unknown")
                row_version = int(row.get("version") or version or 0)
                if doc_id and row_doc_id != doc_id:
                    continue
                if version is not None and row_version != int(version):
                    continue
                rows.append(
                    {
                        "doc_id": row_doc_id,
                        "version": row_version,
                        "page_no": int(row.get("page_no") or row.get("page") or 0),
                        "title": str(row.get("title") or ""),
                        "snippet": str(row.get("snippet") or row.get("text") or row.get("content") or "")[:2000],
                        "text": str(row.get("text") or row.get("snippet") or row.get("content") or "")[:8000],
                    }
                )
        else:
            with _INDEX_LOCK:
                doc_map = _INDEX_STORE.get(tenant_key, {})
                for key, stored_entries in doc_map.items():
                    if doc_id and not key.startswith(f"{doc_id}#"):
                        continue
                    if version is not None and not key.endswith(f"#{version}"):
                        continue
                    rows.extend(stored_entries)

        scored: List[Dict[str, Any]] = []
        for row in rows:
            score = _score_row(query_tokens, row)
            if score <= 0:
                continue
            scored.append(
                {
                    "doc_id": row.get("doc_id", ""),
                    "page_no": int(row.get("page_no") or 0),
                    "title": str(row.get("title") or ""),
                    "snippet": str(row.get("snippet") or ""),
                    "score": round(float(score), 6),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        hits = scored[: max(1, int(top_k or 10))]
        return json.dumps(
            {
                "hits": hits,
                "total": len(scored),
                "tenant_id": tenant_key,
                "doc_id": doc_id,
                "version": version,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("search failed")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


async def _build_index_from_pdf_internal(
    pdf_path: Optional[str],
    file_data: Optional[str],
    options: Optional[Dict[str, Any]],
    vlm_config: Optional[Dict[str, Any]],
    tenant_id: Optional[str],
    doc_id: Optional[str],
    version: Optional[int],
) -> Dict[str, Any]:
    tmp_path: Optional[str] = None
    runtime_token = None

    try:
        source_path = pdf_path
        if file_data:
            raw = base64.b64decode(file_data)
            fd, tmp_path = tempfile.mkstemp(prefix="pageindex_", suffix=".pdf")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            source_path = tmp_path

        if not source_path or not os.path.exists(source_path):
            return {"error": "pdf_path/file_data is required and must exist"}

        user_opt = _normalize_options(options or {})
        opt = ConfigLoader().load(user_opt)

        eff_vlm = _normalize_vlm_config(vlm_config)
        if eff_vlm:
            if eff_vlm.get("model"):
                opt.model = str(eff_vlm.get("model"))
            runtime_token = set_runtime_llm_config(eff_vlm)

        result = page_index_main(source_path, opt)
        entries = _extract_entries(
            result,
            default_doc_id=doc_id or str(result.get("doc_name") or "unknown"),
            version=version,
        )
        stored = _store_entries(tenant_id=tenant_id, doc_id=doc_id, version=version, entries=entries)
        return {
            "result": result,
            "indexed_entries": stored,
            "tenant_id": tenant_id or "default",
            "doc_id": doc_id or str(result.get("doc_name") or "unknown"),
            "version": version,
            "vlm": _vlm_runtime_info(eff_vlm),
        }
    finally:
        if runtime_token is not None:
            clear_runtime_llm_config(runtime_token)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def _build_index_from_markdown_internal(
    md_path: Optional[str],
    md_content: Optional[str],
    options: Optional[Dict[str, Any]],
    vlm_config: Optional[Dict[str, Any]],
    tenant_id: Optional[str],
    doc_id: Optional[str],
    version: Optional[int],
) -> Dict[str, Any]:
    tmp_path: Optional[str] = None
    runtime_token = None

    try:
        source_path = md_path
        if md_content is not None:
            fd, tmp_path = tempfile.mkstemp(prefix="pageindex_", suffix=".md")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(md_content)
            source_path = tmp_path

        if not source_path or not os.path.exists(source_path):
            return {"error": "md_path/md_content is required and must exist"}

        user_opt = _normalize_options(options or {})
        eff_vlm = _normalize_vlm_config(vlm_config)

        model = str(eff_vlm.get("model") or user_opt.get("model") or "gpt-4o-2024-11-20") if eff_vlm else str(
            user_opt.get("model") or "gpt-4o-2024-11-20"
        )
        if eff_vlm:
            runtime_token = set_runtime_llm_config(eff_vlm)

        result = await md_to_tree(
            md_path=source_path,
            if_thinning=_to_bool(user_opt.get("if_thinning", False)),
            min_token_threshold=int(user_opt.get("thinning_threshold", 5000)),
            if_add_node_summary=_yes_no(user_opt.get("if_add_node_summary", "yes")),
            summary_token_threshold=int(user_opt.get("summary_token_threshold", 200)),
            model=model,
            if_add_doc_description=_yes_no(user_opt.get("if_add_doc_description", "no")),
            if_add_node_text=_yes_no(user_opt.get("if_add_node_text", "no")),
            if_add_node_id=_yes_no(user_opt.get("if_add_node_id", "yes")),
        )

        entries = _extract_entries(
            result,
            default_doc_id=doc_id or _doc_name_from_path(source_path),
            version=version,
        )
        stored = _store_entries(tenant_id=tenant_id, doc_id=doc_id, version=version, entries=entries)
        return {
            "result": result,
            "indexed_entries": stored,
            "tenant_id": tenant_id or "default",
            "doc_id": doc_id or _doc_name_from_path(source_path),
            "version": version,
            "vlm": _vlm_runtime_info(eff_vlm),
        }
    finally:
        if runtime_token is not None:
            clear_runtime_llm_config(runtime_token)
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _doc_name_from_path(path: Optional[str]) -> str:
    if not path:
        return "unknown"
    base = os.path.basename(path)
    if not base:
        return "unknown"
    return os.path.splitext(base)[0] or base


def _store_payload(index_payload: Dict[str, Any], tenant_id: Optional[str], doc_id: Optional[str], version: Optional[int]) -> int:
    entries: List[Dict[str, Any]] = []
    rows = index_payload.get("entries") or index_payload.get("hits") or index_payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            entries.append(
                {
                    "doc_id": str(row.get("doc_id") or row.get("document_id") or doc_id or "unknown"),
                    "version": version,
                    "page_no": int(row.get("page_no") or row.get("page") or 0),
                    "title": str(row.get("title") or ""),
                    "snippet": str(row.get("snippet") or row.get("text") or row.get("content") or ""),
                    "text": str(row.get("text") or row.get("snippet") or row.get("content") or ""),
                }
            )
    return _store_entries(tenant_id=tenant_id, doc_id=doc_id, version=version, entries=entries)


def _store_entries(tenant_id: Optional[str], doc_id: Optional[str], version: Optional[int], entries: List[Dict[str, Any]]) -> int:
    if not entries:
        return 0

    tenant_key = str(tenant_id or "default")
    real_doc_id = str(doc_id or entries[0].get("doc_id") or "unknown")
    ver = int(version) if version is not None else 0
    doc_key = f"{real_doc_id}#{ver}"

    normalized: List[Dict[str, Any]] = []
    for entry in entries:
        e = dict(entry)
        e["doc_id"] = str(e.get("doc_id") or real_doc_id)
        e["version"] = ver
        e["page_no"] = int(e.get("page_no") or 0)
        e["title"] = str(e.get("title") or "")
        e["snippet"] = str(e.get("snippet") or e.get("text") or "")[:2000]
        e["text"] = str(e.get("text") or e.get("snippet") or "")[:8000]
        normalized.append(e)

    with _INDEX_LOCK:
        _INDEX_STORE.setdefault(tenant_key, {})[doc_key] = normalized
    return len(normalized)


def _extract_entries(result: Dict[str, Any], default_doc_id: str, version: Optional[int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    structure = result.get("structure") if isinstance(result, dict) else result
    doc_id = str(default_doc_id)

    def walk(nodes: Any):
        if isinstance(nodes, list):
            for item in nodes:
                walk(item)
            return
        if not isinstance(nodes, dict):
            return

        start_idx = nodes.get("start_index")
        end_idx = nodes.get("end_index")
        line_num = nodes.get("line_num")
        page_no = _resolve_page_no(start_idx=start_idx, end_idx=end_idx, line_num=line_num)

        text = str(nodes.get("text") or nodes.get("summary") or nodes.get("prefix_summary") or "")
        title = str(nodes.get("title") or "")
        snippet = (title + "\n" + text).strip() if text else title

        if title or text:
            entries.append(
                {
                    "doc_id": doc_id,
                    "version": version,
                    "page_no": page_no,
                    "title": title,
                    "snippet": snippet[:2000],
                    "text": text[:8000],
                }
            )

        children = nodes.get("nodes")
        if children:
            walk(children)

    walk(structure)
    return entries


def _resolve_page_no(start_idx: Any, end_idx: Any, line_num: Any) -> int:
    try:
        if start_idx is not None:
            return max(0, int(start_idx))
    except Exception:
        pass
    try:
        if end_idx is not None:
            return max(0, int(end_idx))
    except Exception:
        pass
    try:
        if line_num is not None:
            # markdown heuristic: map line number to pseudo page bucket
            return max(1, int(line_num // 60) + 1)
    except Exception:
        pass
    return 0


def _tokenize(text: str) -> List[str]:
    s = (text or "").lower()
    word_tokens = re.findall(r"[a-z0-9_]+", s)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", s)
    return [t for t in (word_tokens + cjk_tokens) if t]


def _score_row(query_tokens: List[str], row: Dict[str, Any]) -> float:
    title = str(row.get("title") or "").lower()
    snippet = str(row.get("snippet") or "").lower()
    text = str(row.get("text") or row.get("snippet") or "").lower()
    combined = "\n".join([title, snippet, text])

    score = 0.0
    for t in query_tokens:
        tf_text = combined.count(t)
        tf_title = title.count(t)
        if tf_text == 0 and tf_title == 0:
            continue
        score += tf_text * 1.0 + tf_title * 2.5

    if query_tokens and all(t in combined for t in query_tokens):
        score += 1.5

    query_phrase = "".join(query_tokens)
    compact_text = re.sub(r"[^\w\u4e00-\u9fff]", "", combined)
    compact_query = re.sub(r"[^\w\u4e00-\u9fff]", "", query_phrase)
    if compact_query and compact_query in compact_text:
        score += 8.0

    raw_query = "".join(query_tokens)
    if raw_query and raw_query in combined:
        score += 4.0

    if "公式" in combined:
        score += 1.0
    if "公式" in raw_query and ("公式(" in combined or "$$" in combined or "[formulas]" in combined):
        score += 6.0
    if "最低" in raw_query and "最低试验压力" in combined:
        score += 4.0
    if "耐压试验" in raw_query and "耐压试验压力" in combined:
        score += 3.0

    # prefer rows with concrete page mapping
    if int(row.get("page_no") or 0) > 0:
        score += 0.2

    return score


def _normalize_options(options: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(options or {})

    # normalize booleans to PageIndex yes/no style flags
    for key in [
        "if_add_node_id",
        "if_add_node_summary",
        "if_add_doc_description",
        "if_add_node_text",
    ]:
        if key in out:
            out[key] = _yes_no(out.get(key))

    return out


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _yes_no(v: Any) -> str:
    return "yes" if _to_bool(v) or str(v).strip().lower() == "yes" else "no"


def _vlm_runtime_info(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    c = cfg or {}
    return {
        "provider": c.get("provider"),
        "model": c.get("model"),
        "base_url": c.get("base_url"),
        "max_tokens": c.get("max_tokens"),
    }


def _normalize_vlm_config(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cfg:
        return None

    c = dict(cfg)
    try:
        timeout_sec = int(c.get("timeout_sec", 60))
    except Exception:
        timeout_sec = 60
    try:
        max_retries = int(c.get("max_retries", 2))
    except Exception:
        max_retries = 2
    try:
        max_tokens = int(c.get("max_tokens", 4096))
    except Exception:
        max_tokens = 4096

    timeout_cap = int(PAGEINDEX_VLM_CALL_TIMEOUT_CAP_SEC)
    if timeout_cap > 0:
        c["timeout_sec"] = max(20, min(timeout_sec, timeout_cap))
    else:
        c["timeout_sec"] = max(20, timeout_sec)

    c["max_retries"] = max(0, min(max_retries, PAGEINDEX_VLM_MAX_RETRIES_CAP))
    c["max_tokens"] = max(512, min(max_tokens, PAGEINDEX_VLM_MAX_TOKENS_CAP))
    return c
