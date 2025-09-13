from __future__ import annotations
import asyncio, time, re, urllib.parse
from typing import List, Dict, Any
import httpx, feedparser
from fastapi import APIRouter, Query

router = APIRouter(prefix="/news", tags=["news"])

# Primary publisher RSS feeds (may be picky in containers)
PRIMARY_FEEDS: Dict[str, str] = {
    "WHO-News":   "https://www.who.int/feeds/entity/mediacentre/news/en/rss.xml",
    "WHO-DON":    "https://www.who.int/feeds/entity/csr/don/en/rss.xml",
    "NIH":        "https://www.nih.gov/news-events/news-releases/feed",
    "MedlinePlus":"https://medlineplus.gov/feeds/news_en.xml",
    "CDC":        "https://tools.cdc.gov/api/v2/resources/media/404952.rss",
    "NatureNews": "https://www.nature.com/nature/articles?type=news&format=rss",
}

# Domain-scoped Google News RSS (reliable fallback)
def google_news_rss_for(domain: str, extra: str = "") -> str:
    # Example: site:who.int health
    q = f"site:{domain}"
    if extra:
        q += f" {extra}"
    params = {
        "q": q,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)

FALLBACK_FEEDS: Dict[str, str] = {
    "WHO-News":    google_news_rss_for("who.int"),
    "WHO-DON":     google_news_rss_for("who.int", "outbreak"),
    "NIH":         google_news_rss_for("nih.gov"),
    "MedlinePlus": google_news_rss_for("medlineplus.gov"),
    "NatureNews":  google_news_rss_for("nature.com", "news"),
    # Keep CDC primary (it already works); add fallback anyway:
    "CDC":         google_news_rss_for("cdc.gov"),
}

CACHE_TTL_SEC = 600
_cache: Dict[str, Any] = {"t": 0.0, "items": [], "counts": {}}

UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 ISHI/1.0",
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
}

def _extract_image(entry: Dict[str, Any]) -> str | None:
    media = entry.get("media_content") or entry.get("media_thumbnail") or []
    if isinstance(media, list) and media:
        url = media[0].get("url")
        if url: return url
    encl = entry.get("enclosures") or []
    if isinstance(encl, list) and encl:
        url = encl[0].get("href") or encl[0].get("url")
        if url: return url
    summary = entry.get("summary") or entry.get("description") or ""
    m = re.search(r'<img[^>]+src="([^"]+)"', summary, flags=re.I)
    return m.group(1) if m else None

def _normalize(source: str, parsed: feedparser.FeedParserDict, cap: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in parsed.entries[:cap]:
        out.append({
            "source": source,
            "title": (e.get("title") or "").strip(),
            "url":   (e.get("link")  or "").strip(),
            "published": e.get("published") or e.get("updated") or "",
            "summary":   (e.get("summary") or e.get("description") or "")[:600],
            "image": _extract_image(e),
            "published_parsed": e.get("published_parsed") or e.get("updated_parsed"),
            "score": 0.0,
        })
    return out

def _age_days(item: Dict[str, Any]) -> float:
    pp = item.get("published_parsed")
    if pp:
        try:
            return max(0.0, (time.time() - time.mktime(pp)) / 86400.0)
        except Exception:
            pass
    return 30.0

def _rank(items: List[Dict[str, Any]]) -> None:
    # Source weights (adjust to taste)
    w = {
        "NatureNews": 1.2,
        "WHO-News":   1.1,
        "WHO-DON":    1.05,
        "NIH":        1.0,
        "CDC":        0.95,
        "MedlinePlus":0.9,
    }
    for it in items:
        age = _age_days(it)
        recency = max(0.0, 30.0 - age) / 30.0
        has_img = 1.0 if it.get("image") else 0.0
        weight = w.get(it["source"], 1.0)
        it["score"] = recency * 0.6 + has_img * 0.2 + weight * 0.2

def _dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        k = it["url"] or it["title"]
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out

async def _fetch_rss(client: httpx.AsyncClient, name: str, url: str) -> List[Dict[str, Any]]:
    try:
        r = await client.get(url, headers=UA_HEADERS, timeout=25.0, follow_redirects=True)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
        return _normalize(name, parsed)
    except Exception as e:
        print(f"[news] {name} primary failed: {e}")
        return []

async def _fetch_with_fallback(client: httpx.AsyncClient, name: str) -> List[Dict[str, Any]]:
    primary = await _fetch_rss(client, name, PRIMARY_FEEDS[name])
    if primary:
        return primary
    # fallback via Google News RSS
    fb_url = FALLBACK_FEEDS.get(name)
    if not fb_url:
        return []
    try:
        r = await client.get(fb_url, headers=UA_HEADERS, timeout=25.0, follow_redirects=True)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
        items = _normalize(name, parsed)
        if not items:
            print(f"[news] {name} fallback yielded 0")
        return items
    except Exception as e:
        print(f"[news] {name} fallback failed: {e}")
        return []

async def _load_all() -> Dict[str, Any]:
    transport = httpx.AsyncHTTPTransport(retries=2)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        tasks = [_fetch_with_fallback(client, n) for n in PRIMARY_FEEDS.keys()]
        res = await asyncio.gather(*tasks, return_exceptions=False)

    items = [x for sub in res for x in sub]
    counts: Dict[str, int] = {}
    for name, sub in zip(PRIMARY_FEEDS.keys(), res):
        counts[name] = len(sub)

    items = _dedup(items)
    _rank(items)
    items.sort(key=lambda x: x["score"], reverse=True)
    for it in items:
        it.pop("published_parsed", None)
    return {"items": items, "counts": counts}

@router.get("")
async def get_news(
    limit: int = Query(10, ge=1, le=50),
    source: str | None = Query(None),
    q: str | None = Query(None),
    debug: int = Query(0, ge=0, le=1)
):
    now = time.time()
    global _cache
    if now - _cache["t"] < CACHE_TTL_SEC and _cache["items"]:
        items, counts = _cache["items"], _cache["counts"]
    else:
        data = await _load_all()
        items, counts = data["items"], data["counts"]
        _cache = {"t": now, "items": items, "counts": counts}

    filtered = items
    if source:
        filtered = [x for x in filtered if x["source"].lower() == source.lower()]
    if q:
        qq = q.lower()
        filtered = [x for x in filtered if qq in (x["title"].lower() + x["summary"].lower())]

    if debug:
        return {"counts": counts, "items": filtered[:limit]}
    return filtered[:limit]
