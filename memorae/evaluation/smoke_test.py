import sys
sys.path.insert(0, '.')
from core.event_store import load_events, EventStore, HAS_BM25
from datetime import datetime, timezone

events = load_events('../memorae_mock_events.json')
now = datetime(2026, 4, 13, 3, 0, tzinfo=timezone.utc)
store = EventStore(events, now)
print("BM25 installed:", HAS_BM25)
print("BM25 index type:", type(store._bm25).__name__)
res = store.retrieve(keywords=["uie", "proposal", "nina"], top_k=3)
for r in res:
    bd = r.breakdown
    print("score=%.3f keys=%s" % (r.score, list(bd.keys())))
    print("  bm25=%.3f kw=%.3f" % (bd.get("bm25", -1), bd.get("keyword_overlap", -1)))
    print(" ", r.event.content[:60])

# Also verify API imports cleanly
try:
    import api
    print("\nAPI import OK")
except Exception as e:
    print("\nAPI import ERROR:", e)
