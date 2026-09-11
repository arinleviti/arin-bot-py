import time
from collections import defaultdict
from fastapi import HTTPException

RATE_LIMIT = 10          # max requests
WINDOW_SECONDS = 60      # per this many seconds
MAX_TRACKED_SESSIONS = 50  # hard cap on how many sessionIds we keep in memory at once

request_log: dict[str, list[float]] = defaultdict(list)


def _evict_oldest_sessions():
    # Only trims when we're actually over the cap — cheap no-op the rest of the time.
    if len(request_log) <= MAX_TRACKED_SESSIONS:
        return

    # For each session, its "last active" time is the most recent timestamp in its list.
    # Sort sessions oldest-last-active first, then drop enough to get back under the cap.
    sessions_by_recency = sorted(
        request_log.items(),
        key=lambda item: max(item[1]) if item[1] else 0,
    )

    num_to_evict = len(request_log) - MAX_TRACKED_SESSIONS
    for session_id, _ in sessions_by_recency[:num_to_evict]:
        del request_log[session_id]


def check_rate_limit(session_id: str):
    now = time.time()
    timestamps = request_log[session_id]

    # keep only timestamps from within the last WINDOW_SECONDS —
    # this is what makes it a *sliding* window instead of a permanent cap
    recent = [t for t in timestamps if now - t < WINDOW_SECONDS]

    if len(recent) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, slow down")

    recent.append(now)
    request_log[session_id] = recent

    # Cap total memory growth — only matters if far more unique sessionIds show up
    # than expected (e.g. a burst of bot traffic), since normal usage stays well
    # under MAX_TRACKED_SESSIONS.
    _evict_oldest_sessions()