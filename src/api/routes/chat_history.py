"""
src/api/routes/chat_history.py
===============================
AI Chat session history — save, list, load, and delete per-user conversations.

WHAT:
    Persists AI chat conversations in PostgreSQL so users can return to
    a previous session, browse past Q&A, and delete old conversations.

WHY per-user (not public):
    Chat sessions may contain sensitive or proprietary research questions
    (e.g. a museum curator asking about a coin they are trying to authenticate).
    History must be scoped to the authenticated user only.

ENDPOINTS:
    GET    /api/chat/history         — list sessions (newest first, no messages body)
    POST   /api/chat/history         — save / update a session
    GET    /api/chat/history/{id}    — load a single session with full messages
    DELETE /api/chat/history/{id}    — delete a session
"""


import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.auth.deps import get_current_user
from src.api.db.models  import ChatSession, User
from src.api.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat History"])


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ChatMessageIn(BaseModel):
    """
    A single chat message as stored/transmitted.

    Role is either "user" or "assistant".  Sources and provider are optional
    (assistant messages have them; user messages do not).
    """
    id:        str
    role:      str
    content:   str
    sources:   list[dict[str, Any]] = Field(default_factory=list)
    provider:  str | None = None
    error:     bool       = False
    userQuery: str | None = None


class SaveSessionRequest(BaseModel):
    """
    Body for POST /api/chat/history.

    title    — display name (typically the first user message, truncated).
    messages — full array of messages in the conversation.
    """
    title:    str = Field(..., min_length=1, max_length=200)
    messages: list[ChatMessageIn] = Field(..., min_length=1)


class SessionSummary(BaseModel):
    """
    One row in the history list — no messages, just metadata for the sidebar.
    """
    id:            str
    title:         str
    message_count: int
    created_at:    str
    updated_at:    str


class SessionDetail(SessionSummary):
    """
    Full session, including all messages — returned for GET /api/chat/history/{id}.
    """
    messages: list[dict[str, Any]]


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_summary(row: ChatSession) -> SessionSummary:
    return SessionSummary(
        id            = row.id,
        title         = row.title,
        message_count = len(row.messages),
        created_at    = row.created_at.isoformat(),
        updated_at    = row.updated_at.isoformat(),
    )


def _to_detail(row: ChatSession) -> SessionDetail:
    return SessionDetail(
        id            = row.id,
        title         = row.title,
        message_count = len(row.messages),
        created_at    = row.created_at.isoformat(),
        updated_at    = row.updated_at.isoformat(),
        messages      = list(row.messages),
    )


# ── GET /api/chat/history ─────────────────────────────────────────────────────

@router.get("/history", response_model=list[SessionSummary])
async def list_chat_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
) -> list[SessionSummary]:
    """
    GET /api/chat/history

    Returns all saved chat sessions for the authenticated user, newest first.
    Excludes the messages body (use GET /api/chat/history/{id} to load a session).

    Authentication: Bearer JWT required.
    """
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(100)           # safety cap — 100 sessions is more than enough
    )
    rows = result.scalars().all()
    return [_to_summary(r) for r in rows]


# ── POST /api/chat/history ────────────────────────────────────────────────────

@router.post("/history", response_model=SessionSummary, status_code=201)
async def save_chat_session(
    body:         SaveSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
) -> SessionSummary:
    """
    POST /api/chat/history

    Creates a new saved chat session for the authenticated user.

    The frontend calls this when the user clicks "New chat" (to checkpoint
    the current conversation) or when the page is about to unload.

    Body fields:
        title    — first user message, truncated to 200 chars
        messages — complete message array

    Returns the saved session summary (with generated id).
    """
    session = ChatSession(
        user_id  = current_user.id,
        title    = body.title[:200],
        messages = [m.model_dump() for m in body.messages],
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    logger.info("Chat session saved: %s (user=%s, msgs=%d)", session.id, current_user.id, len(body.messages))
    return _to_summary(session)


# ── GET /api/chat/history/{session_id} ────────────────────────────────────────

@router.get("/history/{session_id}", response_model=SessionDetail)
async def get_chat_session(
    session_id:   str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
) -> SessionDetail:
    """
    GET /api/chat/history/{session_id}

    Returns a single saved session with the full messages array.
    Used when the user clicks a history item in the sidebar.

    Authentication: Bearer JWT required.
    403 if the session belongs to a different user (no cross-user access).
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied.")
    return _to_detail(session)


# ── DELETE /api/chat/history/{session_id} ─────────────────────────────────────

@router.delete("/history/{session_id}", status_code=204, response_class=Response)
async def delete_chat_session(
    session_id:   str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession   = Depends(get_db),
) -> None:
    """
    DELETE /api/chat/history/{session_id}

    Deletes a saved chat session.  Returns 204 No Content on success,
    404 if the session doesn't exist, 403 if it belongs to another user.

    Authentication: Bearer JWT required.
    """
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied.")
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await db.commit()
    logger.info("Chat session deleted: %s (user=%s)", session_id, current_user.id)

