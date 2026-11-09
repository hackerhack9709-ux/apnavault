"""ApnaVault - personal cloud storage for photos, videos and files.

Runs as a small FastAPI service. Data lives on disk (./data) plus a SQLite
database, so it survives a phone factory reset: the phone is only a client.
"""
from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import time
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import bcrypt
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("VAULT_DATA_DIR", BASE_DIR / "data"))
FILES_DIR = DATA_DIR / "files"
THUMBS_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "vault.db"

SESSION_COOKIE = "vault_session"
SESSION_TTL = 60 * 60 * 24 * 30  # 30 days
CHUNK = 1024 * 1024
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".heic"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"}

for d in (FILES_DIR, THUMBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ApnaVault")

# Cross-origin: set ALLOWED_ORIGINS to your Netlify URL(s), comma-separated.
# Example: ALLOWED_ORIGINS="https://myvault.netlify.app"
_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# When the frontend lives on another domain (Netlify), the session cookie must be
# SameSite=None; Secure. Set COOKIE_SECURE=1 in that case.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"


def set_session_cookie(resp: Response, token: str) -> None:
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL,
        httponly=True,
        samesite="none" if COOKIE_SECURE else "lax",
        secure=COOKIE_SECURE,
        path="/",
    )


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with closing(db()) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS files (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                orig_name   TEXT NOT NULL,
                stored_name TEXT UNIQUE NOT NULL,
                size        INTEGER NOT NULL,
                mime        TEXT NOT NULL,
                kind        TEXT NOT NULL,
                thumb       TEXT,
                starred     INTEGER NOT NULL DEFAULT 0,
                deleted     INTEGER NOT NULL DEFAULT 0,
                uploaded_at TEXT NOT NULL,
                deleted_at  TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id, deleted);
            """
        )
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- #
# Passwords & sessions
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return bcrypt.hashpw(digest, bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    try:
        return bcrypt.checkpw(digest, hashed.encode())
    except ValueError:
        return False


def create_session(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
        (token_hash, user_id, now, now + SESSION_TTL),
    )
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    conn.commit()
    return token


def current_user(request: Request) -> Optional[sqlite3.Row]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with closing(db()) as conn:
        row = conn.execute(
            """
            SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > ?
            """,
            (token_hash, time.time()),
        ).fetchone()
    return row


def require_user(request: Request) -> sqlite3.Row:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="login required")
    return user


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def classify(name: str, mime: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT or mime.startswith("image/"):
        return "image"
    if ext in VIDEO_EXT or mime.startswith("video/"):
        return "video"
    return "file"


def safe_name(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^\w.\- ()\[\]]+", "_", name).strip() or "file"
    return name[:180]


def make_thumbnail(src: Path, dest: Path) -> bool:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((480, 480))
            im.save(dest, "JPEG", quality=82)
        return True
    except Exception:
        return False


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def file_row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["size_h"] = human_size(d["size"])
    d["has_thumb"] = bool(d["thumb"])
    return d


def iter_file(path: Path, start: int, end: int) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def range_response(path: Path, request: Request, mime: str, download_name: Optional[str] = None):
    size = path.stat().st_size
    range_header = request.headers.get("range")
    disposition = "attachment" if download_name else "inline"
    fname = download_name or path.name
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f'{disposition}; filename="{fname}"',
    }
    if range_header:
        m = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if m:
            start = int(m.group(1)) if m.group(1) else 0
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end:
                raise HTTPException(status_code=416, detail="bad range")
            length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            headers["Content-Length"] = str(length)
            return StreamingResponse(
                iter_file(path, start, end), status_code=206, headers=headers, media_type=mime
            )
    headers["Content-Length"] = str(size)
    return StreamingResponse(iter_file(path, 0, size - 1), headers=headers, media_type=mime)


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _startup() -> None:
    init_db()


# --------------------------------------------------------------------------- #
# Embedded UI (no separate files needed)
# --------------------------------------------------------------------------- #
CSS = '/* ============================================================\n   APNAVAULT — "The Archive"\n   A vintage darkroom / analog photo-archive aesthetic.\n   Warm ink base, amber safelight, brass rules, aged paper cards,\n   typewriter metadata, editorial serif headlines, film grain.\n   ============================================================ */\n\n:root {\n  --ink:      #14100b;\n  --ink-2:    #1c1710;\n  --ink-3:    #241d13;\n  --paper:    #e8dfc9;\n  --paper-2:  #d9ccac;\n  --paper-3:  #f4ecd9;\n  --brass:    #c8a04a;\n  --brass-2:  #e6c778;\n  --stamp:    #a8392a;\n  --stamp-2:  #c14d3a;\n  --text:     #efe7d4;\n  --muted:    #9a8d74;\n  --line:     #3a3121;\n  --shadow:   0 22px 50px -20px rgba(0,0,0,.85);\n  --font-display: "Fraunces", Georgia, "Times New Roman", serif;\n  --font-body:    "Spectral", Georgia, serif;\n  --font-mono:    "Courier Prime", "Courier New", monospace;\n}\n\n* { box-sizing: border-box; }\nhtml { scroll-behavior: smooth; }\nbody {\n  margin: 0;\n  min-height: 100vh;\n  color: var(--text);\n  font-family: var(--font-body);\n  font-size: 16px;\n  background-color: var(--ink);\n  background-image:\n    radial-gradient(1100px 620px at 50% -8%, rgba(200,160,74,.20), transparent 62%),\n    radial-gradient(760px 520px at 88% 108%, rgba(168,57,42,.14), transparent 60%),\n    radial-gradient(600px 400px at 6% 96%, rgba(200,160,74,.08), transparent 60%);\n  position: relative;\n}\n\n/* film grain + vignette, layered over everything */\nbody::before {\n  content: "";\n  position: fixed; inset: 0; z-index: 1; pointer-events: none;\n  background: radial-gradient(120% 120% at 50% 45%, transparent 55%, rgba(0,0,0,.55) 100%);\n}\nbody::after {\n  content: "";\n  position: fixed; inset: -50%; z-index: 2; pointer-events: none;\n  opacity: .05; mix-blend-mode: overlay;\n  background-image: url("data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'180\' height=\'180\'%3E%3Cfilter id=\'n\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.85\' numOctaves=\'4\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23n)\'/%3E%3C/svg%3E");\n  animation: grain 900ms steps(3) infinite;\n}\n@keyframes grain {\n  0%   { transform: translate(0,0); }\n  33%  { transform: translate(-4%, 2%); }\n  66%  { transform: translate(3%, -3%); }\n  100% { transform: translate(0,0); }\n}\n\n.hidden { display: none !important; }\n.muted { color: var(--muted); }\n.small { font-size: .78rem; }\na { color: var(--brass-2); }\n\n/* ---------------- type helpers ---------------- */\n.eyebrow {\n  font-family: var(--font-mono);\n  font-size: .68rem;\n  letter-spacing: .34em;\n  text-transform: uppercase;\n  color: var(--brass);\n}\n.rule { height: 1px; background: linear-gradient(90deg, transparent, var(--line) 12%, var(--line) 88%, transparent); border: 0; margin: 0; }\n\n/* ---------------- buttons ---------------- */\n.btn {\n  font-family: var(--font-mono);\n  font-size: .74rem;\n  letter-spacing: .12em;\n  text-transform: uppercase;\n  color: var(--text);\n  background: linear-gradient(180deg, var(--ink-3), var(--ink-2));\n  border: 1px solid var(--line);\n  border-radius: 2px;\n  padding: 10px 15px;\n  cursor: pointer;\n  position: relative;\n  transition: transform .14s ease, border-color .14s ease, color .14s ease, box-shadow .14s ease;\n}\n.btn:hover { border-color: var(--brass); color: var(--brass-2); transform: translateY(-1px); box-shadow: 0 8px 18px -12px rgba(200,160,74,.9); }\n.btn:active { transform: translateY(0); }\n.btn.primary {\n  color: #1a130a;\n  background: linear-gradient(180deg, var(--brass-2), var(--brass));\n  border-color: var(--brass-2);\n  font-weight: 700;\n}\n.btn.primary:hover { color: #1a130a; filter: brightness(1.06); box-shadow: 0 10px 26px -12px rgba(230,199,120,.95); }\n.btn.ghost { background: transparent; }\n.btn.danger { color: #f0b6ac; border-color: #5c2b23; }\n.btn.danger:hover { color: #fff; border-color: var(--stamp-2); box-shadow: 0 8px 18px -12px rgba(193,77,58,.9); }\n.btn.small { padding: 7px 11px; font-size: .68rem; }\n.btn.full { width: 100%; padding: 14px; font-size: .82rem; }\n\n/* ============================================================\n   LOGIN\n   ============================================================ */\n.auth-body { display: grid; place-items: center; padding: 30px 20px; }\n.auth-card {\n  position: relative; z-index: 3;\n  width: 100%; max-width: 430px;\n  background:\n    linear-gradient(180deg, rgba(255,255,255,.35), transparent 30%),\n    linear-gradient(180deg, var(--paper-3), var(--paper));\n  color: #241d13;\n  border-radius: 3px;\n  padding: 30px 28px 26px;\n  box-shadow: var(--shadow), 0 0 0 1px rgba(0,0,0,.5);\n  animation: cardIn .7s cubic-bezier(.2,.8,.25,1) both;\n}\n/* torn top edge */\n.auth-card::before {\n  content: ""; position: absolute; left: 0; right: 0; top: -9px; height: 10px;\n  background: var(--paper-3);\n  clip-path: polygon(0 100%,2% 30%,5% 80%,8% 20%,12% 70%,16% 10%,21% 65%,26% 25%,31% 75%,36% 15%,42% 60%,48% 20%,54% 70%,60% 15%,66% 62%,72% 22%,78% 72%,84% 18%,90% 66%,95% 26%,100% 72%,100% 100%);\n}\n/* brass clip */\n.auth-card::after {\n  content: ""; position: absolute; top: -14px; left: 50%; transform: translateX(-50%);\n  width: 74px; height: 22px; border-radius: 3px;\n  background: linear-gradient(180deg, var(--brass-2), #8f6f2c);\n  box-shadow: 0 4px 10px -3px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.5);\n}\n@keyframes cardIn { from { opacity: 0; transform: translateY(18px) rotate(-.6deg); } to { opacity: 1; transform: none; } }\n\n.brand { text-align: center; margin-bottom: 6px; }\n.logo {\n  width: 66px; height: 66px; margin: 6px auto 12px;\n  display: grid; place-items: center; font-size: 30px;\n  border-radius: 50%;\n  background: radial-gradient(circle at 35% 30%, #2b2416, var(--ink));\n  border: 2px solid var(--brass);\n  box-shadow: 0 0 0 4px rgba(200,160,74,.16), inset 0 0 22px rgba(200,160,74,.28);\n}\n.brand h1 {\n  font-family: var(--font-display);\n  font-weight: 900; font-variation-settings: "opsz" 100, "SOFT" 40;\n  font-size: 2.15rem; letter-spacing: -.02em; margin: 2px 0 6px; color: #1c1509;\n}\n.brand p { margin: 0; color: #5c5140; font-size: .86rem; font-style: italic; }\n\n.banner {\n  margin-top: 16px; padding: 11px 13px;\n  font-family: var(--font-mono); font-size: .72rem; letter-spacing: .02em;\n  color: #6d2a1f;\n  background: rgba(168,57,42,.08);\n  border: 1px dashed rgba(168,57,42,.5);\n  border-radius: 2px;\n}\n\n.tabs { display: flex; gap: 0; margin: 20px 0 4px; border: 1px solid #b9ac8d; border-radius: 2px; overflow: hidden; }\n.tab {\n  flex: 1; cursor: pointer;\n  font-family: var(--font-mono); font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;\n  padding: 11px; background: transparent; color: #6c5f49; border: 0;\n  transition: background .15s, color .15s;\n}\n.tab.active { background: #241d13; color: var(--brass-2); }\n\nlabel {\n  display: block; margin: 15px 0 6px;\n  font-family: var(--font-mono); font-size: .66rem; letter-spacing: .2em; text-transform: uppercase;\n  color: #6c5f49;\n}\ninput, select {\n  width: 100%; font-family: var(--font-mono); font-size: .92rem;\n  color: #241d13; background: transparent;\n  border: 0; border-bottom: 1.5px solid #9c8f72; border-radius: 0;\n  padding: 9px 2px;\n  transition: border-color .15s, background .15s;\n}\ninput::placeholder { color: #a99c80; }\ninput:focus, select:focus { outline: none; border-bottom-color: var(--stamp); background: rgba(200,160,74,.08); }\n#auth-form .btn { margin-top: 22px; }\n.msg { min-height: 20px; margin: 12px 0 0; font-family: var(--font-mono); font-size: .76rem; }\n.msg.error { color: #8f2f21; }\n.msg.ok { color: #3f6b2f; }\n.auth-card .muted.small { color: #7a6d55; }\n\n/* ============================================================\n   APP SHELL\n   ============================================================ */\n.topbar {\n  position: sticky; top: 0; z-index: 20;\n  display: flex; align-items: center; justify-content: space-between; gap: 14px;\n  padding: 12px 22px; flex-wrap: wrap;\n  background: rgba(18,14,9,.92);\n  backdrop-filter: blur(10px);\n  border-bottom: 1px solid var(--line);\n  box-shadow: 0 1px 0 rgba(200,160,74,.22);\n}\n.brand-row { display: flex; align-items: center; gap: 12px; }\n.logo.small {\n  width: 42px; height: 42px; font-size: 19px; border-radius: 50%; margin: 0;\n  display: grid; place-items: center;\n  background: radial-gradient(circle at 35% 30%, #2b2416, var(--ink));\n  border: 1.5px solid var(--brass);\n  box-shadow: inset 0 0 14px rgba(200,160,74,.3);\n}\n.brand-row strong {\n  display: block; font-family: var(--font-display); font-weight: 900;\n  font-size: 1.12rem; letter-spacing: .01em; color: var(--paper);\n}\n#who { font-family: var(--font-mono); font-size: .7rem; letter-spacing: .1em; color: var(--brass); }\n.top-actions { display: flex; gap: 8px; }\n\n.wrap { position: relative; z-index: 3; max-width: 1200px; margin: 0 auto; padding: 26px 22px 90px; }\n\n/* ---------------- hero / intake ---------------- */\n.hero { display: grid; grid-template-columns: 1.05fr 1fr; gap: 26px; align-items: stretch; margin-bottom: 22px; }\n.hero-left { display: flex; flex-direction: column; justify-content: center; }\n.hero h1 {\n  font-family: var(--font-display); font-weight: 900; font-variation-settings: "opsz" 144, "SOFT" 30;\n  font-size: clamp(2rem, 4.4vw, 3.1rem); line-height: .98; letter-spacing: -.025em;\n  margin: 12px 0 12px; color: var(--paper);\n}\n.hero h1 em { font-style: italic; color: var(--brass-2); }\n.hero .lede { color: var(--muted); max-width: 46ch; line-height: 1.6; font-size: .95rem; margin: 0; }\n\n.stats { display: flex; gap: 0; margin-top: 22px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }\n.stat { flex: 1; padding: 14px 18px 14px 0; }\n.stat + .stat { padding-left: 20px; border-left: 1px solid var(--line); }\n.stat span { display: block; font-family: var(--font-display); font-weight: 700; font-size: 1.7rem; color: var(--brass-2); }\n.stat label { margin: 2px 0 0; color: var(--muted); font-size: .6rem; }\n\n/* ---------------- drop zone ---------------- */\n.drop {\n  position: relative; display: grid; place-items: center; min-height: 240px; cursor: pointer;\n  text-align: center; padding: 26px;\n  border: 1px dashed #6a5c3d;\n  border-radius: 3px;\n  background:\n    linear-gradient(180deg, rgba(200,160,74,.05), transparent),\n    repeating-linear-gradient(45deg, rgba(255,255,255,.012) 0 12px, transparent 12px 24px);\n  transition: border-color .2s, background .2s, transform .2s;\n}\n.drop::before, .drop::after {\n  content: ""; position: absolute; width: 16px; height: 16px; border: 1.5px solid var(--brass);\n  opacity: .7; transition: opacity .2s;\n}\n.drop::before { top: 9px; left: 9px; border-right: 0; border-bottom: 0; }\n.drop::after  { bottom: 9px; right: 9px; border-left: 0; border-top: 0; }\n.drop:hover, .drop.drag {\n  border-color: var(--brass-2); transform: translateY(-2px);\n  background: linear-gradient(180deg, rgba(200,160,74,.14), rgba(200,160,74,.03)), repeating-linear-gradient(45deg, rgba(255,255,255,.02) 0 12px, transparent 12px 24px);\n}\n.drop:hover::before, .drop:hover::after { opacity: 1; }\n.drop-ico { font-size: 34px; margin-bottom: 10px; filter: saturate(.6); }\n.drop-inner strong { display: block; font-family: var(--font-display); font-size: 1.15rem; color: var(--paper); margin-bottom: 4px; }\n.drop-inner .small { color: var(--muted); }\n\n/* ---------------- progress ---------------- */\n.progress {\n  position: relative; height: 30px; margin: 14px 0; overflow: hidden;\n  background: var(--ink-2); border: 1px solid var(--line); border-radius: 2px;\n}\n#progress-bar { height: 100%; width: 0; background: linear-gradient(90deg, #8f6f2c, var(--brass-2)); transition: width .2s; }\n#progress-txt { position: absolute; inset: 0; display: grid; place-items: center; font-family: var(--font-mono); font-size: .72rem; letter-spacing: .1em; color: #1a130a; font-weight: 700; }\n\n/* ---------------- toolbar ---------------- */\n.toolbar { display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; align-items: center; margin: 26px 0 14px; }\n.views { display: flex; gap: 0; border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }\n.view-btn {\n  cursor: pointer; background: transparent; border: 0; color: var(--muted);\n  font-family: var(--font-mono); font-size: .7rem; letter-spacing: .12em; text-transform: uppercase;\n  padding: 10px 15px; transition: background .15s, color .15s;\n}\n.view-btn + .view-btn { border-left: 1px solid var(--line); }\n.view-btn:hover { color: var(--brass-2); }\n.view-btn.active { background: var(--brass); color: #1a130a; font-weight: 700; }\n.filters { display: flex; gap: 10px; align-items: center; }\n.filters select, .filters input {\n  width: auto; background: var(--ink-2); border: 1px solid var(--line); border-radius: 2px;\n  color: var(--text); font-family: var(--font-mono); font-size: .74rem; padding: 9px 12px;\n}\n.filters select { color: var(--text); }\n.filters input { width: 210px; }\n.filters input::placeholder { color: #6f6350; }\n.filters input:focus, .filters select:focus { outline: none; border-color: var(--brass); background: var(--ink-3); }\n\n.bulk {\n  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;\n  padding: 11px 15px; margin-bottom: 14px;\n  background: rgba(200,160,74,.07); border: 1px solid var(--line); border-left: 3px solid var(--brass); border-radius: 2px;\n}\n#sel-count { font-family: var(--font-mono); font-size: .74rem; letter-spacing: .08em; color: var(--brass-2); }\n\n/* ---------------- file grid ---------------- */\n.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr)); gap: 20px; }\n\n.card {\n  position: relative; cursor: pointer;\n  background: linear-gradient(180deg, var(--paper-3), var(--paper));\n  color: #241d13;\n  border-radius: 2px;\n  padding: 9px 9px 0;\n  box-shadow: 0 14px 26px -16px rgba(0,0,0,.9), 0 0 0 1px rgba(0,0,0,.35);\n  transition: transform .2s cubic-bezier(.2,.8,.25,1), box-shadow .2s;\n  animation: cardRise .5s cubic-bezier(.2,.8,.25,1) both;\n}\n@keyframes cardRise { from { opacity: 0; transform: translateY(14px) rotate(0deg); } }\n.card:nth-child(4n+1) { transform: rotate(-.7deg); }\n.card:nth-child(4n+2) { transform: rotate(.5deg); }\n.card:nth-child(4n+3) { transform: rotate(-.3deg); }\n.card:nth-child(4n+4) { transform: rotate(.8deg); }\n.card:hover { transform: translateY(-6px) rotate(0deg) scale(1.015); box-shadow: 0 26px 44px -18px rgba(0,0,0,.95), 0 0 0 1px rgba(200,160,74,.5); z-index: 5; }\n.card.selected { box-shadow: 0 0 0 2px var(--brass), 0 20px 34px -18px rgba(0,0,0,.9); transform: translateY(-4px) rotate(0); }\n\n.thumb, .thumb-ph {\n  width: 100%; aspect-ratio: 1/1; display: block; object-fit: cover;\n  border-radius: 1px;\n  background: #2b2416;\n  filter: saturate(.92) contrast(1.03);\n}\n.thumb-ph { display: grid; place-items: center; font-size: 44px; filter: sepia(.35); background: linear-gradient(180deg, #2b2416, #1a140b); }\n\n/* stamp badge */\n.badge {\n  position: absolute; top: 14px; left: 14px;\n  font-family: var(--font-mono); font-size: .58rem; letter-spacing: .18em; text-transform: uppercase;\n  color: var(--stamp); padding: 3px 7px;\n  border: 1.5px solid var(--stamp);\n  border-radius: 2px;\n  background: rgba(244,236,217,.55);\n  transform: rotate(-6deg);\n  opacity: .85;\n  mix-blend-mode: multiply;\n}\n.star {\n  position: absolute; top: 12px; right: 12px; z-index: 3;\n  width: 30px; height: 30px; cursor: pointer; font-size: 15px; line-height: 1;\n  color: #6b5a30; background: rgba(244,236,217,.7); border: 1px solid rgba(0,0,0,.25);\n  border-radius: 50%; transition: transform .15s, color .15s;\n}\n.star:hover { transform: scale(1.12) rotate(8deg); color: var(--stamp); }\n\n.check-wrap { position: absolute; bottom: 46px; right: 12px; z-index: 3; }\n.check { width: 20px; height: 20px; accent-color: var(--stamp); cursor: pointer; }\n\n.card-meta { padding: 9px 3px 11px; border-top: 1px solid rgba(0,0,0,.14); margin-top: 9px; }\n.card-name {\n  font-family: var(--font-mono); font-size: .74rem; font-weight: 700;\n  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #241d13;\n}\n.card-sub { font-family: var(--font-mono); font-size: .62rem; letter-spacing: .04em; color: #7a6d55; margin-top: 3px; }\n\n/* ---------------- empty ---------------- */\n.empty { text-align: center; padding: 70px 20px; color: var(--muted); }\n.empty-ico { font-size: 46px; margin-bottom: 14px; filter: sepia(.4); }\n#empty-text { font-family: var(--font-mono); font-size: .8rem; letter-spacing: .06em; }\n\n/* ---------------- viewer ---------------- */\n.viewer { position: fixed; inset: 0; z-index: 60; display: flex; flex-direction: column; background: rgba(8,6,3,.96); backdrop-filter: blur(4px); }\n.viewer-close {\n  position: absolute; top: 16px; right: 18px; z-index: 2;\n  width: 42px; height: 42px; border-radius: 50%; cursor: pointer;\n  font-size: 17px; color: var(--paper);\n  background: rgba(200,160,74,.14); border: 1px solid var(--line);\n  transition: background .15s, transform .15s;\n}\n.viewer-close:hover { background: rgba(200,160,74,.28); transform: rotate(90deg); }\n.viewer-body { flex: 1; display: grid; place-items: center; padding: 58px 20px 20px; overflow: auto; }\n.viewer-body img, .viewer-body video {\n  max-width: 92vw; max-height: 72vh; border-radius: 2px; background: #000;\n  box-shadow: 0 30px 60px -24px rgba(0,0,0,1), 0 0 0 1px rgba(200,160,74,.35);\n  animation: cardIn .4s ease both;\n}\n.viewer-bar {\n  display: flex; align-items: center; justify-content: center; gap: 12px; flex-wrap: wrap;\n  padding: 14px; border-top: 1px solid var(--line); background: rgba(18,14,9,.92);\n}\n.viewer-bar .name {\n  font-family: var(--font-mono); font-size: .74rem; color: var(--muted);\n  max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;\n}\n\n/* ---------------- toast ---------------- */\n.toast {\n  position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%); z-index: 100;\n  font-family: var(--font-mono); font-size: .76rem; letter-spacing: .04em;\n  color: #1a130a; background: linear-gradient(180deg, var(--brass-2), var(--brass));\n  border-radius: 2px; padding: 12px 20px;\n  box-shadow: var(--shadow);\n  animation: toastIn .3s cubic-bezier(.2,.8,.25,1) both;\n}\n@keyframes toastIn { from { opacity: 0; transform: translate(-50%, 12px); } }\n\n/* ---------------- responsive ---------------- */\n@media (max-width: 820px) {\n  .hero { grid-template-columns: 1fr; gap: 20px; }\n  .drop { min-height: 180px; }\n  .toolbar { flex-direction: column; align-items: stretch; }\n  .views { justify-content: space-between; }\n  .filters { flex-wrap: wrap; }\n  .filters input { width: 100%; }\n  .grid { grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 14px; }\n}\n@media (max-width: 480px) {\n  .wrap { padding: 20px 14px 80px; }\n  .topbar { padding: 10px 14px; }\n  .grid { grid-template-columns: repeat(2, 1fr); }\n  .brand h1 { font-size: 1.85rem; }\n}\n@media (prefers-reduced-motion: reduce) {\n  *, *::before, *::after { animation: none !important; transition: none !important; }\n}\n'

JS = '\'use strict\';\n\n// API_BASE is empty when the app is served by its own backend (local/Docker).\n// On Netlify, set window.API_BASE in /static/config.js to the backend URL.\nconst API_BASE = (window.API_BASE || \'\').replace(/\\/$/, \'\');\nconst LOGIN_PATH = window.LOGIN_PATH || \'/login\';\nconst api = (p) => API_BASE + p;\nconst apiFetch = (p, opts = {}) =>\n  fetch(api(p), { credentials: \'include\', ...opts });\n\nconst state = {\n  view: \'all\',\n  kind: \'all\',\n  q: \'\',\n  files: [],\n  selected: new Set(),\n};\n\nconst $ = (id) => document.getElementById(id);\nconst grid = $(\'grid\');\nconst empty = $(\'empty\');\nconst bulk = $(\'bulk\');\n\nfunction toast(text) {\n  const el = document.createElement(\'div\');\n  el.className = \'toast\';\n  el.textContent = text;\n  document.body.appendChild(el);\n  setTimeout(() => el.remove(), 3200);\n}\n\nfunction esc(s) {\n  return String(s).replace(/[&<>"\']/g, (c) => (\n    { \'&\': \'&amp;\', \'<\': \'&lt;\', \'>\': \'&gt;\', \'"\': \'&quot;\', "\'": \'&#39;\' }[c]\n  ));\n}\n\nfunction fmtTime(ts) {\n  return ts ? ts.replace(\'T\', \' \').slice(0, 16) : \'\';\n}\n\n// -------------------- Data loading --------------------\nasync function loadFiles() {\n  const params = new URLSearchParams({ view: state.view, kind: state.kind, q: state.q });\n  const res = await apiFetch(\'/api/files?\' + params.toString());\n  if (res.status === 401) { window.location.href = LOGIN_PATH; return; }\n  const data = await res.json();\n  state.files = data.files;\n  state.selected.clear();\n  $(\'stat-count\').textContent = data.stats.count;\n  $(\'stat-size\').textContent = data.stats.size_h;\n  render();\n}\n\nfunction isTrash() { return state.view === \'trash\'; }\n\nfunction render() {\n  grid.innerHTML = \'\';\n  const emptyText = isTrash()\n    ? \'Trash khaali hai.\'\n    : state.q\n      ? \'Is naam se kuch nahi mila.\'\n      : \'Abhi kuch nahi hai. Upar se upload karo.\';\n  $(\'empty-text\').textContent = emptyText;\n  empty.classList.toggle(\'hidden\', state.files.length > 0);\n  updateBulkBar();\n\n  for (const f of state.files) {\n    const card = document.createElement(\'div\');\n    card.className = \'card\';\n    card.dataset.id = f.id;\n\n    let media;\n    if (f.has_thumb) {\n      media = `<img class="thumb" loading="lazy" src="${api(\'/api/thumb/\' + f.id)}" alt="${esc(f.orig_name)}" />`;\n    } else if (f.kind === \'video\') {\n      media = `<div class="thumb-ph">🎬</div>`;\n    } else if (f.kind === \'image\') {\n      media = `<div class="thumb-ph">🖼️</div>`;\n    } else {\n      media = `<div class="thumb-ph">📄</div>`;\n    }\n\n    card.innerHTML = `\n      ${media}\n      <span class="badge">${f.kind}</span>\n      ${f.starred && !isTrash() ? \'<span class="star" style="pointer-events:none">★</span>\' : \'\'}\n      <label class="check-wrap">\n        <input class="check" type="checkbox" data-id="${f.id}" />\n      </label>\n      <div class="card-meta">\n        <div class="card-name" title="${esc(f.orig_name)}">${esc(f.orig_name)}</div>\n        <div class="card-sub">${f.size_h} • ${fmtTime(f.uploaded_at)}</div>\n      </div>`;\n\n    card.addEventListener(\'click\', (e) => {\n      if (e.target.classList.contains(\'check\')) return;\n      openViewer(f);\n    });\n    grid.appendChild(card);\n  }\n}\n\nfunction updateBulkBar() {\n  const n = state.selected.size;\n  bulk.classList.toggle(\'hidden\', n === 0);\n  $(\'sel-count\').textContent = n + \' chuni gayi\';\n  const action = $(\'bulk-action\');\n  if (isTrash()) { action.textContent = \'♻️ Restore karo\'; }\n  else if (state.view === \'starred\') { action.textContent = \'⭐ Star hatao\'; }\n  else { action.textContent = \'🗑️ Trash me daalo\'; }\n}\n\ngrid.addEventListener(\'change\', (e) => {\n  if (!e.target.classList.contains(\'check\')) return;\n  const id = Number(e.target.dataset.id);\n  if (e.target.checked) state.selected.add(id);\n  else state.selected.delete(id);\n  e.target.closest(\'.card\').classList.toggle(\'selected\', e.target.checked);\n  updateBulkBar();\n});\n\n$(\'bulk-clear\').onclick = () => {\n  state.selected.clear();\n  document.querySelectorAll(\'.check\').forEach((c) => { c.checked = false; });\n  document.querySelectorAll(\'.card\').forEach((c) => c.classList.remove(\'selected\'));\n  updateBulkBar();\n};\n\n$(\'bulk-action\').onclick = async () => {\n  const ids = [...state.selected];\n  if (!ids.length) return;\n  let url, body;\n  if (isTrash()) { url = \'/api/restore\'; body = { ids }; }\n  else if (state.view === \'starred\') {\n    await Promise.all(ids.map((id) => apiFetch(`/api/file/${id}/star`, { method: \'POST\' })));\n    toast(\'Star hata diya\'); loadFiles(); return;\n  } else { url = \'/api/trash\'; body = { ids }; }\n  await apiFetch(url, {\n    method: \'POST\',\n    headers: { \'Content-Type\': \'application/json\' },\n    body: JSON.stringify(body),\n  });\n  toast(isTrash() ? \'Wapas aa gaya ♻️\' : \'Trash me daal diya\');\n  loadFiles();\n};\n\n// -------------------- Upload --------------------\nconst picker = $(\'picker\');\nconst drop = $(\'drop\');\nconst progress = $(\'progress\');\nconst progressBar = $(\'progress-bar\');\nconst progressTxt = $(\'progress-txt\');\n\npicker.addEventListener(\'change\', () => {\n  if (picker.files.length) uploadFiles([...picker.files]);\n  picker.value = \'\';\n});\n\n[\'dragenter\', \'dragover\'].forEach((ev) =>\n  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add(\'drag\'); })\n);\n[\'dragleave\', \'drop\'].forEach((ev) =>\n  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove(\'drag\'); })\n);\ndrop.addEventListener(\'drop\', (e) => {\n  if (e.dataTransfer.files.length) uploadFiles([...e.dataTransfer.files]);\n});\n\nfunction uploadFiles(files) {\n  return new Promise((resolve) => {\n    const fd = new FormData();\n    files.forEach((f) => fd.append(\'files\', f));\n    const xhr = new XMLHttpRequest();\n    xhr.open(\'POST\', api(\'/api/upload\'));\n    xhr.withCredentials = true;\n    progress.classList.remove(\'hidden\');\n    progressBar.style.width = \'0%\';\n    progressTxt.textContent = \'0%\';\n\n    xhr.upload.onprogress = (e) => {\n      if (!e.lengthComputable) return;\n      const pct = Math.round((e.loaded / e.total) * 100);\n      progressBar.style.width = pct + \'%\';\n      progressTxt.textContent = pct + \'%  (\' + files.length + \' file)\';\n    };\n    xhr.onload = () => {\n      progress.classList.add(\'hidden\');\n      if (xhr.status === 200) {\n        const d = JSON.parse(xhr.responseText || \'{}\');\n        toast((d.count || files.length) + \' file upload ho gayi ✅\');\n        loadFiles();\n      } else if (xhr.status === 401) {\n        window.location.href = LOGIN_PATH;\n      } else {\n        toast(\'Upload fail hua ❌\');\n      }\n      resolve();\n    };\n    xhr.onerror = () => {\n      progress.classList.add(\'hidden\');\n      toast(\'Network error — upload fail ❌\');\n      resolve();\n    };\n    xhr.send(fd);\n  });\n}\n\n// -------------------- Viewer --------------------\nconst viewer = $(\'viewer\');\nconst viewerBody = $(\'viewer-body\');\nconst viewerBar = $(\'viewer-bar\');\n\nfunction openViewer(f) {\n  viewerBody.innerHTML = \'\';\n  viewerBar.innerHTML = \'\';\n  const src = api(`/api/file/${f.id}/view`);\n\n  if (f.kind === \'image\') {\n    const img = document.createElement(\'img\');\n    img.src = src;\n    img.alt = f.orig_name;\n    viewerBody.appendChild(img);\n  } else if (f.kind === \'video\') {\n    const v = document.createElement(\'video\');\n    v.src = src;\n    v.controls = true;\n    v.autoplay = true;\n    v.playsInline = true;\n    viewerBody.appendChild(v);\n  } else {\n    const box = document.createElement(\'div\');\n    box.style.textAlign = \'center\';\n    box.innerHTML = `<div style="font-size:64px">📄</div><p>${esc(f.orig_name)}</p>\n      <p class="muted small">Ye file browser me nahi khulti. Download karke kholo.</p>`;\n    viewerBody.appendChild(box);\n  }\n\n  const name = document.createElement(\'span\');\n  name.className = \'name\';\n  name.textContent = f.orig_name;\n  viewerBar.appendChild(name);\n\n  const dl = document.createElement(\'a\');\n  dl.className = \'btn small\';\n  dl.href = api(`/api/file/${f.id}/download`);\n  dl.textContent = \'⬇️ Download\';\n  viewerBar.appendChild(dl);\n\n  if (!isTrash()) {\n    const star = document.createElement(\'button\');\n    star.className = \'btn small\';\n    star.textContent = f.starred ? \'★ Star hatao\' : \'☆ Favourite\';\n    star.onclick = async () => {\n      const r = await apiFetch(`/api/file/${f.id}/star`, { method: \'POST\' });\n      const d = await r.json();\n      f.starred = d.starred ? 1 : 0;\n      star.textContent = f.starred ? \'★ Star hatao\' : \'☆ Favourite\';\n      loadFiles();\n    };\n    viewerBar.appendChild(star);\n\n    const tr = document.createElement(\'button\');\n    tr.className = \'btn small danger\';\n    tr.textContent = \'🗑️ Trash\';\n    tr.onclick = async () => {\n      await apiFetch(`/api/file/${f.id}/trash`, { method: \'POST\' });\n      closeViewer();\n      toast(\'Trash me daal diya\');\n      loadFiles();\n    };\n    viewerBar.appendChild(tr);\n  } else {\n    const rs = document.createElement(\'button\');\n    rs.className = \'btn small\';\n    rs.textContent = \'♻️ Restore\';\n    rs.onclick = async () => {\n      await apiFetch(`/api/file/${f.id}/restore`, { method: \'POST\' });\n      closeViewer();\n      toast(\'Wapas aa gaya ♻️\');\n      loadFiles();\n    };\n    viewerBar.appendChild(rs);\n\n    const del = document.createElement(\'button\');\n    del.className = \'btn small danger\';\n    del.textContent = \'❌ Hamesha ke liye delete\';\n    del.onclick = async () => {\n      if (!confirm(\'Pakka? Ye file hamesha ke liye delete ho jayegi.\')) return;\n      await apiFetch(`/api/file/${f.id}`, { method: \'DELETE\' });\n      closeViewer();\n      toast(\'Delete ho gayi\');\n      loadFiles();\n    };\n    viewerBar.appendChild(del);\n  }\n\n  viewer.classList.remove(\'hidden\');\n}\n\nfunction closeViewer() {\n  viewer.classList.add(\'hidden\');\n  viewerBody.innerHTML = \'\';\n}\n$(\'viewer-close\').onclick = closeViewer;\nviewer.addEventListener(\'click\', (e) => { if (e.target === viewer) closeViewer(); });\ndocument.addEventListener(\'keydown\', (e) => { if (e.key === \'Escape\') closeViewer(); });\n\n// -------------------- Controls --------------------\ndocument.querySelectorAll(\'.view-btn\').forEach((btn) => {\n  btn.onclick = () => {\n    document.querySelectorAll(\'.view-btn\').forEach((b) => b.classList.remove(\'active\'));\n    btn.classList.add(\'active\');\n    state.view = btn.dataset.view;\n    $(\'empty-trash\').classList.toggle(\'hidden\', !isTrash());\n    loadFiles();\n  };\n});\n\n$(\'empty-trash\').onclick = async () => {\n  if (!confirm(\'Trash ki saari files hamesha ke liye delete ho jayengi. Pakka?\')) return;\n  const r = await apiFetch(\'/api/empty-trash\', { method: \'POST\' });\n  const d = await r.json();\n  toast((d.count || 0) + \' file delete ho gayi\');\n  loadFiles();\n};\n\n$(\'kind\').onchange = (e) => { state.kind = e.target.value; loadFiles(); };\nlet searchTimer;\n$(\'search\').oninput = (e) => {\n  clearTimeout(searchTimer);\n  searchTimer = setTimeout(() => { state.q = e.target.value.trim(); loadFiles(); }, 250);\n};\n\n$(\'logout-btn\').onclick = async () => {\n  await apiFetch(\'/api/logout\', { method: \'POST\' });\n  window.location.href = LOGIN_PATH;\n};\n\n$(\'recover-btn\').onclick = () => {\n  toast(\'Recovery ZIP ban rahi hai... thodi der lagegi\');\n  window.location.href = api(\'/api/recover\');\n};\n\n// -------------------- Boot --------------------\napiFetch(\'/api/me\')\n  .then((r) => (r.ok ? r.json() : Promise.reject()))\n  .then((d) => { $(\'who\').textContent = \'@\' + d.username; })\n  .catch(() => { window.location.href = LOGIN_PATH; });\n\nloadFiles();\n'

LOGIN_HTML = '<!DOCTYPE html>\n<html lang="hi">\n<head>\n<meta charset="utf-8" />\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />\n<title>ApnaVault — Login</title>\n<link rel="preconnect" href="https://fonts.googleapis.com" />\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400;1,9..144,900&family=Spectral:ital,wght@0,400;0,600;1,400&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet" />\n<link rel="stylesheet" href="/static/style.css" />\n</head>\n<body class="auth-body">\n  <div class="auth-card">\n    <div class="brand">\n      <div class="logo">🔒</div>\n      <h1>ApnaVault</h1>\n      <p>Est. 2026 · aapka zaati archive</p>\n    </div>\n\n    <div id="setup-banner" class="banner __SETUP_HIDDEN__">\n      Pehli baar? Apna naya account banao — username + password chuno.\n    </div>\n\n    <div class="tabs">\n      <button id="tab-login" class="tab active" type="button">Login</button>\n      <button id="tab-register" class="tab" type="button">Naya Account</button>\n    </div>\n\n    <form id="auth-form" autocomplete="on">\n      <label for="username">Username</label>\n      <input id="username" name="username" type="text" placeholder="jaise: rahul123" required />\n\n      <label for="password">Password</label>\n      <input id="password" name="password" type="password" placeholder="kam se kam 6 character" required />\n\n      <button id="submit-btn" class="btn primary full" type="submit">Login karo</button>\n    </form>\n\n    <p id="msg" class="msg"></p>\n    <p class="muted small">Tip: password yaad rakho — isi se aapka poora data khulta hai.</p>\n  </div>\n  <script>\n    const form = document.getElementById(\'auth-form\');\n    const msg = document.getElementById(\'msg\');\n    const submitBtn = document.getElementById(\'submit-btn\');\n    const tabLogin = document.getElementById(\'tab-login\');\n    const tabReg = document.getElementById(\'tab-register\');\n    let mode = \'login\';\n\n    function setMode(next) {\n      mode = next;\n      tabLogin.classList.toggle(\'active\', mode === \'login\');\n      tabReg.classList.toggle(\'active\', mode === \'register\');\n      submitBtn.textContent = mode === \'login\' ? \'Login karo\' : \'Account banao\';\n      msg.textContent = \'\';\n    }\n    tabLogin.onclick = () => setMode(\'login\');\n    tabReg.onclick = () => setMode(\'register\');\n\n    form.addEventListener(\'submit\', async (e) => {\n      e.preventDefault();\n      msg.className = \'msg\';\n      msg.textContent = \'Ruko...\';\n      const body = new FormData(form);\n      const url = mode === \'login\' ? \'/api/login\' : \'/api/register\';\n      try {\n        const res = await fetch(url, { method: \'POST\', body });\n        const data = await res.json().catch(() => ({}));\n        if (!res.ok) throw new Error(data.detail || \'Kuch galat hua.\');\n        window.location.href = \'/\';\n      } catch (err) {\n        msg.className = \'msg error\';\n        msg.textContent = err.message;\n      }\n    });\n  </script>\n</body>\n</html>\n'

APP_HTML = '<!DOCTYPE html>\n<html lang="hi">\n<head>\n<meta charset="utf-8" />\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />\n<title>ApnaVault — Aapka Archive</title>\n<link rel="preconnect" href="https://fonts.googleapis.com" />\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,400;1,9..144,900&family=Spectral:ital,wght@0,400;0,600;1,400&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet" />\n<link rel="stylesheet" href="/static/style.css" />\n</head>\n<body>\n<header class="topbar">\n  <div class="brand-row">\n    <div class="logo small">🔒</div>\n    <div>\n      <strong>ApnaVault</strong>\n      <span class="muted small" id="who">—</span>\n    </div>\n  </div>\n  <div class="top-actions">\n    <button id="recover-btn" class="btn">⬇️ Sab Recovery</button>\n    <button id="logout-btn" class="btn ghost">Logout</button>\n  </div>\n</header>\n\n<main class="wrap">\n  <section class="hero">\n    <div class="hero-left">\n      <span class="eyebrow">Zaati digital archive · No. 001</span>\n      <h1>Jo bhi anmol hai,<br /><em>yahan sambhal ke</em> rakho.</h1>\n      <p class="lede">Photos, videos aur zaroori files ek surakshit jagah. Phone reset ho jaye,\n        phone kho jaye — aapka data yahin mehfooz rehta hai. Naye device se login karo aur sab wapas paa lo.</p>\n      <div class="stats">\n        <div class="stat"><span id="stat-count">0</span><label>files rakhi hui</label></div>\n        <div class="stat"><span id="stat-size">0 B</span><label>kul storage</label></div>\n      </div>\n    </div>\n    <label id="drop" class="drop">\n      <input id="picker" type="file" multiple hidden />\n      <div class="drop-inner">\n        <div class="drop-ico">🗂️</div>\n        <strong>Yahan chhodo, ya tap karo</strong>\n        <span class="small">Photos, videos aur koi bhi file — ek saath bahut saari</span>\n      </div>\n    </label>\n  </section>\n\n  <div id="progress" class="progress hidden"><div id="progress-bar"></div><span id="progress-txt">0%</span></div>\n\n  <nav class="toolbar">\n    <div class="views">\n      <button class="view-btn active" data-view="all">Sab</button>\n      <button class="view-btn" data-view="starred">⭐ Favourites</button>\n      <button class="view-btn" data-view="trash">🗑️ Trash</button>\n    </div>\n    <div class="filters">\n      <select id="kind">\n        <option value="all">Sab type</option>\n        <option value="image">Photos</option>\n        <option value="video">Videos</option>\n        <option value="file">Files</option>\n      </select>\n      <input id="search" type="search" placeholder="Naam se dhoondo..." />\n      <button id="empty-trash" class="btn small danger hidden">Trash khaali karo</button>\n    </div>\n  </nav>\n\n  <div id="bulk" class="bulk hidden">\n    <span id="sel-count">0 chuni gayi</span>\n    <button id="bulk-action" class="btn small">Karo</button>\n    <button id="bulk-clear" class="btn small ghost">Hatao</button>\n  </div>\n\n  <div id="empty" class="empty hidden">\n    <div class="empty-ico">📭</div>\n    <p id="empty-text">Abhi kuch nahi hai. Upar se upload karo.</p>\n  </div>\n\n  <section id="grid" class="grid"></section>\n</main>\n\n<div id="viewer" class="viewer hidden">\n  <button id="viewer-close" class="viewer-close">✕</button>\n  <div id="viewer-body" class="viewer-body"></div>\n  <div id="viewer-bar" class="viewer-bar"></div>\n</div>\n\n<script src="/static/app.js"></script>\n</body>\n</html>\n'


@app.get("/static/style.css")
def _style():
    return Response(CSS, media_type="text/css")


@app.get("/static/app.js")
def _script():
    return Response(JS, media_type="application/javascript")


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login")
    return HTMLResponse(APP_HTML)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/")
    with closing(db()) as conn:
        count = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
    return HTMLResponse(
        LOGIN_HTML.replace("__SETUP_HIDDEN__", "" if count == 0 else "hidden")
    )


@app.post("/api/register")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    if len(username) < 3:
        raise HTTPException(400, "Username kam se kam 3 letter ka hona chahiye.")
    if len(password) < 6:
        raise HTTPException(400, "Password kam se kam 6 character ka hona chahiye.")
    with closing(db()) as conn:
        exists = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            raise HTTPException(400, "Ye username pehle se liya hua hai.")
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?,?,?)",
            (username, hash_password(password), now_iso()),
        )
        conn.commit()
        user_id = cur.lastrowid
        token = create_session(conn, user_id)
    resp = JSONResponse({"ok": True})
    set_session_cookie(resp, token)
    return resp


@app.post("/api/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with closing(db()) as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        if not user or not verify_password(password, user["password_hash"]):
            raise HTTPException(401, "Galat username ya password.")
        token = create_session(conn, user["id"])
    resp = JSONResponse({"ok": True})
    set_session_cookie(resp, token)
    return resp


@app.post("/api/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with closing(db()) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE token = ?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )
            conn.commit()
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/", samesite="none" if COOKIE_SECURE else "lax", secure=COOKIE_SECURE)
    return resp


@app.get("/api/me")
def me(request: Request):
    user = require_user(request)
    return {"username": user["username"]}


# --------------------------------------------------------------------------- #
# Upload / list / serve
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
async def upload(request: Request, files: list[UploadFile] = File(...)):
    user = require_user(request)
    saved = []
    with closing(db()) as conn:
        for uf in files:
            orig = safe_name(uf.filename or "file")
            ext = Path(orig).suffix.lower()
            stored = f"{secrets.token_hex(16)}{ext}"
            dest = FILES_DIR / stored
            size = 0
            with open(dest, "wb") as out:
                while True:
                    chunk = await uf.read(CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
            await uf.close()

            mime = uf.content_type or mimetypes.guess_type(orig)[0] or "application/octet-stream"
            kind = classify(orig, mime)
            thumb = None
            if kind == "image":
                thumb_file = f"{Path(stored).stem}.jpg"
                if make_thumbnail(dest, THUMBS_DIR / thumb_file):
                    thumb = thumb_file

            cur = conn.execute(
                """INSERT INTO files (user_id, orig_name, stored_name, size, mime, kind, thumb, uploaded_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (user["id"], orig, stored, size, mime, kind, thumb, now_iso()),
            )
            saved.append(cur.lastrowid)
        conn.commit()
        rows = conn.execute(
            f"SELECT * FROM files WHERE id IN ({','.join('?' * len(saved))})", saved
        ).fetchall() if saved else []
    return {"ok": True, "count": len(saved), "files": [file_row_to_dict(r) for r in rows]}


@app.get("/api/files")
def list_files(request: Request, view: str = "all", kind: str = "all", q: str = ""):
    user = require_user(request)
    where = ["user_id = ?"]
    params: list = [user["id"]]
    if view == "trash":
        where.append("deleted = 1")
    else:
        where.append("deleted = 0")
        if view == "starred":
            where.append("starred = 1")
        if kind in ("image", "video", "file"):
            where.append("kind = ?")
            params.append(kind)
        if q:
            where.append("orig_name LIKE ?")
            params.append(f"%{q}%")
    order = "uploaded_at DESC"
    sql = f"SELECT * FROM files WHERE {' AND '.join(where)} ORDER BY {order}"
    with closing(db()) as conn:
        rows = conn.execute(sql, params).fetchall()
        stats = conn.execute(
            """SELECT COUNT(*) c, COALESCE(SUM(size),0) s FROM files
               WHERE user_id = ? AND deleted = 0""",
            (user["id"],),
        ).fetchone()
    return {
        "files": [file_row_to_dict(r) for r in rows],
        "stats": {"count": stats["c"], "size": stats["s"], "size_h": human_size(stats["s"])},
    }


def _get_owned(conn: sqlite3.Connection, user_id: int, file_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM files WHERE id = ? AND user_id = ?", (file_id, user_id)
    ).fetchone()
    if not row:
        raise HTTPException(404, "File nahi mili.")
    return row


@app.get("/api/file/{file_id}/view")
def view_file(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        row = _get_owned(conn, user["id"], file_id)
    return range_response(FILES_DIR / row["stored_name"], request, row["mime"])


@app.get("/api/file/{file_id}/download")
def download_file(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        row = _get_owned(conn, user["id"], file_id)
    return range_response(
        FILES_DIR / row["stored_name"], request, row["mime"], download_name=row["orig_name"]
    )


@app.get("/api/thumb/{file_id}")
def thumb(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        row = _get_owned(conn, user["id"], file_id)
    if not row["thumb"]:
        raise HTTPException(404, "Thumbnail nahi hai.")
    path = THUMBS_DIR / row["thumb"]
    if not path.exists():
        raise HTTPException(404, "Thumbnail file gayab hai.")
    return FileResponse(path, media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# Mutations: star / trash / restore / delete forever
# --------------------------------------------------------------------------- #
@app.post("/api/file/{file_id}/star")
def star_file(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        row = _get_owned(conn, user["id"], file_id)
        new = 0 if row["starred"] else 1
        conn.execute("UPDATE files SET starred = ? WHERE id = ?", (new, file_id))
        conn.commit()
    return {"ok": True, "starred": bool(new)}


def _soft_delete(conn: sqlite3.Connection, user_id: int, file_id: int) -> None:
    _get_owned(conn, user_id, file_id)
    conn.execute(
        "UPDATE files SET deleted = 1, deleted_at = ? WHERE id = ?", (now_iso(), file_id)
    )
    conn.commit()


@app.post("/api/file/{file_id}/trash")
def trash_file(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        _soft_delete(conn, user["id"], file_id)
    return {"ok": True}


@app.post("/api/trash")
def trash_many(request: Request, payload: dict):
    user = require_user(request)
    ids = payload.get("ids") or []
    with closing(db()) as conn:
        for fid in ids:
            _soft_delete(conn, user["id"], int(fid))
    return {"ok": True, "count": len(ids)}


@app.post("/api/file/{file_id}/restore")
def restore_file(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        _get_owned(conn, user["id"], file_id)
        conn.execute(
            "UPDATE files SET deleted = 0, deleted_at = NULL WHERE id = ?", (file_id,)
        )
        conn.commit()
    return {"ok": True}


@app.post("/api/restore")
def restore_many(request: Request, payload: dict):
    user = require_user(request)
    ids = payload.get("ids") or []
    with closing(db()) as conn:
        for fid in ids:
            _get_owned(conn, user["id"], int(fid))
            conn.execute(
                "UPDATE files SET deleted = 0, deleted_at = NULL WHERE id = ?", (int(fid),)
            )
        conn.commit()
    return {"ok": True, "count": len(ids)}


@app.delete("/api/file/{file_id}")
def delete_forever(file_id: int, request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        row = _get_owned(conn, user["id"], file_id)
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
    (FILES_DIR / row["stored_name"]).unlink(missing_ok=True)
    if row["thumb"]:
        (THUMBS_DIR / row["thumb"]).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/empty-trash")
def empty_trash(request: Request):
    user = require_user(request)
    with closing(db()) as conn:
        rows = conn.execute(
            "SELECT * FROM files WHERE user_id = ? AND deleted = 1", (user["id"],)
        ).fetchall()
        conn.execute("DELETE FROM files WHERE user_id = ? AND deleted = 1", (user["id"],))
        conn.commit()
    for row in rows:
        (FILES_DIR / row["stored_name"]).unlink(missing_ok=True)
        if row["thumb"]:
            (THUMBS_DIR / row["thumb"]).unlink(missing_ok=True)
    return {"ok": True, "count": len(rows)}


# --------------------------------------------------------------------------- #
# Recovery: download everything as a ZIP (works even on a brand new phone)
# --------------------------------------------------------------------------- #
@app.get("/api/recover")
def recover_all(request: Request, kind: str = "all"):
    user = require_user(request)
    with closing(db()) as conn:
        if kind in ("image", "video", "file"):
            rows = conn.execute(
                "SELECT * FROM files WHERE user_id = ? AND deleted = 0 AND kind = ? ORDER BY uploaded_at",
                (user["id"], kind),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM files WHERE user_id = ? AND deleted = 0 ORDER BY uploaded_at",
                (user["id"],),
            ).fetchall()
    if not rows:
        raise HTTPException(404, "Recover karne ke liye koi file nahi hai.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        used: dict[str, int] = {}
        for row in rows:
            path = FILES_DIR / row["stored_name"]
            if not path.exists():
                continue
            arc = row["orig_name"]
            if arc in used:
                used[arc] += 1
                stem, ext = os.path.splitext(arc)
                arc = f"{stem}_{used[arc]}{ext}"
            else:
                used[arc] = 0
            zf.write(path, arcname=arc)
    buf.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    headers = {"Content-Disposition": f'attachment; filename="ApnaVault_recovery_{stamp}.zip"'}
    return StreamingResponse(iter(buf.read, b""), media_type="application/zip", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "12000")))
