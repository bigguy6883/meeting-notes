# SQLite Job Persistence — Design Doc
Date: 2026-02-23

## Problem

Jobs are lost whenever the app restarts because `JobManager` stores state in an
in-memory dict. If the app restarts while a job is mid-processing (transcribing,
summarizing, emailing), the job vanishes silently with no error visible to the user.
Thread exceptions are not surfaced — failed jobs just disappear.

## Solution

Replace the in-memory dict in `JobManager` with a SQLite database. Every status
change is written to disk immediately. On startup, any interrupted jobs are
detected and marked as errors.

## Schema

Single `jobs` table:

```sql
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    label            TEXT NOT NULL,
    status           TEXT NOT NULL,
    summary          TEXT,
    transcript_path  TEXT,
    audio_path       TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL
)
```

## DB Location

`~/meeting-notes/jobs.db` — hardcoded path, no config needed.

## Startup Recovery

On `JobManager.__init__`, query for any jobs with status in
`('pending', 'transcribing', 'summarizing', 'emailing')` and update them to
`status='error'`, `error='interrupted by restart'`. These will appear in the UI
as retryable failed jobs.

## Status Writes

Every `_set_status` call executes an immediate `UPDATE` to SQLite. Thread
exceptions are caught and written as `status='error'` with the full exception
message. No job can silently vanish.

## Scope of Changes

| File | Change |
|------|--------|
| `jobs.py` | Rewrite `JobManager` to use `sqlite3` — schema init, CRUD, status updates |
| `app.py` | Pass `db_path` when constructing `JobManager` |
| `tests/conftest.py` | Provide `JobManager` using `:memory:` SQLite DB |
| `tests/test_jobs.py` | Update fixtures to match new constructor signature |

No changes to `recorder.py`, `transcriber.py`, `summarizer.py`, `emailer.py`,
templates, or routes.

## Out of Scope

- Full meeting history/archive (not needed)
- Job deletion from UI
- Multiple email recipients
- Speaker diarization
