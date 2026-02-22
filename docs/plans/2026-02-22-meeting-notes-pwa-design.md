# Meeting Notes PWA Icons — Design

**Date:** 2026-02-22

## Goal

Upgrade the Meeting Notes favicon from an inline SVG data URI to a full PWA setup with a real icon file and manifest, matching the pattern used by InkFrame and Baby Tracker.

## Changes

| File | Action |
|------|--------|
| `static/icon.svg` | Create — extract SVG from existing data URI |
| `static/manifest.json` | Create — PWA manifest with name, theme color, icon |
| `templates/index.html` | Modify — replace single data URI link with 3 tags |

## static/icon.svg

Same SVG as the current data URI (decoded), with `%23` → `#` and single quotes → double quotes:
- Dark slate background (`#2d3748`, rx=18)
- White notepad page
- Three gray ruled lines (`#e2e8f0`)
- Red diagonal pen group (`#e53e3e` body, `#c53030` tip, `#fc8181` cap)

## static/manifest.json

```json
{
  "name": "Meeting Notes",
  "short_name": "MeetingNotes",
  "description": "Automatic meeting recorder, transcriber, and summarizer",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#f5f5f5",
  "theme_color": "#2d3748",
  "icons": [
    {
      "src": "/static/icon.svg",
      "sizes": "any",
      "type": "image/svg+xml",
      "purpose": "any maskable"
    }
  ]
}
```

## templates/index.html head changes

Replace the single data URI `<link rel="icon">` with:

```html
  <link rel="icon" type="image/svg+xml" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="apple-touch-icon" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="manifest" href="{{ url_for('static', filename='manifest.json') }}">
```

Flask serves `static/` automatically — no app.py changes needed.
