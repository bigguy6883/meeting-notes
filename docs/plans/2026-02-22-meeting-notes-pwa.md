# Meeting Notes PWA Icons Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the Meeting Notes favicon from an inline SVG data URI to a full PWA setup with a static icon file and manifest.

**Architecture:** Create `static/icon.svg` (extracted from the existing data URI) and `static/manifest.json`, then update `templates/index.html` to reference them via Flask's `url_for('static', ...)`. Flask serves the `static/` directory automatically — no `app.py` changes needed.

**Tech Stack:** SVG, HTML, Flask/Jinja2

---

### Task 1: Create `static/icon.svg`

**Files:**
- Create: `static/icon.svg`

No tests for static assets. Visual verification only.

**Step 1: Create the file**

Create `/home/pi/meeting-notes/static/icon.svg` with this exact content (decoded from the existing data URI — `%23` → `#`, single quotes → double quotes):

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="18" fill="#2d3748"/>
  <rect x="15" y="12" width="50" height="62" rx="5" fill="white"/>
  <rect x="23" y="28" width="34" height="3.5" rx="1.75" fill="#e2e8f0"/>
  <rect x="23" y="38" width="34" height="3.5" rx="1.75" fill="#e2e8f0"/>
  <rect x="23" y="48" width="26" height="3.5" rx="1.75" fill="#e2e8f0"/>
  <g transform="translate(65,65) rotate(-40)">
    <rect x="-6" y="-30" width="12" height="48" rx="3" fill="#e53e3e"/>
    <polygon points="-6,18 6,18 0,28" fill="#c53030"/>
    <rect x="-6" y="-34" width="12" height="8" rx="3" fill="#fc8181"/>
  </g>
</svg>
```

**Step 2: Commit**

```bash
git add static/icon.svg
git commit -m "feat: extract favicon SVG to static/icon.svg"
```

---

### Task 2: Create `static/manifest.json`

**Files:**
- Create: `static/manifest.json`

**Step 1: Create the file**

Create `/home/pi/meeting-notes/static/manifest.json`:

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

**Step 2: Commit**

```bash
git add static/manifest.json
git commit -m "feat: add PWA manifest for meeting notes"
```

---

### Task 3: Update `templates/index.html`

**Files:**
- Modify: `templates/index.html:7` (replace the existing data URI link tag)

**Step 1: Replace the data URI link tag**

Find and remove this line (line 7):

```html
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg ...>">
```

Replace it with these three lines:

```html
  <link rel="icon" type="image/svg+xml" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="apple-touch-icon" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="manifest" href="{{ url_for('static', filename='manifest.json') }}">
```

The resulting `<head>` opening should look like:

```html
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Meeting Notes</title>
  <link rel="icon" type="image/svg+xml" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="apple-touch-icon" href="{{ url_for('static', filename='icon.svg') }}">
  <link rel="manifest" href="{{ url_for('static', filename='manifest.json') }}">
  <style>
```

**Step 2: Commit**

```bash
git add templates/index.html
git commit -m "feat: upgrade to PWA favicon with static files and manifest"
```

---

### Task 4: Push to GitHub

```bash
git push origin main
```
