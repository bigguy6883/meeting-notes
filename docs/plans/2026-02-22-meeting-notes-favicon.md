# Meeting Notes Favicon Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a notepad+pen favicon to the Meeting Notes app via a single SVG data URI link tag.

**Architecture:** Embed the SVG directly as a data URI in `<link rel="icon">` inside `<head>` of `templates/index.html`. No new files, no static directory, no Flask changes.

**Tech Stack:** SVG, HTML

---

### Task 1: Add favicon link tag to `templates/index.html`

**Files:**
- Modify: `templates/index.html:6` (after the `<title>` line)

No unit tests for a favicon link tag. Visual verification only.

**Step 1: Insert the link tag**

In `templates/index.html`, insert this line after line 6 (the `<title>` line):

```html
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='18' fill='%232d3748'/><rect x='15' y='12' width='50' height='62' rx='5' fill='white'/><rect x='23' y='28' width='34' height='3.5' rx='1.75' fill='%23e2e8f0'/><rect x='23' y='38' width='34' height='3.5' rx='1.75' fill='%23e2e8f0'/><rect x='23' y='48' width='26' height='3.5' rx='1.75' fill='%23e2e8f0'/><g transform='translate(65,65) rotate(-40)'><rect x='-6' y='-30' width='12' height='48' rx='3' fill='%23e53e3e'/><polygon points='-6,18 6,18 0,28' fill='%23c53030'/><rect x='-6' y='-34' width='12' height='8' rx='3' fill='%23fc8181'/></g></svg>">
```

The SVG encodes:
- `%232d3748` = `#2d3748` (dark slate background, matches Stop button)
- `%23e2e8f0` = `#e2e8f0` (light gray ruled lines)
- `%23e53e3e` = `#e53e3e` (red pen body, matches Record button)
- `%23c53030` = `#c53030` (darker red pen tip)
- `%23fc8181` = `#fc8181` (lighter red pen eraser/cap)

The resulting `<head>` opening should look like:

```html
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Meeting Notes</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,...">
  <style>
```

**Step 2: Verify visually**

Run the app locally (or check in a browser) and confirm the browser tab shows the dark rounded icon with a white page and red pen.

To run the app: `cd /home/pi/meeting-notes && python app.py` then open in browser.

**Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add notepad+pen favicon via SVG data URI"
```
