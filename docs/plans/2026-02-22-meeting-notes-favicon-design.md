# Meeting Notes Favicon — Design

**Date:** 2026-02-22

## Goal

Add a recognizable favicon to the Meeting Notes app so it shows a meaningful icon when saved as a browser bookmark.

## Icon Concept

Notepad + pen on a dark rounded square background:
- Dark slate background (`#2d3748`) — matches the app's Stop button color
- White document/page rectangle with 3 ruled lines — "notes"
- Red diagonal pen overlapping the lower-right corner (`#e53e3e`) — matches the app's Record button color
- Reads clearly at 16px, 32px favicon sizes

## Implementation

**Data URI approach** — SVG embedded directly in a `<link rel="icon">` tag. No new files, no static directory, single line added to `<head>` in `templates/index.html`.

```html
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,...">
```

## File Changes

| File | Change |
|------|--------|
| `templates/index.html` | Add one `<link rel="icon">` line to `<head>` |

No other files touched.
