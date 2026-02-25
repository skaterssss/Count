# AGENTS.md

## Cursor Cloud specific instructions

This is a vanilla HTML/CSS/JS static web app (a children's counting game). There are **no dependencies**, no build step, no package manager, and no tests.

### Running the app

Serve the project root with any static file server:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in a browser. The entry point is `index.html`.

### Notes

- Animal images are loaded from Unsplash at runtime. If the network is unavailable, the game gracefully falls back to emoji.
- There is no linter, test suite, or build system configured for this project.
