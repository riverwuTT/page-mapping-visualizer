# Buffer page-mapping visualizer

A JavaScript port of tt-metal's `buffer.hpp` allocation model, with a single-file
HTML visualizer. Shows, for a list of pages: **which bank/core** it lands in and
**what shard offset**, across interleaved, height / width / block, and ND sharding.

All shapes are expressed **in pages** (the element-shape ÷ page-shape conversion
that `convert_shape_to_pages` does in C++ is assumed already applied).

## Use it

Open **`page_mapping_viz.html`** in a browser — it's fully self-contained (no
network, no server).

## Deploy (GitHub Pages)

Pushing to `main` runs [.github/workflows/pages.yml](.github/workflows/pages.yml),
which builds the single-file HTML, runs the tests, and publishes it as the site's
`index.html` via GitHub Pages.

One-time setup: in the repo's **Settings → Pages**, set **Source = GitHub Actions**
(Pages for a *private* repo requires a GitHub Pro/Team/Enterprise plan; for a Free
plan the repo must be public).

## Files

| File | Role |
|------|------|
| `page_mapping.js`     | The algorithm. Works in node (`require`) and the browser (`window.PageMapping`). |
| `app.js`              | Visualizer UI (depends on `page_mapping.js`). |
| `style.css`           | Styling. |
| `index.html`          | Dev shell that references the three files above. |
| `build.js`            | Inlines css/js into `index.html` → **`page_mapping_viz.html`** (single file). |
| `page_mapping_viz.html` | Built single-file deliverable. |
| `test_page_mapping.js`| Algorithm unit tests (invariants + hand-computed golden cases). |
| `test_ui.js`          | End-to-end smoke test: boots the built HTML in jsdom and drives the UI. |

## Develop / test

```sh
node test_page_mapping.js   # algorithm tests (no deps)
node build.js               # regenerate page_mapping_viz.html after editing js/css
npm install                 # jsdom, for the UI test only
node test_ui.js             # end-to-end UI test against the built HTML
```

## What's modeled (and where it comes from)

- **Interleaved** — `Buffer::page_address`: device page `p` → bank `p % numBanks`,
  slot `⌊p / numBanks⌋`; host page == device page.
- **Height / Width / Block** — `core_to_host_pages` + `generate_buffer_page_mapping`:
  one shard per core, partial-shard clamping, per-core 2D-padded device layout.
- **ND** — `BufferDistributionSpec::compute_page_mapping`: `squeeze_shape_ranks`
  then `iterate_over_shards` / `iterate_within_shard`; shards round-robin across
  cores (`shard_id % num_cores`) and stack within a core
  (`⌊shard_id / num_cores⌋ * shard_volume`).
