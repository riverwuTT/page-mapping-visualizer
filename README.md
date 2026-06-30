# tt-metal sharding visualizers

Two JavaScript ports of tt-metal's data-layout model, with single-file HTML
visualizers and an overview landing page:

- **Tensor sharding** (`tensor.html`) — the full **element → page → core** story.
  A logical N-D tensor shape, a **layout** (row-major or 32×32 tile), and a
  **sharding** scheme (interleave / classic height / width / block / ND) decide
  which *element* lands on which core. Ported from
  [`tensor_spec.hpp`](../tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp)
  (+ `tensor_layout.cpp`, `page_config.cpp`).
- **Buffer page-mapping** (`buffer.html`) — the lower **page → core** layer:
  given a list of *pages*, which bank/core does each land in and at what shard
  offset? Ported from [`buffer.hpp`](../tt_metal/api/tt-metalium/buffer.hpp).

The tensor model sits on top of the buffer model: it adds the element-shape ÷
layout step (`compute_physical_shape` → `get_page_shape`) the buffer model
assumes is already done, then hands the resulting page grid + shard shape to the
**same** page → core logic (`page_mapping.js`).

## Use it

Open **`index.html`** for the overview, or open either single-file deliverable
directly — both are fully self-contained (no network, no server):

- **`tensor_mapping_viz.html`** — the tensor visualizer.
- **`page_mapping_viz.html`** — the buffer visualizer.

## Deploy (GitHub Pages)

Pushing to `main` runs [.github/workflows/pages.yml](.github/workflows/pages.yml),
which builds both single-file HTMLs, runs the tests, and publishes the site:
`index.html` (overview), `buffer.html`, `tensor.html`.

One-time setup: in the repo's **Settings → Pages**, set **Source = GitHub Actions**
(Pages for a *private* repo requires a GitHub Pro/Team/Enterprise plan; for a Free
plan the repo must be public).

## Files

| File | Role |
|------|------|
| `tensor_mapping.js`     | Tensor algorithm: element → page → core. Delegates page → core to `page_mapping.js`. |
| `tensor_app.js`         | Tensor visualizer UI. |
| `tensor.html`           | Tensor dev shell (references external files). |
| `page_mapping.js`       | Buffer algorithm: page → shard → bank. node (`require`) + browser (`window.PageMapping`). |
| `app.js`                | Buffer visualizer UI. |
| `buffer.html`           | Buffer dev shell. |
| `index.html`            | Overview landing page (self-contained). |
| `style.css`             | Shared styling. |
| `build.js`              | Inlines css/js into the dev shells → `tensor_mapping_viz.html` + `page_mapping_viz.html`. |
| `test_page_mapping.js`  | Buffer algorithm unit tests. |
| `test_tensor_mapping.js`| Tensor algorithm unit tests (hand-computed golden cases + invariants). |
| `test_ui.js`            | Buffer end-to-end UI test (jsdom). |
| `test_tensor_ui.js`     | Tensor end-to-end UI test (jsdom). |

## Develop / test

```sh
node test_tensor_mapping.js   # tensor algorithm tests (no deps)
node test_page_mapping.js     # buffer algorithm tests (no deps)
node build.js                 # regenerate both *_viz.html after editing js/css
npm install                   # jsdom, for the UI tests only
node test_ui.js               # buffer UI test against the built HTML
node test_tensor_ui.js        # tensor UI test against the built HTML
bash dev.sh all               # build + every test (puts nvm node on PATH)
```

## What's modeled (and where it comes from)

### Tensor layer (`tensor_mapping.js`)
- **Physical shape** — `TensorLayout::compute_physical_shape`: fold the N-D
  logical shape to 2D `[H, W]` and align each dim (tile → `[32, 32]`; row-major →
  width to the shard width, else `[1]`).
- **Page shape** — `PageConfig::get_page_shape`: a 32×32 tile for TILE layout; one
  row (full width interleaved, or a shard-wide slice when sharded) for ROW_MAJOR.
- **Element → page** — tile the physical 2D shape by the page shape; page ids are
  row-major over the page grid (the same ids the buffer model uses).
- **Sharding** — `height_sharded` / `width_sharded` / `block_sharded` / `sharded`
  compute the N-D shard shape (with REQUIRED/RECOMMENDED shard alignment), which
  `compute_buffer_sharding_args` converts to a page grid + shard shape and a
  distribution strategy. For sharded tensors the N-D distribution spec is
  authoritative (it is what `generate_buffer_page_mapping` uses).

### Buffer layer (`page_mapping.js`)
- **Interleaved** — `Buffer::page_address`: page `p` → bank `p % numBanks`.
- **Height / Width / Block** — `core_to_host_pages` + `generate_buffer_page_mapping`.
- **ND** — `BufferDistributionSpec::compute_page_mapping`: round-robin
  (`shard_id % num_cores`) or 2D grid distribution.
