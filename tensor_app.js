// SPDX-License-Identifier: Apache-2.0
// Visualizer UI for the TENSOR sharding model — the element → page → core
// mapping. Depends on the globals `TensorMapping` (tensor_mapping.js) and
// `PageMapping` (page_mapping.js).
//
//   element grid (physical 2D)  →  pages  →  cores
//   ^ the tensor-specific layer     ^ reuses the buffer page-mapping model
//
// Inputs are in ELEMENTS (a logical N-D shape); the layout (row-major / tile)
// and sharding pick how those elements fold into pages and scatter across cores.

(function () {
    "use strict";
    const TM = window.TensorMapping;

    // [name, cfg] — each preset is a full tensor configuration.
    const PRESETS = [
        ["Tile · interleave", { logicalShape: "64,96", layout: "TILE", sharding: "interleave", bankX: 4, bankY: 1 }],
        ["Tile · height", { logicalShape: "128,64", layout: "TILE", sharding: "height", gridX: 4, gridY: 1 }],
        ["Tile · width", { logicalShape: "64,128", layout: "TILE", sharding: "width", gridX: 4, gridY: 1 }],
        ["Tile · block", { logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 }],
        ["Tile · ND rank-3", { logicalShape: "2,64,64", layout: "TILE", sharding: "nd", ndShardShape: "1,32,32", gridX: 2, gridY: 2, ndStrategy: "round_robin" }],
        ["RM · interleave", { logicalShape: "6,8", layout: "ROW_MAJOR", sharding: "interleave", bankX: 4, bankY: 1 }],
        ["RM · height", { logicalShape: "8,8", layout: "ROW_MAJOR", sharding: "height", gridX: 4, gridY: 1 }],
        ["RM · block", { logicalShape: "8,8", layout: "ROW_MAJOR", sharding: "block", gridX: 2, gridY: 2 }],
    ];

    const divUp = (a, b) => Math.floor((a + b - 1) / b);
    const el = (id) => document.getElementById(id);
    const dom = {
        logicalShape: el("logicalShape"),
        layout: el("layout"),
        tile: el("tile"),
        tileField: el("tileField"),
        dtype: el("dtype"),
        sharding: el("sharding"),
        grid: el("gridField"),
        gridX: el("gridX"),
        gridY: el("gridY"),
        orientation: el("orientation"),
        ndField: el("ndField"),
        ndShardShape: el("ndShardShape"),
        ndStrategy: el("ndStrategy"),
        ndAlignment: el("ndAlignment"),
        bankField: el("bankField"),
        bankX: el("bankX"),
        bankY: el("bankY"),
        colorMode: el("colorMode"),
        granularity: el("granularity"),
        granHeading: el("granHeading"),
        presets: el("presets"),
        error: el("error"),
        warn: el("warn"),
        summary: el("summary"),
        elementView: el("elementView"),
        elementLegend: el("elementLegend"),
        pageLegend: el("pageLegend"),
        pageStrip: el("pageStrip"),
        shardLegend: el("shardLegend"),
        shardsView: el("shardsView"),
        banksView: el("banksView"),
        results: el("results"),
        selbar: el("selbar"),
    };

    // ---- linked selection (click an element / page / shard / bank) ----
    const cellsByPage = new Map();
    const groupToggles = [];
    let selection = new Set();
    let selectedCells = [];

    function linkCell(cell, pageId) {
        let arr = cellsByPage.get(pageId);
        if (!arr) cellsByPage.set(pageId, (arr = []));
        arr.push(cell);
    }
    function registerCell(cell, pageId) {
        cell.dataset.page = pageId;
        linkCell(cell, pageId);
    }
    function registerToggle(elm, pages) {
        elm.dataset.toggle = "1";
        elm._pages = pages;
        groupToggles.push({ el: elm, pages });
    }
    function applySelection() {
        selectedCells.forEach((c) => c.classList.remove("sel"));
        selectedCells = [];
        dom.results.classList.toggle("selecting", selection.size > 0);
        selection.forEach((pid) => {
            const arr = cellsByPage.get(pid);
            if (arr) arr.forEach((c) => (c.classList.add("sel"), selectedCells.push(c)));
        });
        for (const { el: e, pages } of groupToggles) {
            e.classList.toggle("active", pages.length > 0 && pages.every((p) => selection.has(p)));
        }
        updateSelbar();
    }
    function togglePage(p) {
        selection.has(p) ? selection.delete(p) : selection.add(p);
        applySelection();
    }
    function togglePages(pages) {
        if (!pages.length) return;
        const allOn = pages.every((p) => selection.has(p));
        pages.forEach((p) => (allOn ? selection.delete(p) : selection.add(p)));
        applySelection();
    }
    function clearSelection() {
        selection.clear();
        applySelection();
    }
    function updateSelbar() {
        if (selection.size === 0) {
            dom.selbar.classList.remove("on");
            dom.selbar.innerHTML = "";
            return;
        }
        dom.selbar.classList.add("on");
        dom.selbar.innerHTML = `<span>${selection.size} page${selection.size === 1 ? "" : "s"} selected</span>`;
        const btn = document.createElement("button");
        btn.textContent = "Clear";
        btn.onclick = clearSelection;
        dom.selbar.appendChild(btn);
    }

    const PALETTE = [
        "#4ea1ff", "#ff7b72", "#7ee787", "#ffa657", "#d2a8ff", "#79c0ff",
        "#f0883e", "#56d364", "#ff9bce", "#e3b341", "#a5d6ff", "#ffab70",
    ];
    const colorFor = (i) => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];

    // What a cell's color encodes: its destination core (pages on the same core
    // share a color), its page (each page distinct, so the tiling stands out), or
    // its shard (pages in the same shard share a color). `lk` is a pageLookup entry.
    const colorMode = () => dom.colorMode.value;
    function cellColorFor(lk, pageId) {
        const m = colorMode();
        if (m === "page") return colorFor(pageId);
        if (m === "shard") return colorFor(lk.shardId);
        return colorFor(lk.bankId);
    }

    // The "Show" granularity (what one headline cell represents) and the "mapped
    // to" destination (what its color/grouping encodes) form the transformation
    // shown in the headline view, e.g. "Elements → Cores".
    const GRAN_WORD = { element: "Elements", tile: "Tiles", page: "Pages" };
    const DEST_WORD = { core: "Cores", shard: "Shards", page: "Pages" };
    const granularity = () => dom.granularity.value;
    // The destination bucket a page falls into under the active color mode — same
    // grouping cellColorFor uses. Two units with the same key share a color.
    function destKeyOf(lk, pageId) {
        const m = colorMode();
        if (m === "page") return pageId;
        if (m === "shard") return lk.shardId;
        return lk.bankId;
    }

    // Legend for the element / page-grid views, matching the active color mode.
    function colorLegend(res) {
        const m = colorMode();
        if (m === "core") return coreLegend(res, "color = destination core");
        const word = m === "shard" ? "shard" : "page";
        const count = m === "shard" ? res.mapping.numShards : res.mapping.numPages;
        const n = Math.min(count, 12);
        let leg = "";
        for (let i = 0; i < n; i++) {
            leg += `<span class="sw"><span class="box" style="background:${colorFor(i)}"></span>${word} ${i}</span>`;
        }
        if (count > 12) leg += `<span class="sw">… (color = ${word} mod 12)</span>`;
        return `<span style="color:var(--muted)">color = ${word}:</span>${leg}`;
    }

    function div(cls) {
        const d = document.createElement("div");
        if (cls) d.className = cls;
        return d;
    }
    const parseShape = (s) => {
        const parts = String(s).split(",").map((x) => x.trim()).filter((x) => x.length);
        if (!parts.length) throw new Error("empty shape");
        return parts.map((x) => {
            const n = parseInt(x, 10);
            if (!Number.isFinite(n) || n <= 0) throw new Error(`invalid shape entry: "${x}"`);
            return n;
        });
    };
    const intOf = (inp) => {
        const n = parseInt(inp.value, 10);
        if (!Number.isFinite(n) || n <= 0) throw new Error(`invalid number: "${inp.value}"`);
        return n;
    };

    // ---- shareable link: mirror the whole config in the URL hash ----
    // Format: #shape=2x2,layout=TILE,sharding=block,grid=2x2,...  — key=value
    // pairs joined by commas, so shape-valued fields use 'x' (not ',') internally.
    const shapeEnc = (s) => s.split(",").map((x) => x.trim()).filter((x) => x.length).join("x");
    const shapeDec = (s) => s.split("x").map((x) => x.trim()).filter((x) => x.length).join(",");

    // Only the fields that matter for the current mode are emitted, keeping links
    // short; anything omitted falls back to the control's current value on load.
    function encodeHash() {
        const p = [];
        p.push("shape=" + shapeEnc(dom.logicalShape.value));
        p.push("layout=" + dom.layout.value);
        if (dom.layout.value === "TILE") p.push("tile=" + shapeEnc(dom.tile.value));
        p.push("dtype=" + dom.dtype.value);
        p.push("sharding=" + dom.sharding.value);
        if (dom.sharding.value === "interleave") p.push("banks=" + dom.bankX.value + "x" + dom.bankY.value);
        else p.push("grid=" + dom.gridX.value + "x" + dom.gridY.value);
        if (dom.sharding.value === "nd") {
            p.push("ndshape=" + shapeEnc(dom.ndShardShape.value));
            p.push("ndstrat=" + dom.ndStrategy.value);
            p.push("ndalign=" + dom.ndAlignment.value);
        }
        p.push("orient=" + dom.orientation.value);
        p.push("show=" + dom.granularity.value);
        p.push("color=" + dom.colorMode.value);
        return p.join(",");
    }

    let lastHash = null;
    function updateHash() {
        const h = encodeHash();
        if (h === lastHash) return;
        lastHash = h;
        if (window.history && history.replaceState) {
            try {
                history.replaceState(null, "", "#" + h);
            } catch (e) {
                /* jsdom / sandboxed frames may reject; the UI still works */
            }
        }
    }

    // Apply an incoming hash onto the controls. Missing keys leave defaults intact;
    // unknown option values are ignored by the <select> (stays on its default).
    function applyHash() {
        const raw = (location.hash || "").replace(/^#/, "");
        if (!raw) return;
        const obj = {};
        for (const kv of raw.split(",")) {
            const i = kv.indexOf("=");
            if (i > 0) obj[kv.slice(0, i)] = kv.slice(i + 1);
        }
        const setV = (inp, v) => {
            if (v != null && v !== "") inp.value = v;
        };
        setV(dom.logicalShape, obj.shape && shapeDec(obj.shape));
        setV(dom.layout, obj.layout);
        setV(dom.tile, obj.tile && shapeDec(obj.tile));
        setV(dom.dtype, obj.dtype);
        setV(dom.sharding, obj.sharding);
        if (obj.grid) {
            const [x, y] = obj.grid.split("x");
            setV(dom.gridX, x);
            setV(dom.gridY, y);
        }
        if (obj.banks) {
            const [x, y] = obj.banks.split("x");
            setV(dom.bankX, x);
            setV(dom.bankY, y);
        }
        setV(dom.ndShardShape, obj.ndshape && shapeDec(obj.ndshape));
        setV(dom.ndStrategy, obj.ndstrat);
        setV(dom.ndAlignment, obj.ndalign);
        setV(dom.orientation, obj.orient);
        setV(dom.granularity, obj.show);
        setV(dom.colorMode, obj.color);
    }

    function buildPresets() {
        dom.presets.innerHTML = "";
        for (const [name, vals] of PRESETS) {
            const b = document.createElement("button");
            b.textContent = name;
            b.onclick = () => {
                dom.logicalShape.value = vals.logicalShape;
                dom.layout.value = vals.layout;
                dom.sharding.value = vals.sharding;
                if (vals.gridX) dom.gridX.value = vals.gridX;
                if (vals.gridY) dom.gridY.value = vals.gridY;
                if (vals.bankX) dom.bankX.value = vals.bankX;
                if (vals.bankY) dom.bankY.value = vals.bankY;
                if (vals.ndShardShape) dom.ndShardShape.value = vals.ndShardShape;
                if (vals.ndStrategy) dom.ndStrategy.value = vals.ndStrategy;
                render();
            };
            dom.presets.appendChild(b);
        }
    }

    function syncFields() {
        const isTile = dom.layout.value === "TILE";
        const sharding = dom.sharding.value;
        const isInterleave = sharding === "interleave";
        const isND = sharding === "nd";
        dom.tileField.style.display = isTile ? "" : "none";
        dom.grid.style.display = isInterleave ? "none" : "";
        dom.ndField.style.display = isND ? "" : "none";
        dom.bankField.style.display = isInterleave ? "" : "none";
    }

    function render() {
        dom.error.textContent = "";
        dom.warn.textContent = "";
        cellsByPage.clear();
        groupToggles.length = 0;
        selection.clear();
        selectedCells = [];
        syncFields();
        updateHash();

        let res;
        try {
            const cfg = {
                logicalShape: parseShape(dom.logicalShape.value),
                layout: dom.layout.value,
                tile: dom.layout.value === "TILE" ? parseShape(dom.tile.value) : undefined,
                dtype: dom.dtype.value,
                sharding: dom.sharding.value,
                orientation: dom.orientation.value,
            };
            if (cfg.sharding === "interleave") {
                cfg.bankGrid = { x: intOf(dom.bankX), y: intOf(dom.bankY) };
            } else {
                cfg.grid = { x: intOf(dom.gridX), y: intOf(dom.gridY) };
            }
            if (cfg.sharding === "nd") {
                cfg.ndShardShape = parseShape(dom.ndShardShape.value);
                cfg.ndStrategy = dom.ndStrategy.value;
                cfg.ndAlignment = dom.ndAlignment.value;
            }
            res = TM.computeTensorMapping(cfg);
        } catch (e) {
            dom.error.textContent = e.message;
            ["summary", "elementView", "elementLegend", "pageStrip", "pageLegend", "shardsView", "shardLegend", "banksView"].forEach(
                (k) => (dom[k].innerHTML = "")
            );
            return;
        }

        renderSummary(res);
        renderGranularityView(res);
        renderComposition(res);
        renderShards(res);
        renderBanks(res);
        applySelection();
    }

    function renderSummary(res) {
        const m = res.mapping;
        const rows = [
            ["Logical shape", res.logicalShape.join(" × "), true],
            ["Layout", res.layout === "TILE" ? `tile ${res.tile.join("×")}` : "row-major"],
            ["Data type", res.dtype],
            ["Memory layout", res.memoryLayout.toLowerCase().replace("_", " ")],
            ["Alignment", `[${res.alignment.join(", ")}]`],
            ["Padded shape", res.paddedShape.join(" × ")],
            ["Physical 2D (elements)", `${res.physical[0]} × ${res.physical[1]}`, true],
            ["Page shape (elements)", `${res.pageShape[0]} × ${res.pageShape[1]}`],
            ["Tensor in pages", `${res.tensor2dInPages[0]} × ${res.tensor2dInPages[1]}  (= ${m.numPages})`, true],
        ];
        if (res.distribution !== "interleaved") {
            rows.push(["Shard (pages)", res.shardShapeInPages ? res.shardShapeInPages.join(" × ") : "—"]);
            rows.push(["Distribution", res.distribution === "grid_2d" ? "grid (2D)" : "round-robin (1D)"]);
            rows.push(["Shards", m.numShards]);
        }
        rows.push(["Cores (banks)", m.numBanks]);
        rows.push(["Max slots / core", m.maxSlotsPerCore]);
        dom.summary.innerHTML =
            '<table class="stats-table"><tbody>' +
            rows
                .map(([k, v, hot]) => `<tr class="${hot ? "hot" : ""}"><td class="k">${k}</td><td class="v">${v}</td></tr>`)
                .join("") +
            "</tbody></table>";
    }

    // The headline "transformation" view: render one cell per <granularity> unit
    // (element / tile / page), colored/grouped by the <mapped-to> destination.
    // The heading reads e.g. "Elements → Cores".
    function renderGranularityView(res) {
        const g = granularity();
        dom.granHeading.textContent = `${GRAN_WORD[g]} → ${DEST_WORD[colorMode()]}`;
        dom.elementView.innerHTML = "";
        dom.elementLegend.innerHTML = "";
        if (g === "page") return renderPageView(res);
        if (g === "tile") return renderTileView(res);
        return renderElementGrid(res);
    }

    // Page granularity: one cell per page, laid out in the page grid, labeled with
    // the element coordinate range it covers. Reuses the element grid's page-cell
    // renderer (also the oversized-tensor fallback), colored by the active mode.
    function renderPageView(res) {
        dom.elementLegend.innerHTML = colorLegend(res);
        renderPageCoordGrid(res);
    }

    // Tile granularity: one cell per tile-shaped block (tile[h]×tile[w] elements),
    // regardless of layout. In TILE layout a tile coincides with a page; in
    // ROW_MAJOR it is a coarse block that spans many 1×W pages — possibly across
    // several destinations, which is flagged with a corner marker (◩).
    function renderTileView(res) {
        const e = res.element;
        const [tileH, tileW] = res.tile;
        const H = e.H, W = e.W;
        const tilesH = divUp(H, tileH);
        const tilesW = divUp(W, tileW);
        const total = tilesH * tilesW;
        dom.elementLegend.innerHTML = colorLegend(res);

        const TMAX = 4000;
        if (total > TMAX) {
            const note = div("draghint");
            note.textContent =
                `Tile grid is ${tilesH} × ${tilesW} = ${total} tiles — too many to draw. ` +
                `Switch "Show" to pages, or use a smaller shape.`;
            dom.elementView.appendChild(note);
            return;
        }

        const colPx = total > 600 ? 44 : total > 150 ? 62 : 84;
        const grid = div("cells");
        grid.style.gridTemplateColumns = `repeat(${tilesW}, ${colPx}px)`;
        let hasMixed = false;
        for (let tr = 0; tr < tilesH; tr++) {
            for (let tc = 0; tc < tilesW; tc++) {
                const r0 = tr * tileH, c0 = tc * tileW;
                const r1 = Math.min(r0 + tileH, H) - 1;
                const c1 = Math.min(c0 + tileW, W) - 1;
                // pages the tile block overlaps
                const pr0 = Math.floor(r0 / e.ph), pr1 = Math.floor(r1 / e.ph);
                const pc0 = Math.floor(c0 / e.pw), pc1 = Math.floor(c1 / e.pw);
                const present = [];
                for (let pr = pr0; pr <= pr1; pr++) {
                    for (let pc = pc0; pc <= pc1; pc++) {
                        const p = pr * e.pagesW + pc;
                        const lk = res.mapping.pageLookup[p];
                        if (lk) present.push({ p, lk });
                    }
                }
                const cell = div("pcell tcell");
                cell.style.width = colPx + "px";
                if (!present.length) {
                    cell.classList.add("pad");
                    cell.innerHTML =
                        `<span class="from">tile (${tr}, ${tc})</span><span class="to">padding</span>`;
                    cell.title = `tile (${tr}, ${tc})  ·  elements (${r0},${c0}) → (${r1},${c1}) — padding`;
                    grid.appendChild(cell);
                    continue;
                }
                const rep = present[0];
                cell.style.background = cellColorFor(rep.lk, rep.p);
                const keys = new Set(present.map((x) => destKeyOf(x.lk, x.p)));
                const mixed = keys.size > 1;
                if (mixed) { cell.classList.add("mixed"); hasMixed = true; }
                cell.innerHTML =
                    `<span class="from">(${r0}, ${c0})</span>` +
                    `<span class="to">→ (${r1}, ${c1})</span>`;
                const pageIds = present.map((x) => x.p);
                const pageSpan = pageIds.length === 1 ? `page ${pageIds[0]}` :
                    `pages ${pageIds[0]}–${pageIds[pageIds.length - 1]} (${pageIds.length})`;
                const bc = res.mapping.banks[rep.lk.bankId].gridCoord;
                cell.title =
                    `tile (${tr}, ${tc})  ·  elements (${r0},${c0}) → (${r1},${c1})\n` +
                    `${pageSpan}\n` +
                    (mixed
                        ? `spans multiple ${DEST_WORD[colorMode()].toLowerCase()}`
                        : `→ core ${rep.lk.bankId} (${bc.x},${bc.y})  ·  shard ${rep.lk.shardId}`);
                pageIds.forEach((p) => linkCell(cell, p));
                if (pageIds.length === 1) cell.dataset.page = pageIds[0];
                else registerToggle(cell, pageIds);
                grid.appendChild(cell);
            }
        }
        const wrap = div("pagegrid");
        wrap.appendChild(grid);
        dom.elementView.appendChild(wrap);
        if (hasMixed) {
            dom.elementLegend.innerHTML +=
                `<span class="sw" style="color:var(--muted)">◩ = tile spans multiple ${DEST_WORD[colorMode()].toLowerCase()}</span>`;
        }
    }

    // The physical 2D element grid. Each element is one cell, colored by its page's
    // destination (mapped-to mode); page boundaries are drawn as thick gridlines;
    // elements outside the logical shape are hatched padding.
    function renderElementGrid(res) {
        dom.elementView.innerHTML = "";
        const e = res.element;
        const total = e.H * e.W;
        const MAX = 8192;
        if (total > MAX) {
            dom.warn.textContent =
                `Element grid has ${total} elements (> ${MAX}) — too many to draw one cell per element. ` +
                `Showing one cell per page instead, laid out in the element grid's shape, each labeled with the ` +
                `element coordinate range [from] → [to] it covers.`;
            renderPageCoordGrid(res);
            dom.elementLegend.innerHTML = colorLegend(res);
            return;
        }
        const cellPx = total > 2500 ? 11 : total > 900 ? 15 : 22;
        const grid = div("elemgrid");
        grid.style.gridTemplateColumns = `repeat(${e.W}, ${cellPx}px)`;
        grid.style.setProperty("--ph", e.ph);
        grid.style.setProperty("--pw", e.pw);
        for (let r = 0; r < e.H; r++) {
            for (let c = 0; c < e.W; c++) {
                const pageId = e.pageOf(r, c);
                const lk = res.mapping.pageLookup[pageId];
                const cell = div("ecell");
                cell.style.width = cell.style.height = cellPx + "px";
                // thick lines on page boundaries
                if (r % e.ph === 0) cell.classList.add("top");
                if (c % e.pw === 0) cell.classList.add("left");
                if (r === e.H - 1) cell.classList.add("bot");
                if (c === e.W - 1) cell.classList.add("right");
                if (e.isPadding(r, c)) {
                    cell.classList.add("pad");
                    cell.title = `element (${r}, ${c}) — padding\npage ${pageId}`;
                } else if (lk) {
                    cell.style.background = cellColorFor(lk, pageId);
                    const bc = res.mapping.banks[lk.bankId].gridCoord;
                    cell.title =
                        `element (${r}, ${c})\n→ page ${pageId}\n→ core ${lk.bankId} (${bc.x},${bc.y})`;
                    registerCell(cell, pageId);
                }
                grid.appendChild(cell);
            }
        }
        dom.elementView.appendChild(grid);

        dom.elementLegend.innerHTML = colorLegend(res);
    }

    // Fallback for large tensors: one cell per PAGE, arranged as the page grid
    // (which has the same shape/aspect as the element grid), each cell labeled
    // with the element coordinate range [r0,c0] → [r1,c1] the page covers and
    // colored by destination core. Page ids match every other view.
    function renderPageCoordGrid(res) {
        const e = res.element;
        const { pagesH, pagesW, ph, pw } = e;
        const totalPages = pagesH * pagesW;
        const PMAX = 6000;
        if (totalPages > PMAX) {
            const note = div("draghint");
            note.textContent =
                `Page grid is ${pagesH} × ${pagesW} = ${totalPages} pages — too many to annotate. ` +
                `See "page grid → core" below for the full mapping.`;
            dom.elementView.appendChild(note);
            return;
        }
        // wider cells when each page spans a range in both dims (e.g. tiles)
        const colPx = ph > 1 && pw > 1 ? 84 : 70;
        const grid = div("cells");
        grid.style.gridTemplateColumns = `repeat(${pagesW}, ${colPx}px)`;
        for (let pr = 0; pr < pagesH; pr++) {
            for (let pc = 0; pc < pagesW; pc++) {
                const pageId = pr * pagesW + pc;
                const r0 = pr * ph;
                const c0 = pc * pw;
                const r1 = r0 + ph - 1;
                const c1 = c0 + pw - 1;
                const lk = res.mapping.pageLookup[pageId];
                const cell = div("pcell");
                cell.style.width = colPx + "px";
                if (!lk) {
                    cell.classList.add("pad");
                    cell.textContent = `pg ${pageId}`;
                    cell.title = `page ${pageId} — padding`;
                } else {
                    cell.style.background = cellColorFor(lk, pageId);
                    const bc = res.mapping.banks[lk.bankId].gridCoord;
                    cell.innerHTML =
                        `<span class="from">(${r0}, ${c0})</span>` +
                        `<span class="to">→ (${r1}, ${c1})</span>`;
                    cell.title =
                        `page ${pageId}  ·  elements (${r0},${c0}) → (${r1},${c1})\n` +
                        `→ core ${lk.bankId} (${bc.x},${bc.y})  ·  shard ${lk.shardId}`;
                    registerCell(cell, pageId);
                }
                grid.appendChild(cell);
            }
        }
        const wrap = div("pagegrid");
        wrap.appendChild(grid);
        dom.elementView.appendChild(wrap);
    }

    function coreLegend(res, label) {
        const nb = Math.min(res.mapping.numBanks, 12);
        let leg = "";
        for (let i = 0; i < nb; i++) {
            const bc = res.mapping.banks[i].gridCoord;
            leg += `<span class="sw"><span class="box" style="background:${colorFor(i)}"></span>core ${i} (${bc.x},${bc.y})</span>`;
        }
        if (res.mapping.numBanks > 12) leg += `<span class="sw">… (color = core mod 12)</span>`;
        return `<span style="color:var(--muted)">${label}:</span>${leg}`;
    }

    // Page grid → core: the [pagesH × pagesW] page grid, colored by destination
    // core. This is the buffer-level view; page ids match the element grid.
    function renderComposition(res) {
        dom.pageStrip.innerHTML = "";
        dom.pageLegend.innerHTML = colorLegend(res);
        const [ph, pw] = res.tensor2dInPages;
        const cellPx = ph * pw > 400 ? 16 : 30;
        const grid = div("cells");
        grid.style.gridTemplateColumns = `repeat(${pw}, ${cellPx}px)`;
        for (let p = 0; p < ph * pw; p++) {
            const lk = res.mapping.pageLookup[p];
            const c = div("cell");
            c.style.width = cellPx + "px";
            if (!lk) {
                c.className = "cell pad";
                c.textContent = p;
            } else {
                c.style.background = cellColorFor(lk, p);
                c.textContent = p;
                const bc = res.mapping.banks[lk.bankId].gridCoord;
                c.title = `page ${p}\n→ core ${lk.bankId} (${bc.x},${bc.y})\ndevice page ${lk.devicePage} · shard ${lk.shardId}`;
                registerCell(c, p);
            }
            grid.appendChild(c);
        }
        const wrap = div("pagegrid");
        wrap.appendChild(grid);
        dom.pageStrip.appendChild(wrap);
    }

    // ① pages → shards (reuses the buffer model's shards). For interleave there are
    // no shards — each page is its own unit.
    function renderShards(res) {
        dom.shardsView.innerHTML = "";
        dom.shardsView.style.gridTemplateColumns = "";
        const m = res.mapping;
        if (res.distribution === "interleaved") {
            dom.shardLegend.innerHTML = "";
            const note = div("draghint");
            note.textContent = "Interleaved is not sharded — each page is its own unit; pages round-robin across cores.";
            dom.shardsView.appendChild(note);
            return;
        }
        const cols = m.shardGrid[m.shardGrid.length - 1] || 1;
        dom.shardsView.style.gridTemplateColumns = `repeat(${cols}, max-content)`;
        for (const shard of m.shards) {
            const card = div("shard-card");
            card.style.gridColumnStart = (shard.id % cols) + 1;
            card.style.gridRowStart = Math.floor(shard.id / cols) + 1;
            const hd = div("shard-hd");
            hd.innerHTML =
                `<span class="dot" style="background:${colorFor(shard.id)}"></span>` +
                `shard ${shard.id} <span class="core">@[${shard.gridCoord}] · ${shard.shape.join("×")}</span>`;
            registerToggle(hd, shard.pages.filter((p) => p != null));
            card.appendChild(hd);
            card.appendChild(shardGridEl(shard));
            dom.shardsView.appendChild(card);
        }
        const n = Math.min(m.numShards, 12);
        let items = "";
        for (let i = 0; i < n; i++) {
            items += `<span class="sw"><span class="box" style="background:${colorFor(i)}"></span>shard ${i}</span>`;
        }
        if (m.numShards > 12) items += `<span class="sw">… (color = shard mod 12)</span>`;
        dom.shardLegend.innerHTML = `<span style="color:var(--muted)">color = shard:</span>${items}`;
    }

    function shardGridEl(shard) {
        const cols = shard.shape[shard.shape.length - 1];
        const g = div("cells");
        g.style.gridTemplateColumns = `repeat(${cols}, 30px)`;
        for (let off = 0; off < shard.pages.length; off++) {
            const pageId = shard.pages[off];
            const c = div("cell");
            if (pageId == null) {
                c.className = "cell pad";
                c.textContent = "·";
                c.title = `shard ${shard.id}, slot ${off}\n(padding)`;
            } else {
                c.style.background = colorFor(shard.id);
                c.textContent = pageId;
                c.title = `page ${pageId}\nshard ${shard.id}, slot ${off}`;
                registerCell(c, pageId);
            }
            g.appendChild(c);
        }
        return g;
    }

    // ② shards → banks/cores (device pages, shards stacked per core).
    function renderBanks(res) {
        dom.banksView.innerHTML = "";
        const m = res.mapping;
        const interleaved = res.distribution === "interleaved";
        dom.banksView.style.gridTemplateColumns = `repeat(${m.bankGrid.x}, max-content)`;
        for (const bank of m.banks) {
            const col = div("bank");
            col.style.gridColumnStart = bank.gridCoord.x + 1;
            col.style.gridRowStart = bank.gridCoord.y + 1;
            const hd = div("bank-hd");
            hd.innerHTML = `core ${bank.bankId} <span class="core">(${bank.gridCoord.x},${bank.gridCoord.y})</span>`;
            col.appendChild(hd);

            if (interleaved) {
                const grid = div("cells");
                grid.style.gridTemplateColumns = "repeat(1, 30px)";
                const pages = [];
                for (const d of bank.devicePages) {
                    const c = div("cell");
                    c.style.background = colorFor(bank.bankId);
                    c.textContent = d.pageId;
                    c.title = `page ${d.pageId}\ncore ${bank.bankId} (${bank.gridCoord.x},${bank.gridCoord.y})\ndevice slot ${d.devicePage}`;
                    registerCell(c, d.pageId);
                    grid.appendChild(c);
                    pages.push(d.pageId);
                }
                col.appendChild(grid);
                registerToggle(hd, pages);
                dom.banksView.appendChild(col);
                continue;
            }
            if (bank.shardIds.length === 0) {
                const ee = div("shard-label");
                ee.textContent = "(empty)";
                col.appendChild(ee);
            }
            const bankPages = [];
            let devBase = 0;
            for (const sid of bank.shardIds) {
                const shard = m.shards[sid];
                const g = div("shard-group");
                const lab = div("shard-label");
                lab.textContent = `shard ${sid} · dev ${devBase}–${devBase + shard.pages.length - 1}`;
                g.appendChild(lab);
                g.appendChild(shardGridEl(shard));
                col.appendChild(g);
                devBase += shard.pages.length;
                shard.pages.forEach((p) => p != null && bankPages.push(p));
            }
            registerToggle(hd, bankPages);
            dom.banksView.appendChild(col);
        }
    }

    // ---- click to toggle ----
    dom.results.addEventListener("click", (ev) => {
        const cell = ev.target.closest("[data-page]");
        if (cell) {
            togglePage(+cell.dataset.page);
            return;
        }
        const tog = ev.target.closest("[data-toggle]");
        if (tog && tog._pages) togglePages(tog._pages);
    });

    // ---- init ----
    buildPresets();
    [dom.logicalShape, dom.tile, dom.gridX, dom.gridY, dom.bankX, dom.bankY, dom.ndShardShape].forEach((i) =>
        i.addEventListener("input", render)
    );
    [dom.layout, dom.dtype, dom.sharding, dom.orientation, dom.ndStrategy, dom.ndAlignment, dom.colorMode, dom.granularity].forEach((i) =>
        i.addEventListener("change", render)
    );
    // external hash changes (link pasted into the bar, back/forward) re-apply the
    // config; updateHash() uses replaceState so our own writes don't fire this.
    window.addEventListener("hashchange", () => {
        if ((location.hash || "").replace(/^#/, "") === lastHash) return;
        applyHash();
        render();
    });
    applyHash();
    render();
})();
