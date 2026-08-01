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
        ["Tile · width×3 (wrap)", { logicalShape: "64,96", layout: "TILE", sharding: "width", gridX: 2, gridY: 1, shardW: 32 }],
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
        shardDimField: el("shardDimField"),
        shardHField: el("shardHField"),
        shardWField: el("shardWField"),
        shardH: el("shardH"),
        shardW: el("shardW"),
        orientation: el("orientation"),
        ndField: el("ndField"),
        ndShardShape: el("ndShardShape"),
        ndStrategy: el("ndStrategy"),
        ndAlignment: el("ndAlignment"),
        bankField: el("bankField"),
        bankX: el("bankX"),
        bankY: el("bankY"),
        customAlignment: el("customAlignment"),
        showElementCores: el("showElementCores"),
        elementCoresSection: el("elementCoresSection"),
        elementCoresView: el("elementCoresView"),
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
        cube3d: el("cube3d"),
        cubeNote: el("cubeNote"),
        shardLegend: el("shardLegend"),
        shardsView: el("shardsView"),
        banksView: el("banksView"),
        results: el("results"),
        selbar: el("selbar"),
    };

    // 3D cube rotation state (experimental rank-3 view). All cube scenes on the
    // page (the page grid + every rank-3 shard) share one orientation — same
    // model as the buffer visualizer's cube3d.
    const cubeScenes = [];
    let rotX = -20;
    let rotY = -28;
    let dragging3d = false;
    let dragRotated = false;
    let lastX = 0;
    let lastY = 0;

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
    // Optional positive int — blank means "unset" (let the model auto-derive).
    const optIntOf = (inp) => (inp.value.trim() === "" ? undefined : intOf(inp));

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
        const sm = dom.sharding.value;
        if (sm === "block" && dom.shardH.value.trim()) p.push("sh=" + dom.shardH.value.trim());
        if ((sm === "width" || sm === "block") && dom.shardW.value.trim()) p.push("sw=" + dom.shardW.value.trim());
        if (dom.sharding.value === "nd") {
            p.push("ndshape=" + shapeEnc(dom.ndShardShape.value));
            p.push("ndstrat=" + dom.ndStrategy.value);
            p.push("ndalign=" + dom.ndAlignment.value);
        }
        p.push("orient=" + dom.orientation.value);
        if (dom.customAlignment.value.trim()) p.push("align=" + shapeEnc(dom.customAlignment.value));
        if (!dom.showElementCores.checked) p.push("elcores=0");
        if (!dom.cube3d.checked) p.push("cube3d=0");
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
        // shard dims are auto (blank) unless the link carries them — reset first so
        // an incoming link fully determines them rather than inheriting stale input.
        dom.shardH.value = "";
        dom.shardW.value = "";
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
        setV(dom.shardH, obj.sh);
        setV(dom.shardW, obj.sw);
        setV(dom.ndShardShape, obj.ndshape && shapeDec(obj.ndshape));
        setV(dom.ndStrategy, obj.ndstrat);
        setV(dom.ndAlignment, obj.ndalign);
        setV(dom.orientation, obj.orient);
        // Custom alignment is optional — clear first so a link without `align=`
        // does not inherit a stale value from a previous config.
        dom.customAlignment.value = "";
        setV(dom.customAlignment, obj.align && shapeDec(obj.align));
        // Elements→cores section defaults on; only an explicit elcores=0 turns it off.
        dom.showElementCores.checked = obj.elcores !== "0";
        // 3D cube defaults on; only an explicit cube3d=0 turns it off.
        dom.cube3d.checked = obj.cube3d !== "0";
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
                dom.shardH.value = vals.shardH != null ? vals.shardH : "";
                dom.shardW.value = vals.shardW != null ? vals.shardW : "";
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
        // classic sharding takes an explicit shard shape — but height sharding is
        // always the even auto-split, so only width (shard width) and block (both
        // dims) expose an input.
        const hasCustomShard = sharding === "width" || sharding === "block";
        dom.shardDimField.style.display = hasCustomShard ? "" : "none";
        dom.shardHField.style.display = sharding === "block" ? "" : "none";
        dom.shardWField.style.display = sharding === "width" || sharding === "block" ? "" : "none";
    }

    function render() {
        dom.error.textContent = "";
        dom.warn.textContent = "";
        cellsByPage.clear();
        groupToggles.length = 0;
        cubeScenes.length = 0;
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
            // Height sharding always auto-splits (shard height = full-tensor-height /
            // num-cores); only width and block take an explicit shard dimension.
            if (cfg.sharding === "block") cfg.shardHeight = optIntOf(dom.shardH);
            if (cfg.sharding === "width" || cfg.sharding === "block") cfg.shardWidth = optIntOf(dom.shardW);
            if (cfg.sharding === "nd") {
                cfg.ndShardShape = parseShape(dom.ndShardShape.value);
                cfg.ndStrategy = dom.ndStrategy.value;
                cfg.ndAlignment = dom.ndAlignment.value;
            }
            if (dom.customAlignment.value.trim()) {
                cfg.alignment = parseShape(dom.customAlignment.value);
            }
            res = TM.computeTensorMapping(cfg);
        } catch (e) {
            dom.error.textContent = e.message;
            ["summary", "elementView", "elementLegend", "pageStrip", "pageLegend", "shardsView", "shardLegend", "banksView", "elementCoresView"].forEach(
                (k) => (dom[k].innerHTML = "")
            );
            return;
        }

        renderSummary(res);
        renderGranularityView(res);
        renderComposition(res);
        renderShards(res);
        renderBanks(res);
        renderElementCores(res);
        updateShardHints(res);
        applySelection();
    }

    // Reflect the effective (post-alignment) shard shape as the placeholder for the
    // shard-dim inputs, and warn when a classic 1D shard oversubscribes its cores.
    function updateShardHints(res) {
        const nd = res.ndShardShape;
        const sharded = res.distribution !== "interleaved" && nd && nd.length >= 2;
        dom.shardH.placeholder = sharded ? String(nd[nd.length - 2]) : "auto";
        dom.shardW.placeholder = sharded ? String(nd[nd.length - 1]) : "auto";
        // A custom width shard can oversubscribe the cores; height auto-splits, so
        // it never does.
        if (dom.sharding.value === "width" && res.mapping.numShards > res.mapping.numBanks && !dom.warn.textContent) {
            dom.warn.textContent =
                `${res.mapping.numShards} shards over ${res.mapping.numBanks} core${res.mapping.numBanks === 1 ? "" : "s"} — ` +
                `shards wrap round-robin (>1 per core). Classic width sharding normally requires shards ≤ cores.`;
        }
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
            rows.push(["Shard (elements)", res.ndShardShape ? res.ndShardShape.join(" × ") : "—"]);
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
    // The heading reads e.g. "Elements → Cores". With 3D cube on, rank-3
    // element / page views render as rotatable cubes (tiles stay flat).
    function renderGranularityView(res) {
        const g = granularity();
        dom.granHeading.textContent = `${GRAN_WORD[g]} → ${DEST_WORD[colorMode()]}`;
        dom.elementView.innerHTML = "";
        dom.elementLegend.innerHTML = "";
        updateCubeNote(res);
        if (g === "page") return renderPageView(res);
        if (g === "tile") return renderTileView(res);
        return renderElementGrid(res);
    }

    // Shared status for the 3D toggle: front mapping + page-grid / shards.
    function updateCubeNote(res) {
        if (!dom.cube3d.checked) {
            dom.cubeNote.textContent = "";
            return;
        }
        const g = granularity();
        const paddedRank = res.paddedShape.length;
        const pageGrid = res.mapping.pageGrid || res.tensorShapeInPages;
        const pageRank = pageGrid ? pageGrid.length : 0;
        const frontCubes =
            (g === "element" && paddedRank === 3) || (g === "page" && pageRank === 3);
        const lowerCubes =
            pageRank === 3 ||
            (res.mapping.shards || []).some((s) => s.shape && s.shape.length === 3);
        if (frontCubes) {
            dom.cubeNote.textContent = "";
            return;
        }
        if (g === "tile" && lowerCubes) {
            dom.cubeNote.textContent =
                "Tile view stays flat; page grid / shards still cube when rank-3.";
            return;
        }
        if (g === "element" && paddedRank !== 3 && lowerCubes) {
            dom.cubeNote.textContent =
                `Front element view stays flat (padded shape rank-${paddedRank}); page grid / shards still cube when rank-3.`;
            return;
        }
        if (g === "page" && pageRank !== 3 && lowerCubes) {
            dom.cubeNote.textContent =
                `Front page view stays flat (page grid rank-${pageRank}); shards still cube when rank-3.`;
            return;
        }
        if (!frontCubes && !lowerCubes) {
            dom.cubeNote.textContent =
                `3D cube needs rank-3 components; padded shape rank-${paddedRank}, page grid rank-${pageRank}.`;
        }
    }

    // Page granularity: one cell per page, laid out in the page grid, labeled with
    // the element coordinate range it covers. Reuses the element grid's page-cell
    // renderer (also the oversized-tensor fallback), colored by the active mode.
    // With 3D on and a rank-3 page grid, render the same rotatable cube used below.
    function renderPageView(res) {
        dom.elementLegend.innerHTML = colorLegend(res);
        const pageGrid = res.mapping.pageGrid || res.tensorShapeInPages;
        if (dom.cube3d.checked && pageGrid && pageGrid.length === 3) {
            const stage = buildCube(pageGrid, (p) => pageCellEl(res, p, 30), 30);
            const hint = div("cube-hint");
            hint.textContent = "drag to rotate";
            stage.appendChild(hint);
            dom.elementView.appendChild(stage);
            return;
        }
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
    // With 3D cube on and a rank-3 padded shape, render planes of the N-D tensor
    // instead of the folded 2D view (z = dim0, each plane is dim1 × dim2).
    function renderElementGrid(res) {
        dom.elementView.innerHTML = "";
        dom.elementLegend.innerHTML = colorLegend(res);
        if (dom.cube3d.checked && res.paddedShape.length === 3) {
            if (renderElementCube(res)) return;
        }
        const e = res.element;
        const total = e.H * e.W;
        const MAX = 8192;
        if (total > MAX) {
            dom.warn.textContent =
                `Element grid has ${total} elements (> ${MAX}) — too many to draw one cell per element. ` +
                `Showing one cell per page instead, laid out in the element grid's shape, each labeled with the ` +
                `element coordinate range [from] → [to] it covers.`;
            renderPageCoordGrid(res);
            return;
        }
        const cellPx = total > 2500 ? 11 : total > 900 ? 15 : 22;
        const grid = div("elemgrid");
        grid.style.gridTemplateColumns = `repeat(${e.W}, ${cellPx}px)`;
        grid.style.setProperty("--ph", e.ph);
        grid.style.setProperty("--pw", e.pw);
        const frag = document.createDocumentFragment();
        for (let r = 0; r < e.H; r++) {
            for (let c = 0; c < e.W; c++) {
                frag.appendChild(elementGridCell(res, r, c, cellPx, /*pageEdges*/ true));
            }
        }
        grid.appendChild(frag);
        dom.elementView.appendChild(grid);
    }

    // Rank-3 padded shape as a cube of element planes. Linear fold to the physical
    // 2D view: element (z,y,x) → (r = z·d1 + y, c = x), matching compute_physical_shape.
    function renderElementCube(res) {
        const [d0, d1, d2] = res.paddedShape;
        const total = d0 * d1 * d2;
        const MAX = 8192;
        if (total > MAX) {
            dom.warn.textContent =
                `3D element cube has ${d0} × ${d1} × ${d2} = ${total} cells (> ${MAX}) — too many to draw. ` +
                `Showing the folded 2D element grid instead.`;
            return false;
        }
        const cellPx = total > 2500 ? 8 : total > 900 ? 11 : 14;
        const e = res.element;
        const stage = buildCube([d0, d1, d2], (i) => {
            const z = Math.floor(i / (d1 * d2));
            const rem = i % (d1 * d2);
            const y = Math.floor(rem / d2);
            const x = rem % d2;
            const r = z * d1 + y;
            const c = x;
            return elementGridCell(res, r, c, cellPx, /*pageEdges*/ false, [z, y, x]);
        }, cellPx);
        // Tighter plane gap for dense element cubes so layers stay readable.
        stage.querySelectorAll(".cube-plane").forEach((plane, z) => {
            const gap = cellPx < 12 ? 48 : 72;
            plane.style.transform = `translate(-50%, -50%) translateZ(${(z - (d0 - 1) / 2) * gap}px)`;
        });
        const hint = div("cube-hint");
        hint.textContent = `drag to rotate · padded ${d0}×${d1}×${d2}` +
            (e.logicalH !== e.H || e.logicalW !== e.W ? " (incl. padding)" : "");
        stage.appendChild(hint);
        dom.elementView.appendChild(stage);
        return true;
    }

    function elementGridCell(res, r, c, cellPx, pageEdges, ndCoord) {
        const e = res.element;
        const pageId = e.pageOf(r, c);
        const lk = res.mapping.pageLookup[pageId];
        const cell = div("ecell");
        cell.style.width = cell.style.height = cellPx + "px";
        if (pageEdges) {
            if (r % e.ph === 0) cell.classList.add("top");
            if (c % e.pw === 0) cell.classList.add("left");
            if (r === e.H - 1) cell.classList.add("bot");
            if (c === e.W - 1) cell.classList.add("right");
        }
        const ndLabel = ndCoord ? `element [${ndCoord.join(", ")}]\n` : `element (${r}, ${c})\n`;
        if (e.isPadding(r, c)) {
            cell.classList.add("pad");
            cell.title = `${ndLabel}— padding\npage ${pageId}`;
        } else if (lk) {
            cell.style.background = cellColorFor(lk, pageId);
            const bc = res.mapping.banks[lk.bankId].gridCoord;
            cell.title =
                `${ndLabel}→ page ${pageId}\n→ shard ${lk.shardId}\n→ core ${lk.bankId} (${bc.x},${bc.y})`;
            registerCell(cell, pageId);
        }
        return cell;
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

    // Page grid → core: the buffer-level view; page ids match the element grid.
    // Rank-3 page grids (typical ND TILE) can render as a rotatable cube when
    // "3D cube" is on — using mapping.pageGrid, not the folded 2D view.
    function renderComposition(res) {
        dom.pageStrip.innerHTML = "";
        dom.pageLegend.innerHTML = colorLegend(res);
        const pageGrid = res.mapping.pageGrid || res.tensorShapeInPages;
        const use3d = dom.cube3d.checked && pageGrid && pageGrid.length === 3;
        if (use3d) {
            renderPageCube(res, pageGrid);
            return;
        }
        const [ph, pw] = res.tensor2dInPages;
        const cellPx = ph * pw > 400 ? 16 : 30;
        const grid = div("cells");
        grid.style.gridTemplateColumns = `repeat(${pw}, ${cellPx}px)`;
        for (let p = 0; p < ph * pw; p++) {
            grid.appendChild(pageCellEl(res, p, cellPx));
        }
        const wrap = div("pagegrid");
        wrap.appendChild(grid);
        dom.pageStrip.appendChild(wrap);
    }

    function pageCellEl(res, p, cellPx) {
        const lk = res.mapping.pageLookup[p];
        const c = div("cell");
        if (cellPx) c.style.width = cellPx + "px";
        if (!lk) {
            c.className = "cell pad";
            c.textContent = p;
        } else {
            c.style.background = cellColorFor(lk, p);
            c.textContent = p;
            const bc = res.mapping.banks[lk.bankId].gridCoord;
            c.title =
                `page ${p}\n→ core ${lk.bankId} (${bc.x},${bc.y})\n` +
                `device page ${lk.devicePage} · shard ${lk.shardId}`;
            registerCell(c, p);
        }
        return c;
    }

    // A rank-3 thing ([d0,d1,d2]) rendered as a rotatable stack of d0 layer
    // planes, each d1 × d2. `makeCell(i)` builds the cell for linear slot i.
    function buildCube([d0, d1, d2], makeCell, cellPx) {
        const stage = div("cube-stage" + (cellPx < 30 ? " small" : ""));
        const scene = div("cube-scene");
        scene.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
        cubeScenes.push(scene);
        const planeGap = cellPx < 30 ? 64 : 96;
        for (let z = 0; z < d0; z++) {
            const plane = div("cube-plane");
            plane.style.transform = `translate(-50%, -50%) translateZ(${(z - (d0 - 1) / 2) * planeGap}px)`;
            const lab = div("cube-plane-label");
            lab.textContent = `z = ${z}`;
            plane.appendChild(lab);
            const grid = div("cells");
            grid.style.gridTemplateColumns = `repeat(${d2}, ${cellPx}px)`;
            for (let i = 0; i < d1 * d2; i++) grid.appendChild(makeCell(z * d1 * d2 + i));
            plane.appendChild(grid);
            scene.appendChild(plane);
        }
        stage.appendChild(scene);
        stage.addEventListener("mousedown", (e) => {
            dragging3d = true;
            dragRotated = false;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        return stage;
    }

    function renderPageCube(res, pageGrid) {
        const stage = buildCube(pageGrid, (p) => pageCellEl(res, p, 30), 30);
        const hint = div("cube-hint");
        hint.textContent = "drag to rotate";
        stage.appendChild(hint);
        dom.pageStrip.appendChild(stage);
    }

    function shardCubeEl(shard) {
        return buildCube(shard.shape, (off) => shardCellEl(shard, off), 24);
    }

    function shardCellEl(shard, off) {
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
        return c;
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
            const use3d = dom.cube3d.checked && shard.shape.length === 3;
            card.appendChild(use3d ? shardCubeEl(shard) : shardGridEl(shard));
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

    // Result: elements → core — same core-grid layout as shards → cores, but each
    // core shows the *elements* it owns (pages expanded by page shape).
    const ELCORE_MAX = 8192;
    function renderElementCores(res) {
        const show = dom.showElementCores.checked;
        dom.elementCoresSection.style.display = show ? "" : "none";
        dom.elementCoresView.innerHTML = "";
        const prevNote = dom.elementCoresSection.querySelector(".elcore-note");
        if (prevNote) prevNote.remove();
        if (!show) return;

        const m = res.mapping;
        const e = res.element;
        const interleaved = res.distribution === "interleaved";
        // Budget: physical element volume (every element appears on exactly one core).
        const pageFallback = e.H * e.W > ELCORE_MAX;

        if (pageFallback) {
            const note = div("draghint elcore-note");
            note.textContent =
                `Element volume ${e.H} × ${e.W} = ${e.H * e.W} exceeds ${ELCORE_MAX} — ` +
                `showing one cell per page labeled with its element coordinate range.`;
            dom.elementCoresSection.insertBefore(note, dom.elementCoresView);
        }

        dom.elementCoresView.style.gridTemplateColumns = `repeat(${m.bankGrid.x}, max-content)`;
        for (const bank of m.banks) {
            const col = div("bank");
            col.style.gridColumnStart = bank.gridCoord.x + 1;
            col.style.gridRowStart = bank.gridCoord.y + 1;
            const hd = div("bank-hd");
            hd.innerHTML = `core ${bank.bankId} <span class="core">(${bank.gridCoord.x},${bank.gridCoord.y})</span>`;
            col.appendChild(hd);

            if (interleaved) {
                const pages = [];
                for (const d of bank.devicePages) {
                    pages.push(d.pageId);
                    col.appendChild(
                        pageFallback
                            ? pageRangeCell(res, d.pageId, bank.bankId)
                            : pageElementBlock(res, d.pageId, bank.bankId)
                    );
                }
                if (pages.length === 0) {
                    const ee = div("shard-label");
                    ee.textContent = "(empty)";
                    col.appendChild(ee);
                }
                registerToggle(hd, pages);
                dom.elementCoresView.appendChild(col);
                continue;
            }

            if (bank.shardIds.length === 0) {
                const ee = div("shard-label");
                ee.textContent = "(empty)";
                col.appendChild(ee);
            }
            const bankPages = [];
            for (const sid of bank.shardIds) {
                const shard = m.shards[sid];
                const g = div("shard-group");
                const lab = div("shard-label");
                lab.textContent = `shard ${sid} · ${elementShardLabel(res, shard)}`;
                g.appendChild(lab);
                g.appendChild(
                    pageFallback
                        ? shardPageRangeGrid(res, shard, bank.bankId)
                        : shardElementGrid(res, shard, bank.bankId)
                );
                col.appendChild(g);
                shard.pages.forEach((p) => p != null && bankPages.push(p));
            }
            registerToggle(hd, bankPages);
            dom.elementCoresView.appendChild(col);
        }
    }

    function elementShardLabel(res, shard) {
        const e = res.element;
        const shape = shard.shape;
        const spw = shape[shape.length - 1];
        const sph = shape.length >= 2 ? shape[shape.length - 2] : 1;
        const leading = shape.length > 2 ? shape.slice(0, shape.length - 2).reduce((a, b) => a * b, 1) : 1;
        return `${leading * sph * e.ph}×${spw * e.pw} elem`;
    }

    // Expand one page into a ph×pw element block (interleaved per-page stack).
    function pageElementBlock(res, pageId, bankId) {
        const e = res.element;
        const { ph, pw, pagesW } = e;
        const pr = Math.floor(pageId / pagesW);
        const pc = pageId % pagesW;
        const cellPx = ph * pw > 900 ? 6 : ph * pw > 200 ? 10 : 14;
        const grid = div("elcore-grid");
        grid.style.gridTemplateColumns = `repeat(${pw}, ${cellPx}px)`;
        grid.style.marginBottom = "6px";
        const frag = document.createDocumentFragment();
        for (let lr = 0; lr < ph; lr++) {
            for (let lc = 0; lc < pw; lc++) {
                const r = pr * ph + lr;
                const c = pc * pw + lc;
                frag.appendChild(elementCell(res, r, c, pageId, bankId, cellPx));
            }
        }
        grid.appendChild(frag);
        return grid;
    }

    // One labeled page cell with element coordinate range (size-guard fallback).
    function pageRangeCell(res, pageId, bankId) {
        const e = res.element;
        const { ph, pw, pagesW } = e;
        const pr = Math.floor(pageId / pagesW);
        const pc = pageId % pagesW;
        const r0 = pr * ph, c0 = pc * pw;
        const r1 = r0 + ph - 1, c1 = c0 + pw - 1;
        const wrap = div("cells");
        wrap.style.gridTemplateColumns = "1fr";
        wrap.style.marginBottom = "4px";
        const cell = div("pcell");
        cell.style.width = "84px";
        cell.style.background = colorFor(bankId);
        cell.innerHTML =
            `<span class="from">(${r0}, ${c0})</span>` +
            `<span class="to">→ (${r1}, ${c1})</span>`;
        cell.title =
            `page ${pageId}  ·  elements (${r0},${c0}) → (${r1},${c1})\n` +
            `→ core ${bankId}`;
        registerCell(cell, pageId);
        wrap.appendChild(cell);
        return wrap;
    }

    // Flat element grid for a shard: pages arranged in shard.shape, each expanded
    // by page shape. Leading shard dims (rank > 2) fold into height.
    function shardElementGrid(res, shard, bankId) {
        const e = res.element;
        const { ph, pw, pagesW } = e;
        const shape = shard.shape;
        const spw = shape[shape.length - 1];
        const sph = shape.length >= 2 ? shape[shape.length - 2] : 1;
        const leading = shape.length > 2 ? shape.slice(0, shape.length - 2).reduce((a, b) => a * b, 1) : 1;
        const pageRows = leading * sph;
        const elemH = pageRows * ph;
        const elemW = spw * pw;
        const total = elemH * elemW;
        const cellPx = total > 2500 ? 6 : total > 900 ? 10 : 14;
        const grid = div("elcore-grid");
        grid.style.gridTemplateColumns = `repeat(${elemW}, ${cellPx}px)`;
        const frag = document.createDocumentFragment();
        for (let lr = 0; lr < elemH; lr++) {
            for (let lc = 0; lc < elemW; lc++) {
                const lpr = Math.floor(lr / ph);
                const lpc = Math.floor(lc / pw);
                const pageOff = lpr * spw + lpc;
                const pageId = shard.pages[pageOff];
                if (pageId == null) {
                    const cell = div("ecell pad");
                    cell.style.width = cell.style.height = cellPx + "px";
                    cell.title = `shard ${shard.id} padding\nlocal element (${lr}, ${lc})`;
                    frag.appendChild(cell);
                    continue;
                }
                const pr = Math.floor(pageId / pagesW);
                const pc = pageId % pagesW;
                const r = pr * ph + (lr % ph);
                const c = pc * pw + (lc % pw);
                frag.appendChild(elementCell(res, r, c, pageId, bankId, cellPx));
            }
        }
        grid.appendChild(frag);
        return grid;
    }

    // Page-range fallback for a shard (same layout as shardGridEl, labeled ranges).
    function shardPageRangeGrid(res, shard, bankId) {
        const e = res.element;
        const { ph, pw, pagesW } = e;
        const cols = shard.shape[shard.shape.length - 1];
        const g = div("cells");
        g.style.gridTemplateColumns = `repeat(${cols}, 84px)`;
        for (let off = 0; off < shard.pages.length; off++) {
            const pageId = shard.pages[off];
            if (pageId == null) {
                const c = div("pcell pad");
                c.style.width = "84px";
                c.textContent = "·";
                c.title = `shard ${shard.id}, slot ${off}\n(padding)`;
                g.appendChild(c);
                continue;
            }
            const pr = Math.floor(pageId / pagesW);
            const pc = pageId % pagesW;
            const r0 = pr * ph, c0 = pc * pw;
            const r1 = r0 + ph - 1, c1 = c0 + pw - 1;
            const c = div("pcell");
            c.style.width = "84px";
            c.style.background = colorFor(bankId);
            c.innerHTML =
                `<span class="from">(${r0}, ${c0})</span>` +
                `<span class="to">→ (${r1}, ${c1})</span>`;
            c.title =
                `page ${pageId}  ·  elements (${r0},${c0}) → (${r1},${c1})\n` +
                `shard ${shard.id}, slot ${off} → core ${bankId}`;
            registerCell(c, pageId);
            g.appendChild(c);
        }
        return g;
    }

    function elementCell(res, r, c, pageId, bankId, cellPx) {
        const e = res.element;
        const cell = div("ecell");
        cell.style.width = cell.style.height = cellPx + "px";
        if (e.isPadding(r, c)) {
            cell.classList.add("pad");
            cell.title = `element (${r}, ${c}) — padding\npage ${pageId} → core ${bankId}`;
        } else {
            cell.style.background = colorFor(bankId);
            const bc = res.mapping.banks[bankId].gridCoord;
            cell.title =
                `element (${r}, ${c})\n→ page ${pageId}\n→ core ${bankId} (${bc.x},${bc.y})`;
            registerCell(cell, pageId);
        }
        return cell;
    }

    // ---- click to toggle ----
    dom.results.addEventListener("click", (ev) => {
        if (dragRotated) {
            dragRotated = false; // this "click" was the end of a cube rotation
            return;
        }
        const cell = ev.target.closest("[data-page]");
        if (cell) {
            togglePage(+cell.dataset.page);
            return;
        }
        const tog = ev.target.closest("[data-toggle]");
        if (tog && tog._pages) togglePages(tog._pages);
    });

    // ---- drag anywhere to rotate the 3D cube ----
    document.addEventListener("mousemove", (e) => {
        if (!dragging3d || !cubeScenes.length) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;
        rotY += dx * 0.4;
        rotX = Math.max(-89, Math.min(89, rotX - dy * 0.4));
        if (Math.abs(dx) + Math.abs(dy) > 2) dragRotated = true;
        const t = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
        cubeScenes.forEach((s) => (s.style.transform = t));
    });
    document.addEventListener("mouseup", () => {
        dragging3d = false;
    });

    // ---- init ----
    buildPresets();
    [dom.logicalShape, dom.tile, dom.gridX, dom.gridY, dom.bankX, dom.bankY, dom.ndShardShape, dom.shardH, dom.shardW, dom.customAlignment].forEach((i) =>
        i.addEventListener("input", render)
    );
    [dom.layout, dom.dtype, dom.sharding, dom.orientation, dom.ndStrategy, dom.ndAlignment, dom.colorMode, dom.granularity, dom.cube3d].forEach((i) =>
        i.addEventListener("change", render)
    );
    dom.showElementCores.addEventListener("change", render);
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
