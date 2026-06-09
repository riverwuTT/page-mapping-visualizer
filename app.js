// SPDX-License-Identifier: Apache-2.0
// Visualizer UI for the two-compartment page-mapping model. Depends on the
// global `PageMapping` (page_mapping.js). All inputs are expressed in pages.
//
//   Compartment 1: pages -> shards   (page grid + shard shape)
//   Compartment 2: shards -> banks   (bank grid + distribution)

(function () {
    "use strict";
    const PM = window.PageMapping;

    // Each preset selects a sharding model and its inputs.
    const PRESETS = [
        ["Interleave", { model: "interleave", pageGrid: "20", bankX: 6, bankY: 1 }],
        ["Continuous fill", { model: "continuous", pageGrid: "8,2", shardShape: "2,2", bankX: 4, bankY: 1 }],
        ["Grid sharding", { model: "grid", pageGrid: "4,4", shardShape: "2,2", bankX: 2, bankY: 2 }],
        ["Grid (partial)", { model: "grid", pageGrid: "5,5", shardShape: "2,2", bankX: 3, bankY: 3 }],
        ["ND round-robin", { model: "nd", pageGrid: "4,4", shardShape: "2,2", bankX: 1, bankY: 2, distribution: "round_robin" }],
        ["ND grid", { model: "nd", pageGrid: "4,4", shardShape: "2,2", bankX: 2, bankY: 2, distribution: "grid_2d" }],
        ["ND rank-3", { model: "nd", pageGrid: "2,4,4", shardShape: "1,2,2", bankX: 2, bankY: 2, distribution: "round_robin" }],
    ];

    const el = (id) => document.getElementById(id);
    const dom = {
        pageGrid: el("pageGrid"),
        pageCount: el("pageCount"),
        shardShape: el("shardShape"),
        shardShapeField: el("shardShapeField"),
        bankX: el("bankX"),
        bankY: el("bankY"),
        shardingModel: el("shardingModel"),
        distribution: el("distribution"),
        ndDistField: el("ndDistField"),
        orientation: el("orientation"),
        presets: el("presets"),
        error: el("error"),
        summary: el("summary"),
        shardsView: el("shardsView"),
        shardLegend: el("shardLegend"),
        banksView: el("banksView"),
        pageLegend: el("pageLegend"),
        pageStrip: el("pageStrip"),
        results: el("results"),
        selbar: el("selbar"),
        cube3d: el("cube3d"),
        cubeNote: el("cubeNote"),
    };

    // 3D cube rotation state (experimental rank-3 view). All cube scenes on the
    // page (the page grid + every rank-3 shard) share one orientation.
    const cubeScenes = [];
    let rotX = -20;
    let rotY = -28;
    let dragging3d = false;
    let dragRotated = false;
    let lastX = 0;
    let lastY = 0;

    // ---- linked selection: click a page / shard / bank to toggle its highlight.
    // Selected pages light up in every view; everything else greys out. ----
    const cellsByPage = new Map(); // pageId -> [cell elements across all views]
    const groupToggles = []; // { el, pages } for each shard / bank header toggle
    let selection = new Set(); // selected pageIds
    let selectedCells = []; // cells currently carrying .sel

    function registerCell(cell, pageId) {
        cell.dataset.page = pageId;
        let arr = cellsByPage.get(pageId);
        if (!arr) cellsByPage.set(pageId, (arr = []));
        arr.push(cell);
    }
    // Make a header element toggle the highlight of a whole group of pages.
    function registerToggle(el, pages) {
        el.dataset.toggle = "1";
        el._pages = pages;
        groupToggles.push({ el, pages });
    }
    function applySelection() {
        selectedCells.forEach((c) => c.classList.remove("sel"));
        selectedCells = [];
        dom.results.classList.toggle("selecting", selection.size > 0);
        selection.forEach((pid) => {
            const arr = cellsByPage.get(pid);
            if (arr)
                arr.forEach((c) => {
                    c.classList.add("sel");
                    selectedCells.push(c);
                });
        });
        // a shard/bank toggle reads "active" when all of its pages are selected
        for (const { el, pages } of groupToggles) {
            el.classList.toggle("active", pages.length > 0 && pages.every((p) => selection.has(p)));
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
            if (!Number.isFinite(n)) throw new Error(`invalid shape entry: "${x}"`);
            return n;
        });
    };
    const intOf = (inp) => {
        const n = parseInt(inp.value, 10);
        if (!Number.isFinite(n)) throw new Error(`invalid number: "${inp.value}"`);
        return n;
    };

    // A single shard slot cell (page id, or "·" for a padding slot). `colorIdx`
    // tints real-page fills; `devBase` (if given) adds the device page to the title.
    function shardCellEl(shard, off, colorIdx, devBase) {
        const pageId = shard.pages[off];
        const c = div("cell");
        if (pageId == null) {
            c.className = "cell pad";
            c.textContent = "·";
            c.title = `shard ${shard.id}, slot ${off}\n(padding — overhangs page grid)`;
        } else {
            c.style.background = colorFor(colorIdx);
            c.textContent = pageId;
            c.title =
                `page ${pageId}\nshard ${shard.id}, slot ${off}` +
                (devBase != null ? `\ndevice page ${devBase + off}` : "");
            registerCell(c, pageId);
        }
        return c;
    }

    // A shard rendered as its row-major 2D grid: cols = last dim, rows = the rest.
    function shardGridEl(shard, colorIdx, devBase) {
        const cols = shard.shape[shard.shape.length - 1];
        const g = div("cells");
        g.style.gridTemplateColumns = `repeat(${cols}, 30px)`;
        for (let off = 0; off < shard.pages.length; off++) {
            g.appendChild(shardCellEl(shard, off, colorIdx, devBase));
        }
        return g;
    }

    // A rank-3 thing ([d0,d1,d2]) rendered as a rotatable stack of d0 layer
    // planes, each d1 × d2. `makeCell(i)` builds the cell for linear slot i.
    // All cubes share one orientation (cubeScenes) so dragging any rotates all.
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

    // A rank-3 shard rendered as a small cube.
    function shardCubeEl(shard) {
        return buildCube(shard.shape, (off) => shardCellEl(shard, off, shard.id, null), 24);
    }

    function buildPresets() {
        dom.presets.innerHTML = "";
        for (const [name, vals] of PRESETS) {
            const b = document.createElement("button");
            b.textContent = name;
            b.onclick = () => {
                dom.shardingModel.value = vals.model;
                dom.pageGrid.value = vals.pageGrid;
                if (vals.shardShape) dom.shardShape.value = vals.shardShape;
                dom.bankX.value = vals.bankX;
                dom.bankY.value = vals.bankY;
                if (vals.model === "nd") dom.distribution.value = vals.distribution || "round_robin";
                dom.orientation.value = vals.orientation || "row_major";
                render();
            };
            dom.presets.appendChild(b);
        }
    }

    function render() {
        dom.error.textContent = "";
        cellsByPage.clear();
        groupToggles.length = 0;
        cubeScenes.length = 0;
        selection.clear();
        selectedCells = [];
        // Map the four sharding models onto the model API:
        //   interleave -> interleaved | continuous -> legacy height
        //   grid -> legacy block      | nd -> ND (round-robin / grid_2d sub-option)
        const model = dom.shardingModel.value;
        const isInterleave = model === "interleave";
        const isND = model === "nd";
        dom.shardShapeField.style.display = isInterleave ? "none" : "";
        dom.ndDistField.style.display = isND ? "" : "none";
        let distribution = "round_robin";
        let legacyLayout = "block";
        if (model === "interleave") distribution = "interleaved";
        else if (model === "continuous") (distribution = "legacy"), (legacyLayout = "height");
        else if (model === "grid") (distribution = "legacy"), (legacyLayout = "block");
        else distribution = dom.distribution.value; // nd
        let res;
        try {
            const cfg = {
                pageGrid: parseShape(dom.pageGrid.value),
                shardShape: isInterleave ? [1] : parseShape(dom.shardShape.value),
                bankGrid: { x: intOf(dom.bankX), y: intOf(dom.bankY) },
                distribution,
                legacyLayout,
                orientation: dom.orientation.value,
            };
            dom.pageCount.textContent = `= ${PM.volume(cfg.pageGrid)} pages`;
            res = PM.computeMapping(cfg);
        } catch (e) {
            dom.error.textContent = e.message;
            dom.summary.innerHTML = "";
            dom.shardsView.innerHTML = "";
            dom.shardLegend.innerHTML = "";
            dom.banksView.innerHTML = "";
            dom.pageStrip.innerHTML = "";
            return;
        }
        renderSummary(res);
        renderShards(res);
        renderBanks(res);
        renderPageStrip(res);
        applySelection(); // selection was reset; clears any stale highlight/toolbar
    }

    const stat = (k, v) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;
    function renderSummary(res) {
        if (res.distribution === "interleaved") {
            dom.summary.innerHTML =
                stat("Pages", res.numPages) +
                stat("Banks", res.numBanks) +
                stat("Pages/bank (max)", Math.ceil(res.numPages / res.numBanks)) +
                stat("Layout", "interleaved");
            return;
        }
        const pad = res.banks.reduce((a, b) => a + b.devicePages.filter((d) => d.pageId == null).length, 0);
        let s =
            stat("Pages", res.numPages) +
            stat("Shard grid", res.shardGrid.join(" × ")) +
            stat("Shards", res.numShards) +
            stat("Shard volume", res.shardVolume) +
            stat("Banks", res.numBanks) +
            stat("Padding slots", pad);
        // shard rank < page-grid rank -> shapes were folded (squeeze_shape_ranks)
        if (res.squeezed) {
            s += stat("Squeezed", `[${res.inputShardShape}] → page [${res.squeezedPageGrid}] / shard [${res.shardShape}]`);
        }
        dom.summary.innerHTML = s;
    }

    function renderShards(res) {
        dom.shardsView.innerHTML = "";
        dom.shardsView.style.gridTemplateColumns = "";
        if (res.distribution === "interleaved") {
            // interleaved isn't sharded — skip the (numPages) trivial 1-page cards.
            dom.shardLegend.innerHTML = "";
            const note = div("draghint");
            note.textContent =
                "Interleaved is not sharded — each page is its own unit; pages round-robin across banks (below).";
            dom.shardsView.appendChild(note);
            return;
        }
        // Arrange shard cards in the shard grid (shard id is row-major over it).
        const cols = res.shardGrid[res.shardGrid.length - 1] || 1;
        dom.shardsView.style.gridTemplateColumns = `repeat(${cols}, max-content)`;
        for (const shard of res.shards) {
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
            card.appendChild(use3d ? shardCubeEl(shard) : shardGridEl(shard, shard.id, null));
            dom.shardsView.appendChild(card);
        }
        // legend
        const n = Math.min(res.numShards, 12);
        let items = "";
        for (let i = 0; i < n; i++) {
            items += `<span class="sw"><span class="box" style="background:${colorFor(i)}"></span>shard ${i}</span>`;
        }
        if (res.numShards > 12) items += `<span class="sw">… (color = shard mod 12)</span>`;
        dom.shardLegend.innerHTML = `<span style="color:var(--muted)">color = shard:</span>${items}`;
    }

    function renderBanks(res) {
        dom.banksView.innerHTML = "";
        const interleaved = res.distribution === "interleaved";
        // Arrange the bank cells as the bank grid: bank (x,y) sits at column x, row y.
        dom.banksView.style.gridTemplateColumns = `repeat(${res.bankGrid.x}, max-content)`;
        for (const bank of res.banks) {
            const col = div("bank");
            col.style.gridColumnStart = bank.gridCoord.x + 1;
            col.style.gridRowStart = bank.gridCoord.y + 1;
            const hd = div("bank-hd");
            hd.innerHTML = `bank ${bank.bankId} <span class="core">(${bank.gridCoord.x},${bank.gridCoord.y})</span>`;
            col.appendChild(hd);

            if (interleaved) {
                // a column of the bank's pages (one per device slot), colored by bank
                const grid = div("cells");
                grid.style.gridTemplateColumns = "repeat(1, 30px)";
                const pages = [];
                for (const d of bank.devicePages) {
                    const c = div("cell");
                    c.style.background = colorFor(bank.bankId);
                    c.textContent = d.pageId;
                    c.title = `page ${d.pageId}\nbank ${bank.bankId} (${bank.gridCoord.x},${bank.gridCoord.y})\ndevice slot ${d.devicePage}`;
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
                const e = div("shard-label");
                e.textContent = "(empty)";
                col.appendChild(e);
            }
            // device pages are the assigned shards stacked in order
            const bankPages = [];
            let devBase = 0;
            for (const sid of bank.shardIds) {
                const shard = res.shards[sid];
                const g = div("shard-group");
                const lab = div("shard-label");
                lab.textContent = `shard ${sid} · dev ${devBase}–${devBase + shard.pages.length - 1}`;
                g.appendChild(lab);
                g.appendChild(shardGridEl(shard, sid, devBase));
                col.appendChild(g);
                devBase += shard.pages.length;
                shard.pages.forEach((p) => p != null && bankPages.push(p));
            }
            registerToggle(hd, bankPages);
            dom.banksView.appendChild(col);
        }
    }

    // Pages laid out in their grid form (row-major reshape of the page grid),
    // colored by the bank each lands in. cols = last grid dim; higher dims stack
    // as successive blocks of rows, with a separator between the slowest-varying
    // blocks so N-D grids stay readable.
    function renderPageStrip(res) {
        dom.pageStrip.innerHTML = "";
        // legend: color = destination bank
        const nb = Math.min(res.numBanks, 12);
        let leg = "";
        for (let i = 0; i < nb; i++) {
            const bc = res.banks[i].gridCoord;
            leg += `<span class="sw"><span class="box" style="background:${colorFor(i)}"></span>bank ${i} (${bc.x},${bc.y})</span>`;
        }
        if (res.numBanks > 12) leg += `<span class="sw">… (color = bank mod 12)</span>`;
        dom.pageLegend.innerHTML = `<span style="color:var(--muted)">color = destination bank:</span>${leg}`;

        // The page-grid cube depends on the page grid's own rank; shards cube
        // independently on each shard's rank (see renderShards).
        const use3d = dom.cube3d.checked && res.pageGrid.length === 3;
        dom.cubeNote.textContent =
            dom.cube3d.checked && res.pageGrid.length !== 3
                ? `3D cube renders rank-3 components; this page grid is rank-${res.pageGrid.length}.`
                : "";
        if (use3d) {
            renderPageCube(res);
            return;
        }

        const shape = res.pageGrid;
        const cols = shape[shape.length - 1] || res.numPages || 1;
        // size of one slowest-varying block (everything but dim 0), in pages
        const blockSize = shape.length > 2 ? res.numPages / shape[0] : res.numPages;
        const wrap = div("pagegrid");
        let grid = null;
        for (let p = 0; p < res.numPages; p++) {
            if (p % blockSize === 0) {
                grid = div("cells");
                grid.style.gridTemplateColumns = `repeat(${cols}, 30px)`;
                wrap.appendChild(grid);
            }
            grid.appendChild(pageCellEl(res, p));
        }
        dom.pageStrip.appendChild(wrap);
    }

    // A single page cell for the composition views, colored by destination bank.
    function pageCellEl(res, p) {
        const lk = res.pageLookup[p];
        const c = div("cell");
        if (!lk) {
            c.className = "cell pad";
            c.textContent = p;
        } else {
            c.style.background = colorFor(lk.bankId);
            c.textContent = p;
            const bc = res.banks[lk.bankId].gridCoord;
            c.title =
                `page ${p}\n→ bank ${lk.bankId} (${bc.x},${bc.y})\n` +
                `device page ${lk.devicePage} · shard ${lk.shardId}`;
            registerCell(c, p);
        }
        return c;
    }

    // Experimental: render a rank-3 page grid as a rotatable stack of layer
    // planes (a "cube"). z = dim 0; each plane is the dim1 × dim2 grid. Cells are
    // the same page cells used elsewhere, so highlight / selection works in 3D.
    function renderPageCube(res) {
        const stage = buildCube(res.pageGrid, (p) => pageCellEl(res, p), 30);
        const hint = div("cube-hint");
        hint.textContent = "drag to rotate";
        stage.appendChild(hint);
        dom.pageStrip.appendChild(stage);
    }

    // ---- click to toggle: a page cell, or a whole shard / bank header ----
    dom.results.addEventListener("click", (e) => {
        if (dragRotated) {
            dragRotated = false; // this "click" was the end of a cube rotation
            return;
        }
        const cell = e.target.closest(".cell[data-page]");
        if (cell) {
            togglePage(+cell.dataset.page);
            return;
        }
        const tog = e.target.closest("[data-toggle]");
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
    [dom.pageGrid, dom.shardShape, dom.bankX, dom.bankY].forEach((i) =>
        i.addEventListener("input", render)
    );
    [dom.shardingModel, dom.distribution, dom.orientation, dom.cube3d].forEach((i) =>
        i.addEventListener("change", render)
    );
    render();
})();
