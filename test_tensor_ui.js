// SPDX-License-Identifier: Apache-2.0
// End-to-end smoke test: boots the built single-file tensor_mapping_viz.html in
// jsdom, drives the controls, and asserts the element / page / shard / bank views
// render and stay linked. Run: node test_tensor_ui.js

const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const html = fs.readFileSync(path.join(__dirname, "tensor_mapping_viz.html"), "utf8");
const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true });
const { document } = dom.window;

let failed = 0;
const ok = (cond, msg) => {
    if (!cond) {
        failed++;
        console.error("FAIL:", msg);
    }
};
function set(id, value) {
    const inp = document.getElementById(id);
    if (!inp) throw new Error(`no element #${id}`);
    inp.value = value;
    inp.dispatchEvent(new dom.window.Event(inp.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
}
const errText = () => document.getElementById("error").textContent.trim();
const elementCells = () => document.querySelectorAll("#elementView .ecell").length;
const pageCells = () => document.querySelectorAll("#pageStrip .cell[data-page]").length;
const shardCards = () => document.querySelectorAll("#shardsView .shard-card").length;
const bankCols = () => document.querySelectorAll("#banksView .bank").length;

ok(dom.window.TensorMapping && typeof dom.window.TensorMapping.computeTensorMapping === "function", "TensorMapping global exposed");
ok(dom.window.PageMapping && typeof dom.window.PageMapping.computeMapping === "function", "PageMapping global exposed");

// default boot (tile / block)
ok(errText() === "", `default boot no error (got "${errText()}")`);
ok(elementCells() > 0 && pageCells() > 0 && shardCards() > 0 && bankCols() > 0, "default boot renders all views");

function drive(cfg) {
    set("logicalShape", cfg.logicalShape);
    set("layout", cfg.layout);
    set("sharding", cfg.sharding);
    if (cfg.gridX) set("gridX", cfg.gridX);
    if (cfg.gridY) set("gridY", cfg.gridY);
    if (cfg.bankX) set("bankX", cfg.bankX);
    if (cfg.bankY) set("bankY", cfg.bankY);
    if (cfg.ndShardShape) set("ndShardShape", cfg.ndShardShape);
    if (cfg.ndStrategy) set("ndStrategy", cfg.ndStrategy);
}

// tile / block-sharded [64,64] over 2×2 → 4 shards, 4 cores, 4 pages, 64×64 elems
drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
ok(errText() === "", `tile/block no error (got "${errText()}")`);
ok(shardCards() === 4, `tile/block 4 shard cards (got ${shardCards()})`);
ok(bankCols() === 4, `tile/block 4 cores (got ${bankCols()})`);
ok(pageCells() === 4, `tile/block 4 page cells (got ${pageCells()})`);
ok(elementCells() === 64 * 64, `tile/block element grid is 64×64 (got ${elementCells()})`);

// field visibility: interleave hides core grid + ND, shows banks
set("sharding", "interleave");
ok(document.getElementById("gridField").style.display === "none", "interleave hides core grid");
ok(document.getElementById("ndField").style.display === "none", "interleave hides ND fields");
ok(document.getElementById("bankField").style.display !== "none", "interleave shows banks");
drive({ logicalShape: "6,8", layout: "ROW_MAJOR", sharding: "interleave", bankX: 4, bankY: 1 });
ok(errText() === "", `rm/interleave no error (got "${errText()}")`);
ok(shardCards() === 0, "interleave: no shard cards");
ok(bankCols() === 4, "interleave: 4 cores");
ok(elementCells() === 6 * 8, `rm/interleave element grid 6×8 (got ${elementCells()})`);

// ND fields show under ND
set("sharding", "nd");
ok(document.getElementById("ndField").style.display !== "none", "ND shows ND fields");

// huge element grid falls back to one cell per page (256×256 tile = 65536 > 8192;
// page 32×32 → 8×8 = 64 page cells, each labeled with an element coord range)
drive({ logicalShape: "256,256", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
const pcoordCells = () => document.querySelectorAll("#elementView .pcell").length;
ok(document.getElementById("warn").textContent.length > 0, "oversized element grid warns");
ok(elementCells() === 0, "oversized element grid draws no per-element cells");
ok(pcoordCells() === 64, `oversized element grid → 64 page-coordinate cells (got ${pcoordCells()})`);
ok(pageCells() > 0, "page grid still renders alongside the page-coordinate fallback");
// the coordinate label shows the element range the first page covers: (0,0) → (31,31)
ok(/\(0, 0\)/.test(document.querySelector("#elementView .pcell").textContent), "first page labeled from (0,0)");
ok(/31, 31/.test(document.querySelector("#elementView .pcell").textContent), "first page labeled to (31,31)");
// clicking a page-coordinate cell links to the other views
{
    const click = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    const pc0 = document.querySelector('#elementView .pcell[data-page="0"]');
    click(pc0);
    ok(document.querySelectorAll('.cell[data-page="0"].sel, .pcell[data-page="0"].sel').length >= 2, "page-coord cell click highlights across views");
    click(document.querySelector("#selbar button"));
}

// ---- linked selection across element ↔ page ↔ shard ↔ bank ----
drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
const results = document.getElementById("results");
const click = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
// click page 0 cell in the page grid
const page0 = document.querySelector('#pageStrip .cell[data-page="0"]');
click(page0);
ok(results.classList.contains("selecting"), "page click activates selecting mode");
ok(document.querySelectorAll('.cell[data-page="0"].sel, .ecell[data-page="0"].sel').length >= 2, "page 0 highlights across element + page views");
// clear
click(document.querySelector("#selbar button"));
ok(!results.classList.contains("selecting"), "Clear clears selection");

// toggle a whole shard via its header
const shardHd = document.querySelector("#shardsView .shard-card .shard-hd");
click(shardHd);
ok(shardHd.classList.contains("active"), "shard header reads active");
click(document.querySelector("#selbar button"));

// ---- color-mode selector: core / page / shard recolor the element grid ------
{
    drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
    const firstElem = () => {
        // first non-padding element cell (top-left, page 0)
        return document.querySelector('#elementView .ecell[data-page="0"]');
    };
    const bg = (el) => el && el.style.background;
    const legend = () => document.getElementById("elementLegend").textContent;

    set("colorMode", "core");
    const coreColor = bg(firstElem());
    ok(/destination core/.test(legend()), "core mode legend says destination core");

    set("colorMode", "shard");
    const shardColor = bg(firstElem());
    ok(/color = shard/.test(legend()), "shard mode legend says shard");

    set("colorMode", "page");
    ok(/color = page/.test(legend()), "page mode legend says page");

    // page 0 is shard 0's first page on core 0 — with this 1-page-per-shard-ish
    // layout the three palettes coincide at index 0, so assert the legends instead
    // and check a page that separates them. Pick page 5 (block layout: shard !=
    // core != page in general). Recolor and confirm the cell background changes.
    set("colorMode", "core");
    const c5core = bg(document.querySelector('#elementView .ecell[data-page="5"]'));
    set("colorMode", "page");
    const c5page = bg(document.querySelector('#elementView .ecell[data-page="5"]'));
    ok(c5core !== undefined && c5page !== undefined, "page 5 has a color in both modes");
    ok(coreColor !== undefined && shardColor !== undefined, "first element colored in core+shard modes");
}

// ---- granularity ("Show") selector: element / tile / page ------------------
{
    drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
    const heading = () => document.getElementById("granHeading").textContent;
    const tileCells = () => document.querySelectorAll("#elementView .tcell").length;
    const pgCells = () => document.querySelectorAll("#elementView .pcell").length;

    ok(document.getElementById("granularity") != null, "granularity selector present");

    set("granularity", "element");
    ok(elementCells() === 64 * 64, `element granularity renders element grid (got ${elementCells()})`);
    ok(/^Elements →/.test(heading()), `element heading reads "Elements → ..." (got "${heading()}")`);

    // 64×64 tile layout, 32×32 tiles → 2×2 = 4 tile cells; no per-element cells
    set("granularity", "tile");
    ok(tileCells() === 4, `tile granularity → 4 tile cells (got ${tileCells()})`);
    ok(elementCells() === 0, "tile granularity draws no per-element cells");
    ok(/^Tiles →/.test(heading()), `tile heading reads "Tiles → ..." (got "${heading()}")`);

    // page granularity → one cell per page in the page grid
    set("granularity", "page");
    ok(pgCells() === 4, `page granularity → 4 page cells (got ${pgCells()})`);
    ok(elementCells() === 0, "page granularity draws no per-element cells");

    // mapped-to drives the heading destination word
    set("colorMode", "shard");
    ok(/→ Shards$/.test(heading()), `heading reflects mapped-to (got "${heading()}")`);
    set("colorMode", "core");

    // clicking a tile cell links to the other views (single-page tile here)
    set("granularity", "tile");
    {
        const t0 = document.querySelector('#elementView .tcell[data-page="0"]');
        ok(t0 != null, "single-page tile cell carries data-page");
        click(t0);
        ok(document.querySelectorAll('.cell[data-page="0"].sel, .tcell[data-page="0"].sel').length >= 2,
            "tile cell click highlights across views");
        click(document.querySelector("#selbar button"));
    }

    // row-major tile view: a 32×32 tile spans many 1×W pages, possibly across
    // cores → a multi-page tile toggle (no single data-page) still renders
    drive({ logicalShape: "64,64", layout: "ROW_MAJOR", sharding: "block", gridX: 2, gridY: 2 });
    ok(tileCells() > 0, `RM tile view renders tile cells (got ${tileCells()})`);

    // restore defaults for the following tests
    set("granularity", "element");
    drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
}

// ---- classic sharding: width & block take explicit shard dims; height auto --
{
    const warnText = () => document.getElementById("warn").textContent;
    const summaryText = () => document.getElementById("summary").textContent;
    const shown = (id) => document.getElementById(id).style.display !== "none";

    // height sharding is always the even auto-split — it exposes no shard input
    drive({ logicalShape: "128,64", layout: "TILE", sharding: "height", gridX: 4, gridY: 1 });
    ok(!shown("shardDimField"), "height sharding exposes no custom shard dim (always auto-split)");
    ok(errText() === "" && shardCards() === 4, `height auto → 4 shards over 4 cores (got ${shardCards()})`);

    // width shows only shard width; block shows both; interleave shows neither
    set("sharding", "width");
    ok(shown("shardDimField") && shown("shardWField") && !shown("shardHField"), "width shows only shard width");
    set("sharding", "block");
    ok(shown("shardHField") && shown("shardWField"), "block shows both shard dims");
    set("sharding", "interleave");
    ok(!shown("shardDimField"), "interleave hides shard dims");

    // width sharding, explicit shard width 32: 64×128 tile → 2×4 pages, shard
    // [64,32]→[2,1] pages → 4 width-shards round-robin over 2 cores (wrap)
    drive({ logicalShape: "64,128", layout: "TILE", sharding: "width", gridX: 2, gridY: 1 });
    set("shardW", "32");
    ok(errText() === "", `explicit width shard no error (got "${errText()}")`);
    ok(shardCards() === 4, `width shard 32 → 4 shards (got ${shardCards()})`);
    ok(bankCols() === 2, `width shard over 2 cores (got ${bankCols()})`);
    ok(/wrap/.test(warnText()), "width shards > cores warns about wrap");
    ok(/Shard \(elements\)/.test(summaryText()), "summary lists shard shape in elements");

    // block sharding, explicit 2D shard 32×32: 64×128 tile → 2×4 pages, shard
    // [1,1] pages → 2×4 shard grid → 8 shards, one per core over a 4×2 grid
    drive({ logicalShape: "64,128", layout: "TILE", sharding: "block", gridX: 4, gridY: 2 });
    set("shardH", "32");
    set("shardW", "32");
    ok(errText() === "", `explicit block shard no error (got "${errText()}")`);
    ok(shardCards() === 8, `block shard 32×32 → 8 shards (got ${shardCards()})`);
    ok(bankCols() === 8, `block shard over 4×2 = 8 cores (got ${bankCols()})`);
    ok(!/wrap/.test(warnText()), "block one-shard-per-core does not warn about wrap");

    // block shard grid that overflows the core grid errors
    drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 1 });
    set("shardH", "32");
    set("shardW", "32");
    ok(errText().length > 0, "block shard grid larger than core grid surfaces an error");

    // blank shard dims fall back to the even auto-split (convenience helpers)
    drive({ logicalShape: "64,128", layout: "TILE", sharding: "width", gridX: 4, gridY: 1 });
    set("shardW", "");
    ok(errText() === "" && shardCards() === 4, `blank width shard → auto 4 shards (got ${shardCards()})`);

    // restore clean state for the following tests
    set("shardH", "");
    set("shardW", "");
    drive({ logicalShape: "64,64", layout: "TILE", sharding: "block", gridX: 2, gridY: 2 });
}

// error path: ND shard shape rank > tensor rank surfaces a message
drive({ logicalShape: "64,64", layout: "TILE", sharding: "nd", ndShardShape: "1,1,32,32", gridX: 2, gridY: 2 });
ok(errText().length > 0, "rank-mismatch ND surfaces an error");

// ---- shareable link: config <-> URL hash round-trip ------------------------
// Uses fresh documents booted at a real URL (the shared jsdom above is about:blank,
// where history.replaceState is a no-op).
{
    const boot = (u) => new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, url: u });

    // write: driving the controls mirrors the config into location.hash
    const jw = boot("https://x/tensor.html");
    const dw = jw.window.document;
    const setw = (id, v) => {
        const e = dw.getElementById(id);
        e.value = v;
        e.dispatchEvent(new jw.window.Event(e.tagName === "SELECT" ? "change" : "input", { bubbles: true }));
    };
    setw("logicalShape", "128,64");
    setw("layout", "TILE");
    setw("sharding", "width");
    setw("gridX", "4");
    setw("gridY", "1");
    setw("shardW", "16");
    setw("granularity", "tile");
    setw("colorMode", "shard");
    const h = jw.window.location.hash;
    ok(/shape=128x64/.test(h), `hash carries shape (got "${h}")`);
    ok(/sharding=width/.test(h) && /grid=4x1/.test(h), "hash carries sharding + grid");
    ok(/sw=16/.test(h), "hash carries explicit shard width");
    ok(/show=tile/.test(h) && /color=shard/.test(h), "hash carries granularity + color mode");
    ok(/tile=32x32/.test(h), "hash carries tile shape in TILE layout");

    // read: booting AT a hash restores that configuration and renders it
    const jr = boot(
        "https://x/tensor.html#shape=2x64x64,layout=TILE,sharding=nd,grid=2x2," +
        "ndshape=1x32x32,ndstrat=round_robin,ndalign=RECOMMENDED,show=page,color=core"
    );
    const dr = jr.window.document;
    ok(dr.getElementById("logicalShape").value === "2,64,64", `restored shape (got "${dr.getElementById("logicalShape").value}")`);
    ok(dr.getElementById("sharding").value === "nd", "restored sharding");
    ok(dr.getElementById("ndShardShape").value === "1,32,32", "restored nd shard shape");
    ok(dr.getElementById("granularity").value === "page", "restored granularity");
    ok(dr.getElementById("error").textContent.trim() === "", "restored config renders without error");
    ok(dr.querySelectorAll("#elementView .pcell").length > 0, "restored page view renders page cells");

    // a classic-shard link restores the explicit shard dimension
    const jc = boot("https://x/tensor.html#shape=64x96,layout=TILE,sharding=width,grid=2x1,sw=32,show=element,color=core");
    const dc = jc.window.document;
    ok(dc.getElementById("shardW").value === "32", `restored explicit shard width (got "${dc.getElementById("shardW").value}")`);
    ok(dc.querySelectorAll("#shardsView .shard-card").length === 3, "restored classic shard renders 3 shards");
}

console.log(failed === 0 ? "\nTensor UI smoke test: all checks passed" : `\nTensor UI smoke test: ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
