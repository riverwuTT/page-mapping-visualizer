// SPDX-License-Identifier: Apache-2.0
// End-to-end smoke test: boots the built single-file page_mapping_viz.html in
// jsdom, drives the controls, and asserts both compartments render.
// Run: node test_ui.js

const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const html = fs.readFileSync(path.join(__dirname, "page_mapping_viz.html"), "utf8");
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
const shardCards = () => document.querySelectorAll("#shardsView .shard-card").length;
const bankCols = () => document.querySelectorAll("#banksView .bank").length;
const pageCells = () => document.querySelectorAll("#pageStrip .cell").length;

const PM = dom.window.PageMapping;
ok(PM && typeof PM.computeMapping === "function", "PageMapping global exposed in the page");

// default boot
ok(errText() === "", `default boot has no error (got "${errText()}")`);
ok(shardCards() > 0 && bankCols() > 0, "default boot renders both compartments");

function drive({ pageGrid, shardShape, bankX, bankY, distribution, legacyLayout, orientation }) {
    set("pageGrid", pageGrid);
    set("shardShape", shardShape);
    set("bankX", bankX);
    set("bankY", bankY);
    if (distribution === "legacy") {
        set("shardingFamily", "legacy");
        set("legacyLayout", legacyLayout || "block");
    } else {
        set("shardingFamily", "nd");
        set("distribution", distribution);
    }
    set("orientation", orientation);
}

const cases = [
    ["interleaved", { pageGrid: "20", shardShape: "1", bankX: 6, bankY: 1, distribution: "round_robin", orientation: "row_major" }, 20, 20, 6],
    ["height", { pageGrid: "8,2", shardShape: "2,2", bankX: 1, bankY: 4, distribution: "round_robin", orientation: "row_major" }, 4, 16, 4],
    ["block", { pageGrid: "4,4", shardShape: "2,2", bankX: 2, bankY: 2, distribution: "grid_2d", orientation: "row_major" }, 4, 16, 4],
    ["nd3", { pageGrid: "2,4,4", shardShape: "1,2,2", bankX: 2, bankY: 2, distribution: "round_robin", orientation: "row_major" }, 8, 32, 4],
];
for (const [name, cfg, nShards, nPages, nBanks] of cases) {
    drive(cfg);
    ok(errText() === "", `${name}: no error (got "${errText()}")`);
    ok(shardCards() === nShards, `${name}: ${nShards} shard cards (got ${shardCards()})`);
    ok(pageCells() === nPages, `${name}: ${nPages} page cells (got ${pageCells()})`);
    ok(bankCols() === nBanks, `${name}: ${nBanks} bank columns (got ${bankCols()})`);
}

// ---- legacy distribution + layout sub-selector -----------------------------
{
    const field = document.getElementById("legacyLayoutField");
    const ndField = document.getElementById("ndDistField");
    // legacy layout hidden under ND; ND distribution hidden under legacy
    set("shardingFamily", "nd");
    ok(field.style.display === "none", "legacy layout field hidden for ND");
    ok(ndField.style.display !== "none", "ND distribution field shown for ND");
    set("pageGrid", "4,4");
    set("shardShape", "2,2");
    set("bankX", 4);
    set("bankY", 1);
    set("shardingFamily", "legacy");
    ok(field.style.display !== "none", "legacy layout field shown for legacy");
    ok(ndField.style.display === "none", "ND distribution field hidden for legacy");

    set("legacyLayout", "height");
    ok(errText() === "", `legacy height no error (got "${errText()}")`);
    // contiguous chunking: bank 0 holds pages 0..3 in device order
    const bank0 = [...document.querySelectorAll("#banksView .bank")][0];
    const b0pages = [...bank0.querySelectorAll(".cell[data-page]")].map((c) => +c.dataset.page);
    ok(JSON.stringify(b0pages) === JSON.stringify([0, 1, 2, 3]), `legacy height bank0 contiguous (got ${b0pages})`);
}

// ---- interleaved is its own family ------------------------------------------
{
    set("shardingFamily", "interleaved");
    set("pageGrid", "20");
    set("bankX", 6);
    set("bankY", 1);
    ok(document.getElementById("shardShapeField").style.display === "none", "interleaved hides shard shape");
    ok(document.getElementById("ndDistField").style.display === "none", "interleaved hides ND distribution");
    ok(document.getElementById("legacyLayoutField").style.display === "none", "interleaved hides legacy layout");
    ok(errText() === "", `interleaved no error (got "${errText()}")`);
    // shards view is a note, not 20 cards
    ok(document.querySelectorAll("#shardsView .shard-card").length === 0, "interleaved: no shard cards");
    ok(document.querySelectorAll("#banksView .bank").length === 6, "interleaved: 6 bank columns");
    const bank0 = [...document.querySelectorAll("#banksView .bank")][0];
    const b0 = [...bank0.querySelectorAll(".cell[data-page]")].map((c) => +c.dataset.page);
    ok(JSON.stringify(b0) === JSON.stringify([0, 6, 12, 18]), `interleaved bank0 = round-robin (got ${b0})`);
}

// preset buttons exist and are wired
ok(document.querySelectorAll("#presets button").length >= 5, "preset buttons rendered");
document.querySelectorAll("#presets button")[0].click(); // Interleaved
ok(errText() === "", "preset click has no error");

// a legacy preset switches the sharding-model selector over to legacy
{
    const btns = [...document.querySelectorAll("#presets button")];
    const legacyBtn = btns.find((b) => /legacy/i.test(b.textContent));
    ok(legacyBtn, "a legacy preset button exists");
    legacyBtn.click();
    ok(document.getElementById("shardingFamily").value === "legacy", "legacy preset switches family to legacy");
    ok(document.getElementById("legacyLayoutField").style.display !== "none", "legacy preset reveals legacy layout");
    // and an ND preset switches back
    btns.find((b) => /block|grid/i.test(b.textContent)).click();
    ok(document.getElementById("shardingFamily").value === "nd", "ND preset switches family back to nd");
}

// ---- click-to-toggle (linked selection) ------------------------------------
{
    drive({ pageGrid: "4,4", shardShape: "2,2", bankX: 2, bankY: 2, distribution: "grid_2d", orientation: "row_major" });
    const results = document.getElementById("results");
    const click = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    const selCount = (p) => document.querySelectorAll(`.cell[data-page="${p}"].sel`).length;

    // shards laid out in the shard grid (2x2 here)
    ok(document.querySelectorAll("#shardsView .shard-card").length === 4, "4 shard cards");
    const shard0Hd = document.querySelector("#shardsView .shard-card .shard-hd");

    // toggle whole shard 0 (pages 0,1,4,5) via its header
    click(shard0Hd);
    ok(results.classList.contains("selecting"), "shard toggle activates selecting mode");
    ok(shard0Hd.classList.contains("active"), "shard header reads active");
    [0, 1, 4, 5].forEach((p) => ok(selCount(p) >= 2, `page ${p} highlighted across views`));
    ok(selCount(2) === 0, "page 2 (other shard) not selected");

    // toggling the same shard header again clears it
    click(shard0Hd);
    ok(!results.classList.contains("selecting"), "re-clicking shard header toggles it off");

    // toggle a single page cell
    const cell9 = document.querySelector('#shardsView .cell[data-page="9"]');
    click(cell9);
    ok(selCount(9) >= 2, "single page 9 toggled on across views");
    click(cell9);
    ok(selCount(9) === 0, "single page 9 toggled off");

    // toggle a whole bank via its header, then Clear button
    const bankHd = document.querySelector("#banksView .bank .bank-hd");
    click(bankHd);
    ok(results.classList.contains("selecting"), "bank toggle activates selecting mode");
    ok(bankHd.classList.contains("active"), "bank header reads active");
    const clearBtn = document.querySelector("#selbar button");
    ok(clearBtn && clearBtn.textContent === "Clear", "Clear button present");
    click(clearBtn);
    ok(!results.classList.contains("selecting"), "Clear button clears selection");
}

// ---- experimental 3D cube (rank-3 page grid) -------------------------------
{
    const click = (el) => el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
    drive({ pageGrid: "2,4,4", shardShape: "1,2,2", bankX: 2, bankY: 2, distribution: "round_robin", orientation: "row_major" });
    const cube = document.getElementById("cube3d");

    // off by default -> flat layout, no cube stage
    ok(document.querySelectorAll("#pageStrip .cube-stage").length === 0, "cube off by default");

    cube.checked = true;
    cube.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
    ok(document.querySelectorAll("#pageStrip .cube-stage").length === 1, "cube stage rendered when toggled on");
    ok(document.querySelectorAll("#pageStrip .cube-plane").length === 2, "one plane per z (d0=2)");
    ok(document.querySelectorAll("#pageStrip .cell[data-page]").length === 32, "all 32 pages as cube cells");

    // shards also render as cubes (rank-3 shard [1,2,2] -> 1 plane each, 8 shards)
    ok(document.querySelectorAll("#shardsView .cube-stage").length === 8, "each rank-3 shard is a cube");
    ok(document.querySelectorAll("#shardsView .shard-card .cube-stage .cube-plane").length === 8, "shard cubes have planes");
    // a shard cube cell still highlights via the linked selection
    const sc = document.querySelector('#shardsView .cube-stage .cell[data-page]');
    click(sc);
    ok(document.querySelectorAll(`.cell[data-page="${sc.dataset.page}"].sel`).length >= 2, "shard cube cell click highlights across views");
    click(document.querySelector("#selbar button"));

    // selection still works in the cube: clicking a cell highlights it everywhere
    const cell = document.querySelector('#pageStrip .cell[data-page="0"]');
    click(cell);
    ok(document.querySelectorAll('.cell[data-page="0"].sel').length >= 2, "cube cell click highlights across views");
    click(document.querySelector("#selbar button")); // Clear

    // per-component 3D: rank-3 page grid with a rank-2 shard -> page-grid cube
    // renders, but shards stay flat (each shard is 2D after the squeeze).
    drive({ pageGrid: "2,4,4", shardShape: "2,2", bankX: 2, bankY: 2, distribution: "round_robin", orientation: "row_major" });
    ok(document.querySelectorAll("#pageStrip .cube-stage").length === 1, "rank-3 page grid still cubes");
    ok(document.querySelectorAll("#shardsView .cube-stage").length === 0, "rank-2 shards do NOT cube");
    ok(document.querySelectorAll("#shardsView .shard-card .cells").length > 0, "shards render flat");

    // non-rank-3 grid with cube on -> note shown, falls back to flat
    drive({ pageGrid: "4,4", shardShape: "2,2", bankX: 2, bankY: 2, distribution: "grid_2d", orientation: "row_major" });
    ok(document.getElementById("cubeNote").textContent.length > 0, "cube note shown for non-rank-3");
    ok(document.querySelectorAll("#pageStrip .cube-stage").length === 0, "non-rank-3 falls back to flat");
    cube.checked = false;
    cube.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
}

// error path: grid_2d that doesn't fit the bank grid surfaces a message
drive({ pageGrid: "8,8", shardShape: "2,2", bankX: 1, bankY: 1, distribution: "grid_2d", orientation: "row_major" });
ok(errText().length > 0, "undersized grid_2d surfaces an error");
ok(bankCols() === 0, "error clears the bank view");

console.log(failed === 0 ? "\nUI smoke test: all checks passed" : `\nUI smoke test: ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
