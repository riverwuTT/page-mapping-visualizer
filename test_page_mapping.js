// SPDX-License-Identifier: Apache-2.0
// Node tests for the two-compartment page-mapping model.
// Run: node test_page_mapping.js

const PM = require("./page_mapping.js");

let passed = 0;
let failed = 0;
function eq(actual, expected, msg) {
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    if (a === e) passed++;
    else {
        failed++;
        console.error(`FAIL: ${msg}\n   expected: ${e}\n   actual:   ${a}`);
    }
}
function ok(cond, msg) {
    if (cond) passed++;
    else {
        failed++;
        console.error(`FAIL: ${msg}`);
    }
}

const shardPages = (res) => res.shards.map((s) => s.pages);
const bankPages = (res) => res.banks.map((b) => b.devicePages.map((d) => d.pageId));

// Every real page is placed exactly once, and pageLookup agrees with the banks.
function checkCoverage(res, label) {
    const seen = new Array(res.numPages).fill(0);
    for (const b of res.banks) {
        for (const d of b.devicePages) {
            if (d.pageId != null) {
                ok(d.pageId >= 0 && d.pageId < res.numPages, `${label}: page id in range`);
                seen[d.pageId]++;
            }
        }
    }
    ok(seen.every((c) => c === 1), `${label}: every page placed exactly once`);
    for (let p = 0; p < res.numPages; p++) {
        const lk = res.pageLookup[p];
        ok(lk != null, `${label}: pageLookup[${p}] present`);
        if (lk) {
            const cell = res.banks[lk.bankId].devicePages[lk.devicePage];
            eq(cell.pageId, p, `${label}: pageLookup[${p}] back-reference`);
        }
    }
}

// ---- Interleaved == shard volume 1 + round_robin ----------------------------
{
    const res = PM.computeMapping({
        pageGrid: [20],
        shardShape: [1],
        bankGrid: { x: 6, y: 1 },
        distribution: "round_robin",
    });
    eq(res.numShards, 20, "interleaved: 20 single-page shards");
    eq(bankPages(res), [[0, 6, 12, 18], [1, 7, 13, 19], [2, 8, 14], [3, 9, 15], [4, 10, 16], [5, 11, 17]],
        "interleaved 20/6 bank layout");
    eq(res.pageLookup[7], { bankId: 1, devicePage: 1, shardId: 7, shardLocalOffset: 0 }, "interleaved page 7");
    checkCoverage(res, "interleaved");
}

// ---- Interleaved is its own layout (page p -> bank p%B, slot p/B) -----------
{
    const res = PM.computeMapping({
        pageGrid: [20],
        shardShape: [1],
        bankGrid: { x: 6, y: 1 },
        distribution: "interleaved",
    });
    eq(res.distribution, "interleaved", "distribution is interleaved (not nd)");
    eq(res.numShards, 20, "interleaved: each page its own unit");
    eq(res.shardVolume, 1, "interleaved shard volume 1");
    eq(bankPages(res), [[0, 6, 12, 18], [1, 7, 13, 19], [2, 8, 14], [3, 9, 15], [4, 10, 16], [5, 11, 17]],
        "interleaved round-robins pages across banks");
    eq(res.pageLookup[13], { bankId: 1, devicePage: 2, shardId: 13, shardLocalOffset: 0 }, "page 13 -> bank1 slot2");
    checkCoverage(res, "interleaved");

    // standalone, and 2D page grid still flattens row-major
    const i2 = PM.computeInterleaved({ pageGrid: [4, 4], bankGrid: { x: 4, y: 1 }, orientation: "row_major" });
    eq(i2.banks[0].devicePages.map((d) => d.pageId), [0, 4, 8, 12], "interleaved 2D grid, bank0 = every 4th page");
}

// ---- Height: shard spans full width, round_robin ----------------------------
{
    const res = PM.computeMapping({
        pageGrid: [4, 2],
        shardShape: [2, 2],
        bankGrid: { x: 1, y: 2 },
        distribution: "round_robin",
    });
    eq(shardPages(res), [[0, 1, 2, 3], [4, 5, 6, 7]], "height shards");
    eq(bankPages(res), [[0, 1, 2, 3], [4, 5, 6, 7]], "height bank layout");
    checkCoverage(res, "height");
}

// ---- Block: 2D shard grid -> 2D bank grid -----------------------------------
{
    const res = PM.computeMapping({
        pageGrid: [4, 4],
        shardShape: [2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "grid_2d",
        orientation: "row_major",
    });
    eq(shardPages(res), [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], "block shards");
    eq(bankPages(res), [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], "block bank layout");
    eq(res.banks[3].gridCoord, { x: 1, y: 1 }, "block bank 3 coord");
    checkCoverage(res, "block");
}

// ---- Block with partial shards ----------------------------------------------
{
    const res = PM.computeMapping({
        pageGrid: [3, 3],
        shardShape: [2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "grid_2d",
    });
    eq(
        shardPages(res),
        [[0, 1, 3, 4], [2, null, 5, null], [6, 7, null, null], [8, null, null, null]],
        "block partial shards"
    );
    checkCoverage(res, "block partial");
}

// ---- Width: 1 x N shard grid ------------------------------------------------
{
    const res = PM.computeMapping({
        pageGrid: [2, 6],
        shardShape: [2, 2],
        bankGrid: { x: 3, y: 1 },
        distribution: "grid_2d",
    });
    eq(shardPages(res), [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]], "width shards");
    eq(bankPages(res), [[0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11]], "width bank layout");
    checkCoverage(res, "width");
}

// ---- ND: more shards than banks -> round_robin stacking ---------------------
{
    const res = PM.computeMapping({
        pageGrid: [4, 4],
        shardShape: [2, 2],
        bankGrid: { x: 1, y: 2 },
        distribution: "round_robin",
    });
    eq(res.numShards, 4, "nd numShards");
    eq(bankPages(res), [[0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]], "nd stacked bank layout");
    eq(res.banks[0].shardIds, [0, 2], "nd bank0 stacked shards");
    // device page 4 of bank0 is the first page of its 2nd shard (shard 2)
    eq(res.banks[0].devicePages[4].shardId, 2, "nd stacked shardId");
    checkCoverage(res, "nd");
}

// ---- ND rank-3 --------------------------------------------------------------
{
    const res = PM.computeMapping({
        pageGrid: [2, 4, 4],
        shardShape: [1, 2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "round_robin",
    });
    eq(res.numShards, 8, "nd3 numShards");
    eq(res.numPages, 32, "nd3 pages");
    eq(res.shards[0].pages, [0, 1, 4, 5], "nd3 first shard");
    checkCoverage(res, "nd rank3");
}

// ---- shard rank < page-grid rank: squeeze folds the extra page dims ---------
{
    // page [2,4,4] with a 2D shard [2,2] squeezes to page [8,4] / shard [2,2].
    const res = PM.computeMapping({
        pageGrid: [2, 4, 4],
        shardShape: [2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "round_robin",
    });
    ok(res.squeezed, "mismatched rank flagged as squeezed");
    eq(res.squeezedPageGrid, [8, 4], "page grid squeezed to [8,4]");
    eq(res.shardShape, [2, 2], "effective shard shape [2,2]");
    eq(res.pageGrid, [2, 4, 4], "original page grid kept for display");
    eq(res.numShards, 8, "8 shards");
    // identical partition to the equal-rank [2,4,4]/[1,2,2] case
    eq(res.shards[0].pages, [0, 1, 4, 5], "first shard pages match the folded layout");
    checkCoverage(res, "rank-mismatch squeeze");

    // shards always display the SUPPLIED shard shape, never the squeezed one
    const sup = PM.computeMapping({
        pageGrid: [2, 3, 4],
        shardShape: [1, 4],
        bankGrid: { x: 6, y: 1 },
        distribution: "round_robin",
    });
    eq(sup.shards[0].shape, [1, 4], "ND shard keeps supplied shape (not squeezed [4])");
    eq(sup.shards[0].pages, [0, 1, 2, 3], "...with the same row-major page assignment");
    checkCoverage(sup, "supplied-shape shards");

    // standalone: ranks need not match
    const c1 = PM.pagesToShards([2, 4, 4], [2, 2]);
    eq(c1.shardGrid, [4, 2], "squeezed shardGrid [4,2]");
    eq(c1.shards.length, 8, "8 shards standalone");
    eq(c1.shards[0].shape, [2, 2], "standalone shard keeps supplied shape");

    // page rank < shard rank is still an error
    let threw = false;
    try {
        PM.pagesToShards([2, 2], [2, 2, 2]);
    } catch (e) {
        threw = /can't be less than/.test(e.message);
    }
    ok(threw, "page rank < shard rank throws");
}

// ---- legacy sharding: distinct emplacement from ND -------------------------
{
    // legacy HEIGHT chunks the flat page stream contiguously...
    const lh = PM.computeMapping({
        pageGrid: [4, 4],
        shardShape: [2, 2],
        bankGrid: { x: 4, y: 1 },
        distribution: "legacy",
        legacyLayout: "height",
    });
    eq(bankPages(lh), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], "legacy height = contiguous chunks");
    checkCoverage(lh, "legacy height");

    // ...whereas ND / legacy BLOCK gather 2D sub-blocks — different result.
    const lb = PM.computeMapping({
        pageGrid: [4, 4],
        shardShape: [2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "legacy",
        legacyLayout: "block",
    });
    eq(bankPages(lb), [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], "legacy block = 2D sub-blocks");
    const nd = PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 2, y: 2 }, distribution: "grid_2d" });
    eq(bankPages(lb), bankPages(nd), "legacy block content matches ND block");
    ok(JSON.stringify(bankPages(lh)) !== JSON.stringify(bankPages(lb)), "legacy height differs from legacy block");
    checkCoverage(lb, "legacy block");

    // legacy partial shards keep the 2D padding pattern
    const lp = PM.computeMapping({
        pageGrid: [3, 3],
        shardShape: [2, 2],
        bankGrid: { x: 2, y: 2 },
        distribution: "legacy",
        legacyLayout: "block",
    });
    eq(shardPages(lp), [[0, 1, 3, 4], [2, null, 5, null], [6, 7, null, null], [8, null, null, null]], "legacy block partial padding");

    // legacy is one shard per bank: too few banks throws
    let threw = false;
    try {
        PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 2, y: 1 }, distribution: "legacy", legacyLayout: "block" });
    } catch (e) {
        threw = /one shard per bank/.test(e.message);
    }
    ok(threw, "legacy with too few banks throws");

    // grid sharding still requires rank-2 shapes
    threw = false;
    try {
        PM.computeMapping({ pageGrid: [16], shardShape: [4], bankGrid: { x: 4, y: 1 }, distribution: "legacy", legacyLayout: "block" });
    } catch (e) {
        threw = /rank-2/.test(e.message);
    }
    ok(threw, "grid sharding rejects 1D shapes");
}

// ---- continuous fill (height) supports 1D — a shard is just its volume ------
{
    // 1D page grid + 1D shard: contiguous chunks of shard-volume pages.
    const r = PM.computeMapping({
        pageGrid: [16],
        shardShape: [4],
        bankGrid: { x: 4, y: 1 },
        distribution: "legacy",
        legacyLayout: "height",
    });
    eq(r.numShards, 4, "1D continuous: 4 shards");
    eq(r.shards[0].shape, [4], "1D shard keeps its 1D shape");
    eq(bankPages(r), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], "1D continuous contiguous chunks");
    checkCoverage(r, "continuous 1D");

    // 1D with a partial last shard -> tail padding
    const rp = PM.computeMapping({
        pageGrid: [10],
        shardShape: [4],
        bankGrid: { x: 3, y: 1 },
        distribution: "legacy",
        legacyLayout: "height",
    });
    eq(rp.numShards, 3, "1D partial: 3 shards");
    eq(shardPages(rp), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, null, null]], "1D partial last shard tail-padded");
    checkCoverage(rp, "continuous 1D partial");

    // a shard is seen only as its volume: it can even differ in rank from the grid
    const rmixed = PM.computeMapping({
        pageGrid: [16],
        shardShape: [2, 2],
        bankGrid: { x: 4, y: 1 },
        distribution: "legacy",
        legacyLayout: "height",
    });
    eq(rmixed.numShards, 4, "volume-only: 16 pages / vol-4 shard = 4 shards");
    eq(rmixed.shards[0].pages, [0, 1, 2, 3], "volume-only contiguous fill");
    checkCoverage(rmixed, "continuous volume-only");

    // 2D continuous still matches the prior behavior
    const r2 = PM.computeMapping({
        pageGrid: [8, 2],
        shardShape: [2, 2],
        bankGrid: { x: 4, y: 1 },
        distribution: "legacy",
        legacyLayout: "height",
    });
    eq(bankPages(r2), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], "2D continuous unchanged");
}

// ---- slots allocated = max slots per core * core-grid volume ----------------
{
    // grid, one shard per core, no padding: 4*4 = 16 = numPages
    const g = PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 2, y: 2 }, distribution: "grid_2d" });
    eq(g.slotsAllocated, 16, "grid slots = shardVolume * cores");
    eq(g.maxSlotsPerCore, 4, "grid max slots/core");

    // continuous 1D partial: 3 cores * 4 = 12 (2 padding slots over 10 pages)
    const c = PM.computeMapping({ pageGrid: [10], shardShape: [4], bankGrid: { x: 3, y: 1 }, distribution: "legacy", legacyLayout: "height" });
    eq(c.slotsAllocated, 12, "continuous slots include padding");

    // ND stacking: 4 shards over 2 cores -> 2 shards/core * vol 4 = 8/core * 2 = 16
    const nd = PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 1, y: 2 }, distribution: "round_robin" });
    eq(nd.maxSlotsPerCore, 8, "ND stacked max slots/core");
    eq(nd.slotsAllocated, 16, "ND stacked slots = maxPerCore * cores");

    // interleave: ceil(20/6)=4 per bank * 6 = 24
    const il = PM.computeMapping({ pageGrid: [20], shardShape: [1], bankGrid: { x: 6, y: 1 }, distribution: "interleaved" });
    eq(il.slotsAllocated, 24, "interleave slots = ceil(pages/banks) * banks");

    // empty cores still allocate (lock-step over the whole core grid)
    const e = PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 3, y: 3 }, distribution: "grid_2d" });
    eq(e.slotsAllocated, 36, "empty cores count toward the core-grid volume (4 * 9)");
}

// ---- compartment functions usable standalone --------------------------------
{
    const c1 = PM.pagesToShards([4, 4], [2, 2]);
    eq(c1.numShards, 4, "pagesToShards numShards");
    eq(c1.shardGrid, [2, 2], "pagesToShards shardGrid");
    const c2 = PM.shardsToBanks(c1, { x: 4, y: 1 }, "round_robin", "row_major");
    eq(c2.banks.length, 4, "shardsToBanks bank count");
}

// ---- error paths ------------------------------------------------------------
{
    let threw = false;
    try {
        PM.computeMapping({ pageGrid: [4, 4], shardShape: [2, 2], bankGrid: { x: 1, y: 1 }, distribution: "grid_2d" });
    } catch (e) {
        threw = /doesn't fit/.test(e.message);
    }
    ok(threw, "grid_2d undersized bank grid throws");

    threw = false;
    try {
        PM.computeMapping({ pageGrid: [4, 4, 4], shardShape: [2, 2, 2], bankGrid: { x: 2, y: 2 }, distribution: "grid_2d" });
    } catch (e) {
        threw = /rank-2/.test(e.message);
    }
    ok(threw, "grid_2d with rank-3 throws");
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
