// SPDX-License-Identifier: Apache-2.0
// Unit tests for tensor_mapping.js — the element → page → core model. Mixes
// hand-computed golden cases (traced from the C++) with structural invariants.
// Run: node test_tensor_mapping.js   (no deps)

const T = require("./tensor_mapping.js");

let failed = 0;
let count = 0;
function eq(actual, expected, msg) {
    count++;
    const a = JSON.stringify(actual);
    const e = JSON.stringify(expected);
    if (a !== e) {
        failed++;
        console.error(`FAIL: ${msg}\n   expected ${e}\n   got      ${a}`);
    }
}
function ok(cond, msg) {
    count++;
    if (!cond) {
        failed++;
        console.error(`FAIL: ${msg}`);
    }
}

// ---- low-level shape/layout helpers (traced from tensor_layout.cpp) ----------
eq(T.computePhysicalShape([2, 3, 30, 30], [32, 32]), [192, 32], "physical: 4D tile-aligned");
eq(T.computePhysicalShape([30], [32, 32]), [32, 32], "physical: rank1 with rank2 alignment");
eq(T.computePhysicalShape([2, 3, 30, 30], [1]), [180, 30], "physical: row-major interleaved");
eq(T.computePaddedShape([2, 3, 30, 30], [32, 32]), [2, 3, 32, 32], "padded: 4D tile");
eq(T.computePaddedShape([8, 8], [32]), [8, 32], "padded: row-major width align");
eq(T.computeLogical2dShape([2, 3, 30, 30]), [180, 30], "logical 2d fold");
eq(T.recommendedShardAlignment(T.pageConfig("ROW_MAJOR"), "BFLOAT16"), [32], "rm recommended align bf16");
eq(T.recommendedShardAlignment(T.pageConfig("ROW_MAJOR"), "FLOAT32"), [16], "rm recommended align fp32");

// ---- golden: tile, block-sharded [128,128] over 2×2 -------------------------
{
    const r = T.computeTensorMapping({ logicalShape: [128, 128], layout: "TILE", sharding: "block", grid: { x: 2, y: 2 } });
    eq(r.physical, [128, 128], "A physical");
    eq(r.pageShape, [32, 32], "A page shape = tile");
    eq(r.tensor2dInPages, [4, 4], "A tensor in pages");
    eq(r.shardShapeInPages, [2, 2], "A shard in pages");
    eq(r.ndShardShape, [64, 64], "A nd shard shape (elements)");
    eq(r.memoryLayout, "BLOCK_SHARDED", "A classified block");
    eq(r.distribution, "grid_2d", "A grid_2d distribution");
    eq(r.mapping.numShards, 4, "A 4 shards");
    eq(r.mapping.numBanks, 4, "A 4 cores");
    // shard 0 = top-left 2×2 page sub-block = pages 0,1,4,5; lands on core 0
    eq(r.mapping.banks[0].devicePages.map((d) => d.pageId), [0, 1, 4, 5], "A core0 = top-left block");
    // element (0,0) → page 0 → core 0; element (32,32) → page 5 → core ?
    eq(r.element.pageOf(0, 0), 0, "A element (0,0) → page 0");
    eq(r.element.pageOf(32, 32), 5, "A element (32,32) → page 5");
    eq(r.mapping.pageLookup[0].bankId, 0, "A page0 on core0");
}

// ---- golden: tile, height-sharded [128,64] over 4×1 -------------------------
{
    const r = T.computeTensorMapping({ logicalShape: [128, 64], layout: "TILE", sharding: "height", grid: { x: 4, y: 1 } });
    eq(r.physical, [128, 64], "B physical");
    eq(r.ndShardShape, [32, 64], "B nd shard");
    eq(r.tensor2dInPages, [4, 2], "B tensor in pages");
    eq(r.shardShapeInPages, [1, 2], "B shard in pages");
    eq(r.memoryLayout, "HEIGHT_SHARDED", "B classified height");
    eq(r.distribution, "round_robin", "B round-robin");
    eq(r.mapping.numShards, 4, "B 4 shards");
    eq(r.mapping.banks[0].devicePages.map((d) => d.pageId), [0, 1], "B core0 = first row of pages");
    eq(r.mapping.banks[1].devicePages.map((d) => d.pageId), [2, 3], "B core1 = second row");
}

// ---- golden: row-major, interleave [6,8], 4 banks ---------------------------
{
    const r = T.computeTensorMapping({ logicalShape: [6, 8], layout: "ROW_MAJOR", sharding: "interleave", bankGrid: { x: 4, y: 1 } });
    eq(r.physical, [6, 8], "C physical");
    eq(r.pageShape, [1, 8], "C page shape = one row");
    eq(r.tensor2dInPages, [6, 1], "C tensor in pages (6 row-pages)");
    eq(r.distribution, "interleaved", "C interleaved");
    eq(r.mapping.numBanks, 4, "C 4 banks");
    eq(r.mapping.banks[0].devicePages.map((d) => d.pageId), [0, 4], "C bank0 round-robin");
    eq(r.element.pageOf(2, 5), 2, "C element row 2 → page 2");
    eq(r.element.pageOf(5, 0), 5, "C element row 5 → page 5");
}

// ---- golden: row-major, block-sharded [8,8] over 2×2 (width align kicks in) --
{
    const r = T.computeTensorMapping({ logicalShape: [8, 8], layout: "ROW_MAJOR", sharding: "block", grid: { x: 2, y: 2 } });
    eq(r.alignment, [32], "D alignment = shard width");
    eq(r.physical, [8, 32], "D physical width padded to shard width");
    eq(r.ndShardShape, [4, 32], "D nd shard (width recommended-aligned to 32)");
    eq(r.pageShape, [1, 32], "D page = one shard-wide row");
    eq(r.tensor2dInPages, [8, 1], "D tensor in pages");
    eq(r.shardShapeInPages, [4, 1], "D shard in pages");
    eq(r.memoryLayout, "BLOCK_SHARDED", "D block");
    eq(r.mapping.numShards, 2, "D 2 shards");
    // columns 8..31 are padding (logical width is 8)
    ok(r.element.isPadding(0, 8) && r.element.isPadding(7, 31), "D right columns are padding");
    ok(!r.element.isPadding(7, 7), "D logical corner is real");
}

// ---- golden: ND rank-3, tile [2,64,64], shard [1,32,32], round-robin --------
{
    const r = T.computeTensorMapping({
        logicalShape: [2, 64, 64], layout: "TILE", sharding: "nd",
        ndShardShape: [1, 32, 32], grid: { x: 2, y: 2 }, ndStrategy: "round_robin",
    });
    eq(r.physical, [128, 64], "E physical (2·64 folded into height)");
    eq(r.pageShape, [32, 32], "E tile page");
    eq(r.tensorShapeInPages, [2, 2, 2], "E tensor in pages (rank-3)");
    eq(r.shardShapeInPages, [1, 1, 1], "E shard in pages");
    eq(r.mapping.numShards, 8, "E 8 shards (one page each)");
    eq(r.mapping.numBanks, 4, "E 4 cores");
}

// ---- invariants across a sweep ----------------------------------------------
const SWEEP = [
    { logicalShape: [96, 96], layout: "TILE", sharding: "interleave", bankGrid: { x: 3, y: 1 } },
    { logicalShape: [128, 128], layout: "TILE", sharding: "height", grid: { x: 4, y: 1 } },
    { logicalShape: [128, 128], layout: "TILE", sharding: "width", grid: { x: 4, y: 1 } },
    { logicalShape: [128, 256], layout: "TILE", sharding: "block", grid: { x: 2, y: 2 } },
    { logicalShape: [64, 64], layout: "ROW_MAJOR", sharding: "height", grid: { x: 4, y: 1 } },
    { logicalShape: [3, 64, 64], layout: "TILE", sharding: "nd", ndShardShape: [1, 32, 32], grid: { x: 2, y: 4 }, ndStrategy: "round_robin" },
];
for (const cfg of SWEEP) {
    const tag = `${cfg.layout}/${cfg.sharding}/${cfg.logicalShape}`;
    const r = T.computeTensorMapping(cfg);
    const m = r.mapping;

    // every logical (non-padding) element maps to a page that lands on a real core
    let elementsOk = true;
    for (let rr = 0; rr < r.element.logicalH && elementsOk; rr++) {
        for (let cc = 0; cc < r.element.logicalW; cc++) {
            const pid = r.element.pageOf(rr, cc);
            if (!m.pageLookup[pid]) {
                elementsOk = false;
                break;
            }
        }
    }
    ok(elementsOk, `${tag}: every logical element resolves to a core`);

    // each real page appears exactly once across all cores' device pages
    const seen = new Map();
    for (const b of m.banks) {
        for (const d of b.devicePages) {
            if (d.pageId == null) continue;
            seen.set(d.pageId, (seen.get(d.pageId) || 0) + 1);
        }
    }
    let dupes = 0;
    for (const [, c] of seen) if (c !== 1) dupes++;
    ok(dupes === 0, `${tag}: no page duplicated across cores`);
    eq(seen.size, m.numPages, `${tag}: all ${m.numPages} pages placed exactly once`);

    // page grid volume matches the page count
    eq(r.tensor2dInPages[0] * r.tensor2dInPages[1], m.numPages, `${tag}: page-grid volume = num pages`);
}

console.log(failed === 0 ? `\ntensor_mapping: all ${count} checks passed` : `\ntensor_mapping: ${failed}/${count} failed`);
process.exit(failed === 0 ? 0 : 1);
