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

// ROW_MAJOR default Alignment is the width of the row (page/shard width).
{
    const pc = T.pageConfig("ROW_MAJOR");
    eq(T.createDefaultAlignment(pc, "BFLOAT16", { shardSpec: null, ndShardSpec: null }), [1],
        "rm interleaved default align {1}");
    eq(T.createDefaultAlignment(pc, "BFLOAT16", { shardSpec: { shape: [4, 12] }, ndShardSpec: null }), [12],
        "rm sharded default align = row/shard width");
    eq(T.createDefaultAlignment(pc, "BFLOAT16", { shardSpec: null, ndShardSpec: { shardShape: [4, 12] } }), [12],
        "rm nd-sharded default align = row/shard width");

    const h = T.computeTensorMapping({
        logicalShape: [2, 3, 5], layout: "ROW_MAJOR", sharding: "height", grid: { x: 2, y: 2 },
    });
    eq(h.alignment, [5], "rm height-sharded Alignment = row width");
    eq(h.pageShape, [1, 5], "rm height-sharded page = one full row");
    eq(h.alignment[h.alignment.length - 1], h.pageShape[1], "rm Alignment last dim == page width");
}

// Explicit shard shape is honored for classic height / width / block.
{
    const h = T.computeTensorMapping({
        logicalShape: [2, 3, 5], layout: "TILE", tile: [2, 2], sharding: "height",
        grid: { x: 2, y: 2 }, shardHeight: 4,
    });
    eq(h.ndShardShape, [4, 6], "explicit height shard [4, fullW]");

    const w = T.computeTensorMapping({
        logicalShape: [2, 3, 5], layout: "TILE", tile: [2, 2], sharding: "width",
        grid: { x: 2, y: 2 }, shardWidth: 2,
    });
    eq(w.ndShardShape, [8, 2], "explicit width shard [fullH, 2]");

    const b = T.computeTensorMapping({
        logicalShape: [2, 3, 5], layout: "TILE", tile: [2, 2], sharding: "block",
        grid: { x: 2, y: 2 }, shardHeight: 4, shardWidth: 3,
    });
    eq(b.ndShardShape, [4, 4], "explicit block shard width rounded to tile (3→4)");
    // physical [8,6], shard [4,4] → ceil(8/4)×ceil(6/4) = 2×2 shards
    eq(b.mapping.numShards, 4, "explicit block fits 2×2 core grid");
    eq(b.mapping.shardShape.slice(-2), [2, 2], "explicit block shard in pages");
}

// Classic block on rank-3 uses the folded 2D page grid (post-flatten), so
// GRID_2D never sees an N-D page shape that squeezes down to rank-1.
{
    const r = T.computeTensorMapping({
        logicalShape: [2, 3, 5], layout: "ROW_MAJOR", sharding: "block",
        grid: { x: 2, y: 2 },
    });
    eq(r.physical, [6, 32], "RM rank-3 block physical (width RECOMMENDED-aligned)");
    eq(r.tensor2dInPages, [6, 1], "RM rank-3 block folded 2D pages");
    eq(r.distribution, "grid_2d", "RM rank-3 block is GRID_2D");
    eq(r.mapping.shardGrid.length, 2, "RM rank-3 block shard grid stays rank-2");
    eq(r.mapping.numShards, 2, "RM rank-3 block → 2 shards on 2×2 (width-1 page)");
}

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

// ---- explicit block shard shape is honored (not RECOMMENDED-rounded) --------
// Regression: grid sharding must use the shard width the caller supplies. The
// convenience block_sharded auto-split rounds a ROW_MAJOR width up to the 64B
// recommended alignment (golden D above), but an EXPLICIT shard shape uses
// REQUIRED so the width stays exact — matching width/height sharding.
{
    const r = T.computeTensorMapping({
        logicalShape: [8, 8], layout: "ROW_MAJOR", sharding: "block",
        grid: { x: 2, y: 2 }, shardHeight: 4, shardWidth: 4,
    });
    eq(r.ndShardShape, [4, 4], "explicit RM block width honored (not rounded to 32)");
    eq(r.pageShape, [1, 4], "explicit RM block page = one shard-wide row");
    eq(r.mapping.numShards, 4, "explicit RM block → real 2×2 = 4 shards");

    // TILE still snaps the shard to the tile (mandatory), regardless of explicitness
    const t = T.computeTensorMapping({
        logicalShape: [64, 128], layout: "TILE", sharding: "block",
        grid: { x: 4, y: 2 }, shardHeight: 32, shardWidth: 64,
    });
    eq(t.ndShardShape, [32, 64], "explicit TILE block shard width honored");
    eq(t.mapping.numShards, 4, "TILE block shard 32×64 → 4 shards");
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

// ---- N-D padding vs folded logical_2d (rank-3 middle-dim pad) --------------
// logical [3,7,3] + alignment [1,3,2] → padded [3,9,4], physical [27,4].
// isPadding must use N-D bounds: every z-plane keeps y=0..6 real. Folding only
// logical_2d [21,3] would false-pad z=2 y≥3 (r = 2·9+y ≥ 21).
{
    const r = T.computeTensorMapping({
        logicalShape: [3, 7, 3],
        layout: "ROW_MAJOR",
        sharding: "nd",
        ndShardShape: [1, 3, 2],
        grid: { x: 2, y: 2 },
        ndStrategy: "round_robin",
        ndAlignment: "REQUIRED",
        alignment: [1, 3, 2],
    });
    eq(r.paddedShape, [3, 9, 4], "ND pad: padded shape");
    eq(r.physical, [27, 4], "ND pad: physical");
    eq(r.logical2d, [21, 3], "ND pad: logical_2d fold");
    // z=2, y=3..6 are logical — must NOT be padding (the visualizer bug)
    ok(!r.element.isPadding(2 * 9 + 3, 0), "ND pad: z=2 y=3 is real");
    ok(!r.element.isPadding(2 * 9 + 6, 2), "ND pad: z=2 y=6 x=2 is real");
    // y=7..8 are padded middle dim on every plane
    ok(r.element.isPadding(0 * 9 + 7, 0), "ND pad: z=0 y=7 is pad");
    ok(r.element.isPadding(2 * 9 + 7, 0), "ND pad: z=2 y=7 is pad");
    // x=3 is padded width
    ok(r.element.isPadding(0, 3), "ND pad: x=3 is pad");
    ok(!r.element.isPadding(0, 2), "ND pad: x=2 is real");
    // Count real cells in physical grid — should equal logical volume 3·7·3 = 63
    let real = 0;
    for (let rr = 0; rr < r.element.H; rr++) {
        for (let cc = 0; cc < r.element.W; cc++) {
            if (!r.element.isPadding(rr, cc)) real++;
        }
    }
    eq(real, 63, "ND pad: real cell count = logical volume");
}

// ---- custom Alignment (TensorLayout 4th ctor arg) ---------------------------
// Empty Alignment{} → create_default_alignment. Non-empty is merged via
// initialize_alignment (each user dim rounded up to the layout default).
{
    eq(T.initializeAlignment([], [32, 32]), [32, 32], "init align: empty → default");
    eq(T.initializeAlignment([64, 64], [32, 32]), [64, 64], "init align: larger user wins");
    eq(T.initializeAlignment([16, 16], [32, 32]), [32, 32], "init align: user rounded up to default");
    eq(T.initializeAlignment([8], [32, 32]), [32, 32], "init align: shorter user right-aligned");

    // TILE + custom [64,64]: physical pads beyond the tile default.
    const t = T.computeTensorMapping({
        logicalShape: [30, 30], layout: "TILE", sharding: "interleave",
        bankGrid: { x: 4, y: 1 }, alignment: [64, 64],
    });
    eq(t.alignment, [64, 64], "custom TILE alignment stored");
    eq(t.physical, [64, 64], "custom TILE alignment pads physical");
    eq(t.paddedShape, [64, 64], "custom TILE alignment pads padded_shape");

    // Same custom alignment under block sharding (visualizer applies it to the
    // final TensorLayout, not the C++ .sharded() path which would drop it).
    const b = T.computeTensorMapping({
        logicalShape: [30, 30], layout: "TILE", sharding: "block",
        grid: { x: 2, y: 2 }, alignment: [64, 64],
    });
    eq(b.alignment, [64, 64], "custom TILE alignment survives sharding rebuild");
    eq(b.physical, [64, 64], "custom TILE alignment pads physical when sharded");

    // ROW_MAJOR interleaved: default align is [1], so custom [8] sticks.
    const r = T.computeTensorMapping({
        logicalShape: [6, 5], layout: "ROW_MAJOR", sharding: "interleave",
        bankGrid: { x: 2, y: 1 }, alignment: [8],
    });
    eq(r.alignment, [8], "custom RM interleaved alignment");
    eq(r.physical, [6, 8], "custom RM width pad");
    eq(r.paddedShape, [6, 8], "custom RM padded_shape");
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
