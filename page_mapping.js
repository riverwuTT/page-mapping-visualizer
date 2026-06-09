// SPDX-License-Identifier: Apache-2.0
//
// JavaScript model of tt-metal's buffer page-mapping / allocation behavior,
// framed as two composable compartments. This is the BufferDistributionSpec
// model generalized; the named tt-metal layouts are just configurations of it.
//
// Inputs:
//   1. number of input pages  — a flat 1D list 0..N-1.
//   2. page grid (pages)      — how that 1D list is folded into 2D/ND (row-major).
//                               Its volume IS the number of pages.
//   3. bank grid              — how banks / cores are laid out as a 2D grid.
//
// Compartment 1  pages -> shards   (page grid, shard shape) -> vector<Shard>
//   A shard is an N-D vector of page-ids (length = shard volume), row-major
//   within the shard, padded with a sentinel (null) where it overhangs the grid.
//   Mirrors iterate_over_shards / iterate_within_shard in buffer_distribution_spec.cpp.
//
// Compartment 2  shards -> banks   (shards, bank grid, distribution) -> vector<Bank>
//   Each bank receives shard(s), stacked into its device-page space.
//     - round_robin : shard s -> bank (s % numBanks), stacked by floor(s / numBanks).
//                     (Interleaved == shard volume 1 + round_robin == Buffer::page_address.)
//     - grid_2d     : 2D shard-grid position -> 2D bank-grid position (classic block).
//
// Everything is expressed in pages.

(function (root, factory) {
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.PageMapping = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    // Sentinel for a shard slot that overhangs the page grid (no real page).
    const PADDING = null;

    const divUp = (a, b) => Math.floor((a + b - 1) / b);
    const roundUp = (a, b) => divUp(a, b) * b;
    const align = roundUp;
    const volume = (shape) => shape.reduce((acc, d) => acc * d, 1);

    // strides[i] = product of dims after i (compute_strides in shape.cpp).
    function computeStrides(shape) {
        if (shape.length === 0) return [];
        let n = volume(shape);
        if (n === 0) return shape.map(() => 0);
        const strides = [];
        for (let i = 0; i < shape.length; i++) {
            n = Math.floor(n / shape[i]);
            strides.push(n);
        }
        return strides;
    }

    // Single contiguous bank/core grid at (0,0). rowWise == ROW_MAJOR.
    function gridToCores(gridX, gridY, rowWise, maxCores) {
        const cores = [];
        if (rowWise) {
            for (let y = 0; y < gridY; y++) for (let x = 0; x < gridX; x++) cores.push({ x, y });
        } else {
            for (let x = 0; x < gridX; x++) for (let y = 0; y < gridY; y++) cores.push({ x, y });
        }
        if (maxCores != null && cores.length > maxCores) cores.length = maxCores;
        return cores;
    }

    // squeeze_shape_ranks: collapse trailing dims that match / divide, so callers
    // may pass a shard of lower rank. Kept for fidelity; the UI passes equal ranks.
    const at = (shape, negIdx) => shape[shape.length + negIdx];
    function squeezeShapeRanks(tensorShape, shardShape) {
        if (tensorShape.length < shardShape.length) {
            throw new Error("page-grid rank can't be less than shard rank");
        }
        const tVol = volume(tensorShape);
        const sVol = volume(shardShape);
        const newT = [];
        const newS = [];
        let matching = false;
        let divisible = false;
        let curT = 1;
        let curS = 1;
        for (let dim = -1; dim >= -shardShape.length; dim--) {
            const ts = at(tensorShape, dim);
            const ss = at(shardShape, dim);
            let merge = false;
            if (dim < -1) merge = matching || (ss === 1 && divisible);
            if (merge) {
                newT[newT.length - 1] *= ts;
                newS[newS.length - 1] *= ss;
            } else {
                newT.push(ts);
                newS.push(ss);
                matching = true;
            }
            matching = matching && ts === ss;
            divisible = ts % ss === 0;
            curT *= ts;
            curS *= ss;
            if (curT === tVol && curS === sVol) break;
        }
        for (let dim = -shardShape.length - 1; dim >= -tensorShape.length; dim--) {
            newT[newT.length - 1] *= at(tensorShape, dim);
        }
        newT.reverse();
        newS.reverse();
        return [newT, newS];
    }

    // =============================================================================
    // COMPARTMENT 1:  pages -> shards
    // =============================================================================
    // Returns { shards, shardGrid, numShards, shardVolume }.
    //   shard = { id, gridCoord: [..], shape: [..](full padded shard shape),
    //             pages: [pageId|PADDING] (length shardVolume, row-major in-shard) }
    function pagesToShards(pageGrid, shardShape) {
        // Shard rank may be lower than the page-grid rank; squeeze folds the extra
        // leading page dims so the two ranks match (mirrors squeeze_shape_ranks).
        // This does not change the page → shard result, only its representation.
        if (pageGrid.length < shardShape.length) {
            throw new Error(
                `page grid rank (${pageGrid.length}) can't be less than shard rank (${shardShape.length})`
            );
        }
        if (pageGrid.length !== shardShape.length) {
            [pageGrid, shardShape] = squeezeShapeRanks(pageGrid, shardShape);
        }
        const rank = pageGrid.length;
        if (shardShape.some((d) => d <= 0)) throw new Error("shard dims must be > 0");
        if (pageGrid.some((d) => d < 0)) throw new Error("page-grid dims must be >= 0");

        const shardVolume = volume(shardShape);
        const shardGrid = pageGrid.map((p, i) => Math.max(divUp(p, shardShape[i]), 1));
        const numShards = volume(shardGrid);
        const pageStrides = computeStrides(pageGrid); // page-id contribution per dim
        const shardStrides = computeStrides(shardShape); // in-shard slot contribution per dim
        const actual = new Array(rank).fill(0); // valid extent of the current shard per dim

        const shards = [];

        function iterateWithin(dim, srcOffset, dstOffset, pages) {
            if (dim === rank) {
                pages[dstOffset] = srcOffset; // srcOffset is the flat page id
                return;
            }
            for (let i = 0; i < actual[dim]; i++) {
                iterateWithin(dim + 1, srcOffset, dstOffset, pages);
                srcOffset += pageStrides[dim];
                dstOffset += shardStrides[dim];
            }
        }

        function iterateOver(dim, srcOffset, gridCoord) {
            if (dim === rank) {
                const pages = new Array(shardVolume).fill(PADDING);
                iterateWithin(0, srcOffset, 0, pages);
                shards.push({ id: shards.length, gridCoord: gridCoord.slice(), shape: shardShape.slice(), pages });
                return;
            }
            const shardSize = shardShape[dim];
            for (let i = 0; i < shardGrid[dim]; i++) {
                const isLast = i === shardGrid[dim] - 1;
                const partial = pageGrid[dim] % shardSize;
                actual[dim] = isLast && partial !== 0 ? partial : shardSize;
                gridCoord.push(i);
                iterateOver(dim + 1, srcOffset + i * shardSize * pageStrides[dim], gridCoord);
                gridCoord.pop();
            }
        }

        if (volume(pageGrid) !== 0) iterateOver(0, 0, []);
        // pageGrid / shardShape returned are the (possibly squeezed) shapes the
        // partition actually used; shards carry the same squeezed shard shape.
        return { shards, shardGrid, numShards, shardVolume, pageGrid, shardShape };
    }

    // =============================================================================
    // COMPARTMENT 2:  shards -> banks
    // =============================================================================
    function bankIdToCoord(id, bankGrid, rowMajor) {
        return rowMajor
            ? { x: id % bankGrid.x, y: Math.floor(id / bankGrid.x) }
            : { x: Math.floor(id / bankGrid.y), y: id % bankGrid.y };
    }
    function bankCoordToId(x, y, bankGrid, rowMajor) {
        return rowMajor ? y * bankGrid.x + x : x * bankGrid.y + y;
    }

    // Returns { banks, numBanks }.
    //   bank = { bankId, gridCoord:{x,y}, shardIds:[..],
    //            devicePages:[{ devicePage, pageId|PADDING, shardId, shardLocalOffset }] }
    function shardsToBanks(compart1, bankGrid, distribution, orientation) {
        const rowMajor = orientation !== "col_major";
        const numBanks = bankGrid.x * bankGrid.y;
        if (numBanks <= 0) throw new Error("bank grid must have at least one bank");
        const { shards, shardGrid, shardVolume } = compart1;

        const banks = [];
        for (let b = 0; b < numBanks; b++) {
            banks.push({ bankId: b, gridCoord: bankIdToCoord(b, bankGrid, rowMajor), shardIds: [] });
        }

        if (distribution === "grid_2d") {
            if (shardGrid.length !== 2) {
                throw new Error("2D-grid distribution needs a rank-2 page/shard shape");
            }
            const [rows, cols] = shardGrid; // dim0 = rows, dim1 = cols
            const needX = rowMajor ? cols : rows;
            const needY = rowMajor ? rows : cols;
            if (needX > bankGrid.x || needY > bankGrid.y) {
                throw new Error(
                    `shard grid ${rows}x${cols} doesn't fit bank grid ${bankGrid.x}x${bankGrid.y}`
                );
            }
            for (const shard of shards) {
                const [sr, sc] = shard.gridCoord;
                const x = rowMajor ? sc : sr;
                const y = rowMajor ? sr : sc;
                banks[bankCoordToId(x, y, bankGrid, rowMajor)].shardIds.push(shard.id);
            }
        } else {
            // round_robin (ROUND_ROBIN_1D)
            for (const shard of shards) {
                banks[shard.id % numBanks].shardIds.push(shard.id);
            }
        }

        // Lay assigned shards into each bank's device-page space (stacked in order).
        const pageLookup = [];
        for (const bank of banks) {
            bank.devicePages = [];
            let devicePage = 0;
            for (const sid of bank.shardIds) {
                const shard = shards[sid];
                for (let off = 0; off < shard.pages.length; off++) {
                    const pageId = shard.pages[off];
                    bank.devicePages.push({
                        devicePage,
                        pageId,
                        shardId: sid,
                        shardLocalOffset: off,
                    });
                    if (pageId != null) {
                        pageLookup[pageId] = { bankId: bank.bankId, devicePage, shardId: sid, shardLocalOffset: off };
                    }
                    devicePage++;
                }
            }
        }

        return { banks, numBanks, pageLookup };
    }

    // =============================================================================
    // LEGACY (classic) SHARDING  — distinct emplacement from ND
    // =============================================================================
    // Ports core_to_host_pages (buffer.cpp): the page → shard assignment for the
    // classic HEIGHT / WIDTH / BLOCK layouts.
    //   - HEIGHT chops the flat row-major page stream into CONTIGUOUS chunks of
    //     shard-volume pages (page_id++), regardless of the tensor width. This is
    //     the key difference from ND, which gathers an N-D strided sub-block.
    //   - WIDTH / BLOCK share one 2D strided-gather branch (same content as ND).
    // Returns per-shard host-page lists + each shard's actual (valid) [rows, cols].
    function coreToHostPages(layout, numShards, shardInPages, tensor2d) {
        const retVec = [];
        const retShardShape = [];
        for (let i = 0; i < numShards; i++) {
            retVec.push([]);
            retShardShape.push([shardInPages[0], shardInPages[1]]);
        }
        const pagesPerShard = shardInPages[0] * shardInPages[1];

        if (layout === "height") {
            let remPages = tensor2d[0] * tensor2d[1];
            let pageId = 0;
            for (let i = 0; i < numShards; i++) {
                if (remPages === 0) {
                    retShardShape[i] = [0, 0];
                } else {
                    const numCols = Math.min(pagesPerShard, remPages);
                    if (pagesPerShard > remPages) {
                        retShardShape[i] = [Math.floor(remPages / shardInPages[1]), shardInPages[1]];
                    }
                    for (let j = 0; j < numCols; j++) retVec[i].push(pageId++);
                    remPages -= numCols;
                }
            }
        } else {
            const numShardColumns = shardInPages[1] === 0 ? 0 : divUp(tensor2d[1], shardInPages[1]);
            let iOffset = 0;
            let jOffset = 0;
            let shardInRow = 0;
            for (let shardIdx = 0; shardIdx < numShards; shardIdx++) {
                let i = 0;
                let j = 0;
                for (i = iOffset; i < shardInPages[0] + iOffset; i++) {
                    if (i >= tensor2d[0]) break;
                    for (j = jOffset; j < shardInPages[1] + jOffset && j < tensor2d[1]; j++) {
                        retVec[shardIdx].push(i * tensor2d[1] + j);
                    }
                }
                retShardShape[shardIdx] = [i - iOffset, j - jOffset];
                if (shardInRow + 1 === numShardColumns) {
                    shardInRow = 0;
                    jOffset = 0;
                    iOffset += shardInPages[0];
                } else {
                    shardInRow++;
                    jOffset += shardInPages[1];
                }
            }
        }
        return { retVec, retShardShape };
    }

    // Full legacy pipeline: pages -> shards (core_to_host_pages, re-laid into each
    // shard's padded 2D grid by generate_buffer_page_mapping) -> banks (exactly one
    // shard per bank, shard i -> bank i in row/col-major order).
    function computeLegacy(cfg) {
        const layout = cfg.legacyLayout || "block";
        const tensor2d = cfg.pageGrid;
        const shardInPages = cfg.shardShape;
        if (tensor2d.length !== 2 || shardInPages.length !== 2) {
            throw new Error("legacy sharding needs a rank-2 page grid and shard shape");
        }
        const [H, W] = tensor2d;
        const [sh, sw] = shardInPages;
        if (sh <= 0 || sw <= 0) throw new Error("shard dims must be > 0");
        const shardPages = sh * sw;

        let numShards;
        let shardGrid;
        if (layout === "height") {
            numShards = Math.max(divUp(H * W, shardPages), 1);
            shardGrid = [numShards, 1];
        } else {
            const rows = Math.max(divUp(H, sh), 1);
            const cols = Math.max(divUp(W, sw), 1);
            numShards = rows * cols;
            shardGrid = [rows, cols];
        }

        const rowMajor = (cfg.orientation || "row_major") !== "col_major";
        const numBanks = cfg.bankGrid.x * cfg.bankGrid.y;
        if (numBanks < numShards) {
            throw new Error(
                `legacy ${layout} needs ${numShards} banks (one shard per bank) but bank grid has ${numBanks}`
            );
        }

        const { retVec, retShardShape } = coreToHostPages(layout, numShards, shardInPages, tensor2d);

        // generate_buffer_page_mapping: re-lay each shard into its padded [sh, sw] grid.
        const shards = [];
        const cols = shardGrid[shardGrid.length - 1];
        for (let c = 0; c < numShards; c++) {
            const padded = new Array(shardPages).fill(PADDING);
            let valid = 0;
            for (let sx = 0; sx < sh; sx++) {
                for (let sy = 0; sy < sw; sy++) {
                    if (sx < retShardShape[c][0] && sy < retShardShape[c][1]) {
                        padded[sx * sw + sy] = retVec[c][valid++];
                    }
                }
            }
            const gridCoord = layout === "height" ? [c, 0] : [Math.floor(c / cols), c % cols];
            shards.push({ id: c, gridCoord, shape: [sh, sw], pages: padded });
        }

        // one shard per bank, sequential (shard i -> bank i)
        const banks = [];
        for (let b = 0; b < numBanks; b++) {
            banks.push({ bankId: b, gridCoord: bankIdToCoord(b, cfg.bankGrid, rowMajor), shardIds: [] });
        }
        for (let c = 0; c < numShards; c++) banks[c].shardIds.push(c);

        const pageLookup = [];
        for (const bank of banks) {
            bank.devicePages = [];
            let dp = 0;
            for (const sid of bank.shardIds) {
                const shard = shards[sid];
                for (let off = 0; off < shard.pages.length; off++) {
                    const pageId = shard.pages[off];
                    bank.devicePages.push({ devicePage: dp, pageId, shardId: sid, shardLocalOffset: off });
                    if (pageId != null) {
                        pageLookup[pageId] = { bankId: bank.bankId, devicePage: dp, shardId: sid, shardLocalOffset: off };
                    }
                    dp++;
                }
            }
        }

        return { shards, shardGrid, numShards, shardVolume: shardPages, banks, numBanks, pageLookup, layout };
    }

    // =============================================================================
    // INTERLEAVED  — its own layout (TensorMemoryLayout::INTERLEAVED, not sharded).
    // =============================================================================
    // Every page is its own unit; pages round-robin across banks:
    //   page p -> bank (p % numBanks), device slot floor(p / numBanks).
    // (ND with a volume-1 shard reproduces this, but interleaved is its own thing.)
    function computeInterleaved(cfg) {
        const numPages = volume(cfg.pageGrid);
        const rowMajor = (cfg.orientation || "row_major") !== "col_major";
        const numBanks = cfg.bankGrid.x * cfg.bankGrid.y;
        if (numBanks <= 0) throw new Error("bank grid must have at least one bank");

        // each page is a 1-page unit ("shard" of volume 1)
        const shards = [];
        for (let p = 0; p < numPages; p++) {
            shards.push({ id: p, gridCoord: [p], shape: [1], pages: [p] });
        }

        const banks = [];
        for (let b = 0; b < numBanks; b++) {
            banks.push({ bankId: b, gridCoord: bankIdToCoord(b, cfg.bankGrid, rowMajor), shardIds: [], devicePages: [] });
        }
        const pageLookup = [];
        for (let p = 0; p < numPages; p++) {
            const b = p % numBanks;
            const slot = Math.floor(p / numBanks);
            banks[b].shardIds.push(p);
            banks[b].devicePages.push({ devicePage: slot, pageId: p, shardId: p, shardLocalOffset: 0 });
            pageLookup[p] = { bankId: b, devicePage: slot, shardId: p, shardLocalOffset: 0 };
        }

        return { shards, shardGrid: [numPages], numShards: numPages, shardVolume: 1, banks, numBanks, pageLookup };
    }

    // =============================================================================
    // PUBLIC ENTRY:  pages -> shards -> banks
    // =============================================================================
    // cfg = { pageGrid:[..], shardShape:[..], bankGrid:{x,y},
    //         distribution:"round_robin"|"grid_2d", orientation:"row_major"|"col_major" }
    function computeMapping(cfg) {
        const pageGrid = cfg.pageGrid.slice();
        const shardShape = cfg.shardShape.slice();
        const distribution = cfg.distribution || "round_robin";
        const orientation = cfg.orientation || "row_major";

        if (distribution === "interleaved") {
            const I = computeInterleaved({ ...cfg, pageGrid, orientation });
            return {
                pageGrid,
                shardShape: [1],
                inputShardShape: [1],
                squeezed: false,
                squeezedPageGrid: pageGrid,
                bankGrid: cfg.bankGrid,
                distribution,
                orientation,
                numPages: volume(pageGrid),
                shardGrid: I.shardGrid,
                numShards: I.numShards,
                shardVolume: I.shardVolume,
                shards: I.shards,
                banks: I.banks,
                numBanks: I.numBanks,
                pageLookup: I.pageLookup,
            };
        }

        if (distribution === "legacy") {
            const L = computeLegacy({ ...cfg, pageGrid, shardShape, orientation });
            return {
                pageGrid,
                shardShape,
                inputShardShape: shardShape,
                squeezed: false,
                squeezedPageGrid: pageGrid,
                bankGrid: cfg.bankGrid,
                distribution,
                orientation,
                legacyLayout: L.layout,
                numPages: volume(pageGrid),
                shardGrid: L.shardGrid,
                numShards: L.numShards,
                shardVolume: L.shardVolume,
                shards: L.shards,
                banks: L.banks,
                numBanks: L.numBanks,
                pageLookup: L.pageLookup,
            };
        }

        const c1 = pagesToShards(pageGrid, shardShape);
        const c2 = shardsToBanks(c1, cfg.bankGrid, distribution, orientation);

        const numPages = volume(pageGrid);
        const squeezed = c1.pageGrid.length !== pageGrid.length;
        return {
            pageGrid, // original, as entered — drives the composition view layout
            shardShape: c1.shardShape, // effective (squeezed) shard shape; matches `shards`
            inputShardShape: shardShape, // as entered
            squeezed, // true when shard rank < page-grid rank (shapes were folded)
            squeezedPageGrid: c1.pageGrid,
            bankGrid: cfg.bankGrid,
            distribution,
            orientation,
            numPages,
            shardGrid: c1.shardGrid,
            numShards: c1.numShards,
            shardVolume: c1.shardVolume,
            shards: c1.shards,
            banks: c2.banks,
            numBanks: c2.numBanks,
            pageLookup: c2.pageLookup,
        };
    }

    return {
        computeMapping,
        pagesToShards,
        shardsToBanks,
        computeLegacy,
        coreToHostPages,
        computeInterleaved,
        squeezeShapeRanks,
        computeStrides,
        gridToCores,
        divUp,
        roundUp,
        align,
        volume,
        PADDING,
    };
});
