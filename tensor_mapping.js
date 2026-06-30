// SPDX-License-Identifier: Apache-2.0
//
// JavaScript model of tt-metal's TENSOR sharding behavior — the element → page
// → core mapping. This is the layer that sits ON TOP of the buffer page-mapping
// model (page_mapping.js): a tensor adds an element-shape ÷ layout step that the
// buffer model assumed was already done ("all shapes are expressed in pages").
//
// Ported from (tt_metal/impl/tensor/spec + api/.../experimental/tensor/spec):
//   - tensor_layout.cpp  : compute_physical_shape / compute_padded_shape /
//                          compute_logical_2d_shape / initialize_alignment /
//                          compute_buffer_sharding_args
//   - page_config.cpp    : get_page_shape / create_default_alignment /
//                          {required,recommended}_shard_shape_alignment
//   - tensor_spec.cpp    : height_sharded / width_sharded / block_sharded /
//                          sharded / sharded_across_dims(_except) /
//                          populate_legacy_shard_spec_from_nd
//   - buffer_distribution_spec.cpp : convert_shape_to_pages / from_shard_spec
//
// Pipeline (all element counts, then converted to pages at the end):
//
//   logical_shape (N-D, elements)
//        │  fold to 2D + align         compute_physical_shape
//        ▼
//   physical 2D [H, W] (elements)
//        │  ÷ page_shape               get_page_shape (tile → 32×32, RM → 1×W)
//        ▼
//   tensor_shape_in_pages  ──────────► page_mapping.js  ──► page → core
//   shard_shape_in_pages   (the buffer-level model takes over here, unchanged)
//
// The TENSOR-specific output is the element → page assignment; the page → core
// step is delegated to PageMapping verbatim (BufferDistributionSpec semantics).

(function (root, factory) {
    if (typeof module === "object" && module.exports) {
        module.exports = factory(require("./page_mapping.js"));
    } else {
        root.TensorMapping = factory(root.PageMapping);
    }
})(typeof self !== "undefined" ? self : this, function (PM) {
    "use strict";

    const divUp = (a, b) => Math.floor((a + b - 1) / b);
    const roundUp = (a, b) => (b === 0 ? a : divUp(a, b) * b);
    const volume = (shape) => shape.reduce((acc, d) => acc * d, 1);
    const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
    const lcm = (a, b) => (a / gcd(a, b)) * b;
    // negative-index access, mirroring tt::tt_metal::Shape's operator[](-k).
    const at = (arr, negIdx) => arr[arr.length + negIdx];

    // rm_element_size_bytes (page_config.cpp). Block-float dtypes fall back to
    // float storage in ROW_MAJOR.
    const ELEMENT_SIZE_BYTES = {
        BFLOAT16: 2,
        FLOAT32: 4,
        INT32: 4,
        UINT32: 4,
        UINT16: 2,
        FP8_E4M3: 1,
        UINT8: 1,
        BFLOAT8_B: 4,
        BFLOAT4_B: 4,
    };
    function elementSizeBytes(dtype) {
        const s = ELEMENT_SIZE_BYTES[dtype];
        if (s == null) throw new Error(`unsupported data type: ${dtype}`);
        return s;
    }
    const RECOMMENDED_MEMORY_ALIGNMENT_BYTES = 64;

    // Layout = "ROW_MAJOR" | "TILE".  A PageConfig is { layout, tile:[h,w] }.
    function pageConfig(layout, tile) {
        return { layout, tile: tile || [32, 32] };
    }

    // ---------------------------------------------------------------------------
    //  PageConfig  (page_config.cpp)
    // ---------------------------------------------------------------------------

    // create_default_alignment: the alignment a layout imposes by default. This is
    // what initialize_alignment() falls back to when no explicit alignment is set.
    function createDefaultAlignment(pc, dtype, mem) {
        if (pc.layout === "TILE") {
            return [pc.tile[0], pc.tile[1]];
        }
        // ROW_MAJOR
        if (mem.shardSpec) return [mem.shardSpec.shape[1]];
        if (mem.ndShardSpec) return [at(mem.ndShardSpec.shardShape, -1)];
        return [1];
    }

    // get_page_shape: the page is a tile for TILE layout; for ROW_MAJOR it is one
    // row wide as a full physical row (interleaved) or a shard-width slice (sharded).
    function getPageShape(pc, physical2d, dtype, mem, physicalShardShape2d) {
        if (pc.layout === "TILE") {
            return [pc.tile[0], pc.tile[1]];
        }
        // ROW_MAJOR
        const [H, W] = physical2d;
        if (H === 0 || W === 0) {
            return [1, Math.floor(4 / elementSizeBytes(dtype))];
        }
        if (mem.shardSpec && mem.memoryLayout !== "HEIGHT_SHARDED") {
            if (!physicalShardShape2d) {
                throw new Error("width/block sharded RM needs a physical shard shape for page width");
            }
            return [1, physicalShardShape2d[1]];
        }
        if (isSharded(mem) && mem.ndShardSpec) {
            return [1, at(mem.ndShardSpec.shardShape, -1)];
        }
        return [1, W];
    }

    function requiredShardAlignment(pc) {
        return pc.layout === "TILE" ? [pc.tile[0], pc.tile[1]] : [1];
    }
    function recommendedShardAlignment(pc, dtype) {
        if (pc.layout === "TILE") return [pc.tile[0], pc.tile[1]];
        const es = elementSizeBytes(dtype);
        return [Math.floor(lcm(RECOMMENDED_MEMORY_ALIGNMENT_BYTES, es) / es)];
    }

    const isSharded = (mem) => mem.memoryLayout !== "INTERLEAVED";

    // ---------------------------------------------------------------------------
    //  TensorLayout  (tensor_layout.cpp)
    // ---------------------------------------------------------------------------

    // initialize_alignment: merge a (possibly empty) user alignment over the
    // layout's default alignment, rounding each user dim up to the default's.
    function initializeAlignment(userAlignment, defaultAlignment) {
        if (!userAlignment || userAlignment.length === 0) {
            return defaultAlignment.slice();
        }
        const size = Math.max(userAlignment.length, defaultAlignment.length);
        const result = new Array(size).fill(1);
        for (let i = 0; i < userAlignment.length; i++) {
            result[i + result.length - userAlignment.length] = userAlignment[i];
        }
        for (let i = 0; i < defaultAlignment.length; i++) {
            const idx = i + result.length - defaultAlignment.length;
            result[idx] = roundUp(result[idx], defaultAlignment[i]);
        }
        return result;
    }

    // compute_logical_2d_shape: fold all-but-last dims into height, last is width.
    function computeLogical2dShape(shape) {
        if (shape.length < 2) return [1, at(shape, -1)];
        let width = at(shape, -1);
        let height = at(shape, -2);
        for (let i = -3; i >= -shape.length; i--) height *= at(shape, i);
        return [height, width];
    }

    // compute_physical_shape: fold to 2D AND align each dim from the back, so the
    // last dim accumulates into width, all others into height — each rounded up to
    // its alignment entry as it is folded in.
    function computePhysicalShape(shape, alignment) {
        const rank = shape.length;
        const alignmentRank = alignment.length;
        let width = 1;
        let height = 1;
        const maxRank = Math.max(rank, alignmentRank);
        for (let i = -1; i >= -maxRank; i--) {
            let dim = i === -1 ? width : height;
            if (i >= -rank) dim *= at(shape, i);
            if (i >= -alignmentRank) dim = roundUp(dim, at(alignment, i));
            if (i === -1) width = dim;
            else height = dim;
        }
        return [height, width];
    }

    // compute_padded_shape: the N-D padded shape implied by the alignment. The last
    // two dims round up to alignment[-1]/[-2]; higher dims fold alignment via the
    // accumulated-alignment rule (or pass through when no alignment applies).
    function computePaddedShape(shape, alignment) {
        const rank = shape.length;
        const aSize = alignment.length;
        const out = new Array(Math.max(rank, aSize)).fill(0);
        let rankIndex = rank - 1;
        let alignmentIndex = aSize - 1;
        let paddedIndex = out.length - 1;
        let accumAlignment = 1;
        for (; alignmentIndex >= 0; rankIndex--, alignmentIndex--, paddedIndex--) {
            const shapeValue = rankIndex >= 0 ? shape[rankIndex] : 1;
            const alignmentValue = alignment[alignmentIndex];
            let paddedValue;
            if (rankIndex >= rank - 2) {
                paddedValue = roundUp(shapeValue, alignmentValue);
            } else if (accumAlignment % alignmentValue === 0) {
                paddedValue = shapeValue;
            } else if (alignmentValue % accumAlignment === 0) {
                paddedValue = roundUp(shapeValue, Math.floor(alignmentValue / accumAlignment));
            } else {
                throw new Error("Padded shape can't be deduced from alignment and shape");
            }
            out[paddedIndex] = paddedValue;
            if (rankIndex !== rank - 1) accumAlignment *= paddedValue;
        }
        for (; rankIndex >= 0; rankIndex--, paddedIndex--) {
            out[paddedIndex] = shape[rankIndex];
        }
        return out;
    }

    // ---------------------------------------------------------------------------
    //  MemoryConfig / TensorLayout / TensorSpec assembly
    // ---------------------------------------------------------------------------

    // A "layout" bundles dtype + page config + memory config + the resolved
    // alignment (TensorLayout). The memory config carries the sharding intent.
    function makeLayout(dtype, pc, mem) {
        const alignment = initializeAlignment([], createDefaultAlignment(pc, dtype, mem));
        return { dtype, pc, mem, alignment };
    }

    // A spec is logical_shape + layout, with cached padded / logical-2d / physical
    // shapes — exactly the fields TensorSpec caches. Building it also runs
    // populate_sharding_specs (legacy ↔ nd reconciliation).
    function makeSpec(logicalShape, layout) {
        const spec = {
            logicalShape: logicalShape.slice(),
            layout,
            paddedShape: computePaddedShape(logicalShape, layout.alignment),
            logical2d: computeLogical2dShape(logicalShape),
            physical: computePhysicalShape(logicalShape, layout.alignment),
        };
        populateShardingSpecs(spec);
        return spec;
    }

    // populate_sharding_specs: keep the legacy 2D shard spec and the N-D shard spec
    // in sync. Convenience constructors create N-D specs; this back-fills the legacy
    // 2D spec (and its HEIGHT/WIDTH/BLOCK classification) when it is expressible.
    function populateShardingSpecs(spec) {
        const mem = spec.layout.mem;
        if (mem.createdWithNd) {
            const upd = populateLegacyShardSpecFromNd(spec);
            if (upd) {
                spec.layout.mem = upd;
            }
        } else if (mem.shardSpec) {
            spec.layout.mem = populateNdShardSpecFromLegacy(spec);
        }
    }

    function populateNdShardSpecFromLegacy(spec) {
        const mem = spec.layout.mem;
        const ss = mem.shardSpec;
        let ndShape = [ss.shape[0], ss.shape[1]];
        if (spec.paddedShape.length === 1) {
            ndShape = [ss.shape[1]];
        }
        let strategy = ss.strategy || "round_robin";
        if (mem.memoryLayout === "BLOCK_SHARDED") strategy = "grid_2d";
        return {
            ...mem,
            ndShardSpec: { shardShape: ndShape, grid: ss.grid, orientation: ss.orientation, strategy },
        };
    }

    // populate_legacy_shard_spec_from_nd: try to flatten an N-D shard shape into a
    // legacy 2D [h, w] shard and classify the memory layout. Returns the updated
    // memory config, or leaves nd-only (no legacy spec) when not expressible.
    function populateLegacyShardSpecFromNd(spec) {
        const mem = spec.layout.mem;
        const nd = mem.ndShardSpec;
        const ndShape = nd.shardShape;
        const padded = spec.paddedShape;
        const paddedVolume = volume(padded);

        let shardShape = [1, at(ndShape, -1)];
        let curTensorVolume = at(padded, -1);
        let expressible = true;
        for (let dim = -2; dim >= -ndShape.length; dim--) {
            const tensorSize = at(padded, dim);
            const shardSize = at(ndShape, dim);
            curTensorVolume *= tensorSize;
            shardShape[0] *= shardSize;
            if (tensorSize === shardSize) continue;
            if (volume(ndShape) !== shardShape[0] * shardShape[1]) {
                expressible = false;
                break;
            }
            const isLastDim = dim === -ndShape.length;
            const isDivisible = tensorSize % shardSize === 0;
            const allNextOnes = paddedVolume === curTensorVolume;
            if (isLastDim || isDivisible || allNextOnes) break;
            expressible = false;
            break;
        }
        // When the N-D shard can't be flattened to a legacy 2D spec, the layout
        // stays ND_SHARDED (no legacy shardSpec is back-filled).
        if (!expressible) {
            return { ...mem, ndShardSpec: nd };
        }

        const shardSpec = { shape: shardShape, grid: nd.grid, orientation: nd.orientation };

        // Fit check: number of shards must fit the cores.
        let numH = divUp(spec.physical[0], shardSpec.shape[0]);
        let numW = divUp(spec.physical[1], shardSpec.shape[1]);
        if (nd.orientation !== "row_major") {
            const t = numH;
            numH = numW;
            numW = t;
        }
        if (numH * numW > gridNumCores(nd.grid)) {
            return { ...mem, ndShardSpec: nd }; // nd-only, no legacy spec
        }

        // Classify HEIGHT / WIDTH / BLOCK.
        let kind = "BLOCK_SHARDED";
        if (nd.strategy === "round_robin") {
            if (shardSpec.shape[0] === paddedVolume / at(padded, -1)) kind = "WIDTH_SHARDED";
            else if (shardSpec.shape[1] === at(padded, -1)) kind = "HEIGHT_SHARDED";
        }

        if (kind !== "BLOCK_SHARDED") {
            return { ...mem, memoryLayout: kind, shardSpec, ndShardSpec: nd };
        }

        // Block sharding needs a single contiguous grid.
        if (gridNumRanges(nd.grid) !== 1) return { ...mem, ndShardSpec: nd };
        const gridSize = gridSizeOf(nd.grid);
        if (nd.strategy === "round_robin" && numW !== gridSize.x) return { ...mem, ndShardSpec: nd };
        if (numW > gridSize.x || numH > gridSize.y) return { ...mem, ndShardSpec: nd };

        return { ...mem, memoryLayout: "BLOCK_SHARDED", shardSpec, ndShardSpec: nd };
    }

    // ---------------------------------------------------------------------------
    //  Grid helpers — a grid is { x, y } (one contiguous rectangle from (0,0)).
    // ---------------------------------------------------------------------------
    const gridNumCores = (g) => g.x * g.y;
    const gridNumRanges = () => 1; // UI only models a single contiguous rectangle
    const gridSizeOf = (g) => ({ x: g.x, y: g.y });

    // ---------------------------------------------------------------------------
    //  TensorSpec convenience constructors  (tensor_spec.cpp)
    // ---------------------------------------------------------------------------

    // sharded(): apply shard-shape alignment, build a new N-D-shard memory config,
    // and rebuild the spec. shardAlignment ∈ { "NONE", "REQUIRED", "RECOMMENDED" }.
    function sharded(baseSpec, ndShardSpec, shardAlignment) {
        const pc = baseSpec.layout.pc;
        const dtype = baseSpec.layout.dtype;
        const shardShape = ndShardSpec.shardShape.slice();
        if (shardAlignment !== "NONE") {
            const alignment =
                shardAlignment === "REQUIRED"
                    ? requiredShardAlignment(pc)
                    : recommendedShardAlignment(pc, dtype);
            for (let dim = 1; dim <= alignment.length; dim++) {
                shardShape[shardShape.length - dim] = roundUp(
                    shardShape[shardShape.length - dim],
                    alignment[alignment.length - dim]
                );
            }
        }
        const nd = { ...ndShardSpec, shardShape };
        const mem = {
            bufferType: baseSpec.layout.mem.bufferType,
            // MemoryConfig(buffer_type, nd_shard_spec) starts as ND_SHARDED;
            // populate_legacy_shard_spec_from_nd reclassifies to HEIGHT/WIDTH/BLOCK
            // when the shard flattens to a legacy 2D spec, else it stays ND_SHARDED.
            memoryLayout: "ND_SHARDED",
            shardSpec: null,
            ndShardSpec: nd,
            createdWithNd: true,
        };
        const layout = makeLayout(dtype, pc, mem);
        return makeSpec(baseSpec.logicalShape, layout);
    }

    function heightSharded(baseSpec, grid, orientation) {
        orientation = orientation || "row_major";
        const numCores = gridNumCores(grid);
        const shardHeight = divUp(baseSpec.physical[0], numCores);
        const nd = {
            shardShape: [shardHeight, baseSpec.physical[1]],
            grid,
            orientation,
            strategy: "round_robin",
        };
        return sharded(baseSpec, nd, "REQUIRED");
    }

    function widthSharded(baseSpec, grid, orientation) {
        orientation = orientation || "row_major";
        const numCores = gridNumCores(grid);
        const shardWidth = divUp(baseSpec.physical[1], numCores);
        const nd = {
            shardShape: [baseSpec.physical[0], shardWidth],
            grid,
            orientation,
            strategy: "round_robin",
        };
        return sharded(baseSpec, nd, "REQUIRED");
    }

    function blockSharded(baseSpec, grid, orientation) {
        orientation = orientation || "row_major";
        const gs = gridSizeOf(grid);
        const shardHeight = divUp(baseSpec.physical[0], orientation === "row_major" ? gs.y : gs.x);
        const shardWidth = divUp(baseSpec.physical[1], orientation === "row_major" ? gs.x : gs.y);
        const nd = {
            shardShape: [shardHeight, shardWidth],
            grid,
            orientation,
            strategy: "grid_2d",
        };
        return sharded(baseSpec, nd, "RECOMMENDED");
    }

    // sharded_across_dims / _except: minimal shard (1) on the chosen (or all-but-
    // chosen) dims; padded extent elsewhere. Strategy is round-robin (1D).
    function shardedAcrossDims(baseSpec, dims, grid, orientation) {
        orientation = orientation || "row_major";
        const shardShape = baseSpec.paddedShape.slice();
        for (const d of dims) shardShape[d < 0 ? shardShape.length + d : d] = 1;
        return sharded(baseSpec, { shardShape, grid, orientation, strategy: "round_robin" }, "RECOMMENDED");
    }

    // ---------------------------------------------------------------------------
    //  BufferDistributionSpec bridge  (buffer_distribution_spec.cpp)
    // ---------------------------------------------------------------------------
    // convert_shape_to_pages: only the last two dims are divided by the page shape.
    function convertShapeToPages(shape, pageShape) {
        const out = shape.slice();
        if (out.length >= 1) out[out.length - 1] = divUp(at(out, -1), pageShape[1]);
        if (out.length >= 2) out[out.length - 2] = divUp(at(out, -2), pageShape[0]);
        return out;
    }

    // compute_buffer_sharding_args: produce the page-grid + shard-shape (in pages)
    // and distribution strategy that the buffer-level model consumes. For sharded
    // tensors the N-D distribution spec (from the N-D shard) is authoritative — it
    // is what generate_buffer_page_mapping uses when a distribution spec is present.
    function computeBufferShardingArgs(spec) {
        const { layout } = spec;
        const { mem, dtype, pc } = layout;
        const physical = spec.physical;

        if (!isSharded(mem)) {
            const pageShape = getPageShape(pc, physical, dtype, mem, null);
            const tensorShapeInPages = convertShapeToPages(spec.paddedShape, pageShape);
            return {
                sharded: false,
                pageShape,
                physical,
                tensorShapeInPages,
                tensor2dInPages: [
                    divUp(physical[0], pageShape[0]),
                    divUp(physical[1], pageShape[1]),
                ],
                distribution: "interleaved",
            };
        }

        const physicalShardShape = mem.shardSpec ? mem.shardSpec.shape : null;
        const pageShape = getPageShape(pc, physical, dtype, mem, physicalShardShape);

        // The N-D shard drives the distribution (legacy spec, if any, is mirror-only).
        const nd = mem.ndShardSpec;
        const tensorShapeInPages = convertShapeToPages(spec.paddedShape, pageShape);
        const shardShapeInPages = convertShapeToPages(nd.shardShape, pageShape);
        return {
            sharded: true,
            pageShape,
            physical,
            tensorShapeInPages,
            shardShapeInPages,
            tensor2dInPages: [divUp(physical[0], pageShape[0]), divUp(physical[1], pageShape[1])],
            distribution: nd.strategy, // "round_robin" | "grid_2d"
            grid: nd.grid,
            orientation: nd.orientation,
        };
    }

    // ---------------------------------------------------------------------------
    //  ELEMENT → PAGE  (the tensor-specific layer)
    // ---------------------------------------------------------------------------
    // Tile the physical 2D shape by the page shape. Page ids are row-major over the
    // page grid — identical ordering to the buffer-level tensor_shape_in_pages, so
    // a page id means the same thing in both the element view and the page→core view.
    //   element (r, c) → page (⌊r/ph⌋·pagesW + ⌊c/pw⌋); padding when r≥lh or c≥lw.
    function elementToPage(physical, pageShape, logical2d) {
        const [H, W] = physical;
        const [ph, pw] = pageShape;
        const pagesH = divUp(H, ph);
        const pagesW = divUp(W, pw);
        const [lh, lw] = logical2d;
        return {
            H,
            W,
            ph,
            pw,
            pagesH,
            pagesW,
            logicalH: lh,
            logicalW: lw,
            // page id of element (r, c)
            pageOf: (r, c) => Math.floor(r / ph) * pagesW + Math.floor(c / pw),
            // offset within its page (row-major in-page)
            slotOf: (r, c) => (r % ph) * pw + (c % pw),
            isPadding: (r, c) => r >= lh || c >= lw,
        };
    }

    // ---------------------------------------------------------------------------
    //  PUBLIC ENTRY
    // ---------------------------------------------------------------------------
    // cfg = {
    //   logicalShape: [..],          // N-D, in elements
    //   layout: "ROW_MAJOR"|"TILE",
    //   tile: [h, w],                // for TILE (default 32×32)
    //   dtype: "BFLOAT16"|...,
    //   sharding: "interleave"|"height"|"width"|"block"|"nd",
    //   grid: { x, y },              // core grid (sharded modes)
    //   orientation: "row_major"|"col_major",
    //   ndShardShape: [..],          // for sharding === "nd"
    //   ndStrategy: "round_robin"|"grid_2d",   // for "nd"
    //   ndAlignment: "NONE"|"REQUIRED"|"RECOMMENDED", // for "nd" (default RECOMMENDED)
    //   bankGrid: { x, y },          // for interleave (where pages round-robin)
    // }
    function computeTensorMapping(cfg) {
        const dtype = cfg.dtype || "BFLOAT16";
        const layoutName = cfg.layout === "ROW_MAJOR" ? "ROW_MAJOR" : "TILE";
        const pc = pageConfig(layoutName, cfg.tile);
        const orientation = cfg.orientation || "row_major";

        // Base (interleaved) spec — convenience constructors shard relative to it.
        const baseMem = {
            bufferType: "L1",
            memoryLayout: "INTERLEAVED",
            shardSpec: null,
            ndShardSpec: null,
            createdWithNd: false,
        };
        const baseSpec = makeSpec(cfg.logicalShape, makeLayout(dtype, pc, baseMem));

        let spec;
        if (cfg.sharding === "interleave") {
            spec = baseSpec;
        } else if (cfg.sharding === "height") {
            spec = heightSharded(baseSpec, cfg.grid, orientation);
        } else if (cfg.sharding === "width") {
            spec = widthSharded(baseSpec, cfg.grid, orientation);
        } else if (cfg.sharding === "block") {
            spec = blockSharded(baseSpec, cfg.grid, orientation);
        } else if (cfg.sharding === "nd") {
            const nd = {
                shardShape: cfg.ndShardShape.slice(),
                grid: cfg.grid,
                orientation,
                strategy: cfg.ndStrategy || "round_robin",
            };
            spec = sharded(baseSpec, nd, cfg.ndAlignment || "RECOMMENDED");
        } else {
            throw new Error(`unknown sharding mode: ${cfg.sharding}`);
        }

        const bufArgs = computeBufferShardingArgs(spec);
        const em = elementToPage(bufArgs.physical, bufArgs.pageShape, spec.logical2d);

        // Delegate page → core to the buffer-level model (page_mapping.js).
        let mapping;
        if (!bufArgs.sharded) {
            const bankGrid = cfg.bankGrid || cfg.grid || { x: gridDefaultBanks(bufArgs), y: 1 };
            mapping = PM.computeMapping({
                pageGrid: bufArgs.tensorShapeInPages,
                shardShape: [1],
                bankGrid,
                distribution: "interleaved",
                orientation,
            });
        } else {
            mapping = PM.computeMapping({
                pageGrid: bufArgs.tensorShapeInPages,
                shardShape: bufArgs.shardShapeInPages,
                bankGrid: { x: cfg.grid.x, y: cfg.grid.y },
                distribution: bufArgs.distribution, // "round_robin" | "grid_2d"
                orientation,
            });
        }

        return {
            cfg,
            spec,
            dtype,
            layout: layoutName,
            tile: pc.tile,
            memoryLayout: spec.layout.mem.memoryLayout,
            alignment: spec.layout.alignment,
            logicalShape: spec.logicalShape,
            paddedShape: spec.paddedShape,
            logical2d: spec.logical2d,
            physical: spec.physical,
            pageShape: bufArgs.pageShape,
            tensor2dInPages: bufArgs.tensor2dInPages,
            tensorShapeInPages: bufArgs.tensorShapeInPages,
            shardShapeInPages: bufArgs.shardShapeInPages || null,
            ndShardShape: spec.layout.mem.ndShardSpec ? spec.layout.mem.ndShardSpec.shardShape : null,
            distribution: bufArgs.distribution,
            element: em,
            mapping, // the full buffer-level page→core result (reuses page_mapping.js)
        };
    }

    const gridDefaultBanks = () => 1;

    return {
        computeTensorMapping,
        // exposed for unit tests / reuse
        computePhysicalShape,
        computePaddedShape,
        computeLogical2dShape,
        createDefaultAlignment,
        initializeAlignment,
        getPageShape,
        requiredShardAlignment,
        recommendedShardAlignment,
        computeBufferShardingArgs,
        convertShapeToPages,
        elementToPage,
        heightSharded,
        widthSharded,
        blockSharded,
        sharded,
        shardedAcrossDims,
        makeSpec,
        makeLayout,
        pageConfig,
        elementSizeBytes,
        divUp,
        roundUp,
        lcm,
        volume,
    };
});
