// SPDX-License-Identifier: Apache-2.0
// Inlines the CSS + JS into each dev shell, producing the self-contained
// single-file deliverables. No drift: the code shipped in the HTML is
// byte-for-byte the tested source modules.
//
//   buffer.html  + style.css + page_mapping.js + app.js
//        → page_mapping_viz.html
//   tensor.html  + style.css + page_mapping.js + tensor_mapping.js + tensor_app.js
//        → tensor_mapping_viz.html

const fs = require("fs");
const path = require("path");
const dir = __dirname;

const read = (f) => fs.readFileSync(path.join(dir, f), "utf8");
const css = read("style.css");
const pm = read("page_mapping.js");

function emit(out, html) {
    const p = path.join(dir, out);
    fs.writeFileSync(p, html);
    console.log(`wrote ${p} (${html.length} bytes)`);
}

// ---- buffer visualizer ----
{
    const app = read("app.js");
    const html = read("buffer.html")
        .replace('<!--__STYLE__--><link rel="stylesheet" href="style.css" />', `<style>\n${css}</style>`)
        .replace('<!--__PAGE_MAPPING_JS__--><script src="page_mapping.js"></script>', `<script>\n${pm}</script>`)
        .replace('<!--__APP_JS__--><script src="app.js"></script>', `<script>\n${app}</script>`);
    emit("page_mapping_viz.html", html);
}

// ---- tensor visualizer ----
{
    const tm = read("tensor_mapping.js");
    const tapp = read("tensor_app.js");
    const html = read("tensor.html")
        .replace('<!--__STYLE__--><link rel="stylesheet" href="style.css" />', `<style>\n${css}</style>`)
        .replace('<!--__PAGE_MAPPING_JS__--><script src="page_mapping.js"></script>', `<script>\n${pm}</script>`)
        .replace('<!--__TENSOR_MAPPING_JS__--><script src="tensor_mapping.js"></script>', `<script>\n${tm}</script>`)
        .replace('<!--__TENSOR_APP_JS__--><script src="tensor_app.js"></script>', `<script>\n${tapp}</script>`);
    emit("tensor_mapping_viz.html", html);
}
