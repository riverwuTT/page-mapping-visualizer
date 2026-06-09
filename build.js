// SPDX-License-Identifier: Apache-2.0
// Inlines style.css, page_mapping.js and app.js into index.html, producing the
// self-contained single-file page_mapping_viz.html. No drift: the algorithm
// shipped in the HTML is byte-for-byte the tested page_mapping.js.

const fs = require("fs");
const path = require("path");
const dir = __dirname;

const read = (f) => fs.readFileSync(path.join(dir, f), "utf8");

let html = read("index.html");
const css = read("style.css");
const pm = read("page_mapping.js");
const app = read("app.js");

html = html
    .replace('<!--__STYLE__--><link rel="stylesheet" href="style.css" />', `<style>\n${css}</style>`)
    .replace('<!--__PAGE_MAPPING_JS__--><script src="page_mapping.js"></script>', `<script>\n${pm}</script>`)
    .replace('<!--__APP_JS__--><script src="app.js"></script>', `<script>\n${app}</script>`);

const out = path.join(dir, "page_mapping_viz.html");
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${html.length} bytes)`);
