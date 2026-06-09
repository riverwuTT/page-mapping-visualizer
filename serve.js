// SPDX-License-Identifier: Apache-2.0
// Minimal static file server for the visualizer. Usage: node serve.js [port]
const http = require("http");
const fs = require("fs");
const path = require("path");

const dir = __dirname;
const port = parseInt(process.argv[2], 10) || 8000;
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".md": "text/markdown" };

http
    .createServer((req, res) => {
        let rel = decodeURIComponent(req.url.split("?")[0]);
        if (rel === "/") rel = "/index.html";
        const file = path.join(dir, path.normalize(rel));
        if (!file.startsWith(dir)) {
            res.writeHead(403).end("forbidden");
            return;
        }
        fs.readFile(file, (err, data) => {
            if (err) {
                res.writeHead(404).end("not found");
                return;
            }
            res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
            res.end(data);
        });
    })
    .listen(port, () => console.log(`serving ${dir} at http://localhost:${port}/`));
