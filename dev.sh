#!/usr/bin/env bash
# Convenience runner: puts node (installed via nvm, not on the default PATH) in
# scope, then runs a project task. Usage: bash dev.sh {build|test|ui|all|serve [port]}
set -e
export PATH="/root/.nvm/versions/node/v26.3.0/bin:$PATH"
cd "$(dirname "$0")"
case "${1:-all}" in
    build) node build.js ;;
    test)  node test_page_mapping.js && node test_tensor_mapping.js ;;
    ui)    node build.js && node test_ui.js && node test_tensor_ui.js ;;
    all)   node build.js && node test_page_mapping.js && node test_tensor_mapping.js && node test_ui.js && node test_tensor_ui.js ;;
    serve) node serve.js "${2:-8000}" ;;
    *) echo "usage: bash dev.sh {build|test|ui|all|serve [port]}"; exit 1 ;;
esac
