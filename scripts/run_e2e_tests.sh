#!/usr/bin/env bash
# ==============================================================================
# Dual Track E2E Test Suite Runner for DyRF-BO
# Executes 4-Tier E2E Test Suite verifying Epistemic Uncertainty in SMAC3
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Resolve pytest executable
if [[ -f "${PROJECT_ROOT}/.venv/bin/pytest" ]]; then
    PYTEST_BIN="${PROJECT_ROOT}/.venv/bin/pytest"
elif command -v pytest &>/dev/null; then
    PYTEST_BIN="$(command -v pytest)"
else
    echo "[ERROR] pytest not found in .venv/bin/pytest or system PATH." >&2
    exit 1
fi

echo "=============================================================================="
echo "          DyRF-BO Epistemic Uncertainty Quantification in SMAC3"
echo "                       End-to-End Test Suite"
echo "=============================================================================="
echo "Project Root: ${PROJECT_ROOT}"
echo "Pytest Bin:   ${PYTEST_BIN}"
echo "Current Git:  $(git rev-parse --abbrev-ref HEAD)"
echo "------------------------------------------------------------------------------"

TARGET="tests/e2e/"
EXTRA_ARGS=()

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier1|tier1)
            TARGET="tests/e2e/test_tier1_features.py"
            shift
            ;;
        --tier2|tier2)
            TARGET="tests/e2e/test_tier2_boundaries.py"
            shift
            ;;
        --tier3|tier3)
            TARGET="tests/e2e/test_tier3_combinations.py"
            shift
            ;;
        --tier4|tier4)
            TARGET="tests/e2e/test_tier4_scenarios.py"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Executing: ${PYTEST_BIN} ${TARGET} ${EXTRA_ARGS[*]:-}"
"${PYTEST_BIN}" "${TARGET}" "${EXTRA_ARGS[@]}"

EXIT_CODE=$?
echo "------------------------------------------------------------------------------"
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[✓] All E2E tests passed successfully!"
else
    echo "[✗] E2E test suite failed with exit code ${EXIT_CODE}"
fi
echo "=============================================================================="

exit ${EXIT_CODE}
