#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-.venv/bin/python}
export SSL_CERT_FILE=${SSL_CERT_FILE:-$($PY -c 'import certifi; print(certifi.where())')}

$PY -m common.data
$PY -m Q1.q1a_tree_baseline
$PY -m Q1.q1bc_bagging
$PY -m Q1.q1d_tree_vs_bagging
$PY -m Q2.q2a_toy_round
$PY -m Q2.q2b_boosting_grid
$PY -m Q2.q2c_boost_vs_bag
$PY -m Q2.q2d_masked_comparison
$PY -m Q2.check_against_sklearn
$PY -m Q3.q3_refits
$PY -m Q3.q3_analysis
$PY -m Q3.q3c_coverage
$PY -m common.export_config
echo "done: results/*.json, results/logs/*.log, results/figures/*.png"
