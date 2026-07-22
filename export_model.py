"""
Export a trained boosting model (XGBoost / LightGBM / CatBoost) to a
self-contained static class for native inference in C# or Go.

Supported tasks: regression, binary classification, multiclass classification.

Usage (interactive):
    python export_model.py

Usage (CLI — C#):
    python export_model.py model_lgb.txt  -c LgbmModel -o LgbmModel.cs
    python export_model.py model_xgb.json -c XgbModel  -o XgbModel.cs
    python export_model.py model_cb.cbm   -c CbModel   -o CbModel.cs

Usage (CLI — Go):
    python export_model.py model_lgb.txt  -c lgbm -o lgbm/model.go --lang go
    python export_model.py model_xgb.json -c xgb  -o xgb/model.go  --lang go
    python export_model.py model_cb.cbm   -c cb   -o cb/model.go   --lang go
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Callable

import numpy as np

_Node = dict[str, Any]

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def f64(x: float) -> str:
    """C# double literal with full precision."""
    return format(float(x), ".17g") + "d"


def f32_lit(x: float) -> str:
    """C# float literal at float32 precision."""
    return format(float(np.float32(x)), ".9g") + "f"


def go_f64(x: float) -> str:
    """Go float64 literal with full precision."""
    return format(float(x), ".17g")


def go_f32(x: float) -> str:
    """Go float32 literal (exact float32 value, no suffix - typed by context)."""
    return format(float(np.float32(x)), ".9g")


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


# LightGBM missing-value semantics (per split node):
#   missing_type None : NaN is treated as 0.0, then compared normally
#   missing_type Zero : NaN and |v| <= 1e-35 go to the default side
#   missing_type NaN  : NaN goes to the default side
_LGB_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _flatten_lgb(node: dict[str, Any], nodes: list[_Node]) -> int:
    idx = len(nodes)
    if "leaf_value" in node:
        nodes.append({"leaf": True, "value": float(node["leaf_value"])})
        return idx
    entry: _Node = {
        "leaf": False,
        "feature": node["split_feature"],
        "threshold": float(node["threshold"]),
        "default_left": bool(node.get("default_left", True)),
        "missing": _LGB_MISSING.get(str(node.get("missing_type", "None")), 0),
        "left": None,
        "right": None,
    }
    nodes.append(entry)
    entry["left"] = _flatten_lgb(node["left_child"], nodes)
    entry["right"] = _flatten_lgb(node["right_child"], nodes)
    return idx


def export_lgbm(
    model_path: str, output: str, class_name: str, lang: str = "cs"
) -> None:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=model_path)
    dump = booster.dump_model()
    trees: list[Any] = dump["tree_info"]

    num_class = int(booster.params.get("num_class", 1))
    objective = booster.params.get("objective", "")

    if num_class > 1:
        mode = "multiclass"
    elif "binary" in objective:
        mode = "binary"
    else:
        mode = "regression"

    nodes: list[_Node] = []
    roots: list[int] = []
    for t in trees:
        roots.append(_flatten_lgb(t["tree_structure"], nodes))

    if lang == "go":
        _write_go_node_array(
            nodes,
            roots,
            class_name,
            output,
            base_score=0.0,
            xgboost=False,
            mode=mode,
            num_class=num_class,
        )
    else:
        _write_cs_node_array(
            nodes,
            roots,
            class_name,
            output,
            base_score=0.0,
            xgboost=False,
            mode=mode,
            num_class=num_class,
        )

    _print_summary("LightGBM", mode, lang, len(trees), len(nodes), output)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def export_xgb(model_path: str, output: str, class_name: str, lang: str = "cs") -> None:
    import math

    with open(model_path, encoding="utf-8") as f:
        model_json: dict[str, Any] = json.load(f)

    lmp: dict[str, Any] = model_json["learner"]["learner_model_param"]
    # scalar ("0.5") or, since XGBoost 2.x multiclass, a per-class vector
    # ("[-2.76E-2,5.52E-2,...]")
    base_scores = [
        float(v) for v in str(lmp["base_score"]).strip("[]").split(",")
    ]

    num_class = int(str(lmp.get("num_class", "0")))
    objective = str(model_json["learner"]["objective"]["name"])

    if num_class > 1:
        mode = "multiclass"
    elif "logistic" in objective or "binary" in objective:
        mode = "binary"
    else:
        mode = "regression"

    # XGBoost stores base_score in the objective's *output* space and starts
    # the margin sum from ProbToMargin(base_score):
    #   logistic objectives -> logit(p); squared error etc. -> identity.
    # XGBoost >= 2.0 auto-estimates base_score from the training labels, so
    # for binary models it is a real probability, not the neutral 0.5.
    class_base: list[float] | None = None
    if mode == "binary":
        p = base_scores[0]
        if not 0.0 < p < 1.0:
            print(f"Error: cannot convert base_score={p} to a logit.")
            sys.exit(1)
        effective_base = math.log(p / (1.0 - p))
    elif mode == "multiclass":
        # per-class raw margins, added to each class score
        if len(base_scores) == 1:
            base_scores = base_scores * num_class
        class_base = base_scores
        effective_base = 0.0
    else:
        effective_base = base_scores[0]

    raw_trees: list[Any] = model_json["learner"]["gradient_booster"]["model"]["trees"]

    nodes: list[_Node] = []
    roots: list[int] = []
    for t in raw_trees:
        roots.append(_flatten_xgb_flat(t, nodes))

    if lang == "go":
        _write_go_node_array(
            nodes,
            roots,
            class_name,
            output,
            base_score=effective_base,
            xgboost=True,
            mode=mode,
            num_class=num_class,
            class_base=class_base,
        )
    else:
        _write_cs_node_array(
            nodes,
            roots,
            class_name,
            output,
            base_score=effective_base,
            xgboost=True,
            mode=mode,
            num_class=num_class,
            class_base=class_base,
        )

    _print_summary(
        "XGBoost",
        mode,
        lang,
        len(raw_trees),
        len(nodes),
        output,
        extra=f"base_score={lmp['base_score']}",
    )


def _flatten_xgb_flat(tree: dict[str, Any], nodes: list[_Node]) -> int:
    lc: Any = tree["left_children"]
    rc: Any = tree["right_children"]
    si: Any = tree["split_indices"]
    sc: Any = tree["split_conditions"]
    bw: Any = tree["base_weights"]
    dl: Any = tree["default_left"]

    def visit(nid: int) -> int:
        pos = len(nodes)
        if lc[nid] == -1:
            nodes.append({"leaf": True, "value": float(bw[nid])})
        else:
            entry: _Node = {
                "leaf": False,
                "feature": int(si[nid]),
                "threshold": float(sc[nid]),
                # missing (NaN) values follow the trained default direction
                "missing_left": bool(int(dl[nid])),
                "left": None,
                "right": None,
            }
            nodes.append(entry)
            entry["left"] = visit(lc[nid])
            entry["right"] = visit(rc[nid])
        return pos

    return visit(0)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


def export_cb(model_path: str, output: str, class_name: str, lang: str = "cs") -> None:
    import catboost as cb

    model = cb.CatBoost()
    _ = model.load_model(model_path)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)
    try:
        model.save_model(tmp_path, format="json")
        with open(tmp_path, encoding="utf-8") as f:
            cb_json: dict[str, Any] = json.load(f)
    finally:
        os.unlink(tmp_path)

    params: dict[str, Any] = cb_json.get("model_info", {}).get("params", {})
    loss: Any = params.get("loss_function", "RMSE")
    if isinstance(loss, dict):
        loss = loss.get("type", "RMSE")
    loss_lower = str(loss).lower()

    if "multiclass" in loss_lower:
        mode = "multiclass"
        num_class = int(params.get("classes_count", 0))
        if num_class == 0:
            num_class = (
                len(cb_json["oblivious_trees"][0]["leaf_values"])
                if cb_json["oblivious_trees"]
                else 2
            )
    elif "logloss" in loss_lower or "crossentropy" in loss_lower:
        mode = "binary"
        num_class = 1
    else:
        mode = "regression"
        num_class = 1

    trees: list[Any] = cb_json["oblivious_trees"]
    sb: Any = cb_json.get("scale_and_bias", [1.0, [0.0]])
    scale = float(sb[0]) if isinstance(sb[0], (int, float)) else float(sb[0][0])
    bias = float(sb[1][0]) if isinstance(sb[1], list) else float(sb[1])

    # NaN routing per float feature: "AsTrue" sets the split bit for NaN,
    # "AsFalse"/"AsIs" leave it unset (v > border is false for NaN anyway).
    nan_true_by_feature: dict[int, bool] = {}
    for ff in cb_json.get("features_info", {}).get("float_features", []):
        idx = int(ff.get("feature_index", ff.get("flat_feature_index", 0)))
        nan_true_by_feature[idx] = ff.get("nan_value_treatment") == "AsTrue"

    if lang == "go":
        _write_go_catboost(
            trees, scale, bias, class_name, output, mode=mode,
            nan_true_by_feature=nan_true_by_feature,
        )
    else:
        _write_cs_catboost(
            trees, scale, bias, class_name, output, mode=mode,
            nan_true_by_feature=nan_true_by_feature,
        )

    _print_summary(
        "CatBoost",
        mode,
        lang,
        len(trees),
        0,
        output,
        extra=f"scale={scale}  bias={bias}",
    )


# ---------------------------------------------------------------------------
# C# writers
#
# Leaf nodes: Left[i] == -1 (sentinel; no IsLeaf array needed)
# XGBoost: threshold is float[] + cast (float) to match DMatrix float32
# `using System` omitted - .NET 6+ implicit global usings cover System.Math
# ---------------------------------------------------------------------------


def _write_cs_node_array(
    nodes: list[_Node],
    roots: list[int],
    class_name: str,
    output: str,
    base_score: float = 0.0,
    xgboost: bool = False,
    mode: str = "regression",
    num_class: int = 1,
    class_base: list[float] | None = None,
) -> None:
    has_class_base = class_base is not None and any(v != 0.0 for v in class_base)
    feature: list[int] = []
    threshold: list[float] = []
    left_arr: list[int] = []
    right_arr: list[int] = []
    value: list[float] = []
    missing_child: list[int] = []  # child index taken when the feature is NaN
    missing_type: list[int] = []  # LightGBM: 0 = None, 1 = Zero, 2 = NaN

    for n in nodes:
        if n["leaf"]:
            feature.append(0)
            threshold.append(0.0)
            left_arr.append(-1)
            right_arr.append(0)
            value.append(n["value"])
            missing_child.append(0)
            missing_type.append(0)
        else:
            feature.append(n["feature"])
            threshold.append(n["threshold"])
            left_arr.append(n["left"])
            right_arr.append(n["right"])
            value.append(0.0)
            if xgboost:
                missing_child.append(n["left"] if n["missing_left"] else n["right"])
                missing_type.append(0)
            else:
                missing_child.append(n["left"] if n["default_left"] else n["right"])
                missing_type.append(n["missing"])

    # LightGBM: extra arrays only needed when some node routes missing values
    lgb_has_missing = any(t != 0 for t in missing_type)

    if xgboost:
        thr_decl = (
            "    private static readonly float[]  Threshold = ["
            + ", ".join(f32_lit(v) for v in threshold)
            + "];"
        )
    else:
        thr_decl = (
            "    private static readonly double[] Threshold = ["
            + ", ".join(f64(v) for v in threshold)
            + "];"
        )

    lines: list[str] = [
        "using System.Runtime.CompilerServices;",
        "",
        f"public static class {class_name}",
        "{",
        f"    private const int TreeCount = {len(roots)};",
    ]

    if mode == "multiclass":
        lines.append(f"    private const int NumClasses   = {num_class};")
        lines.append("    private const int TreesPerClass = TreeCount / NumClasses;")

    if base_score != 0.0:
        lines.append(f"    private const double BaseScore = {f64(base_score)};")

    if has_class_base:
        assert class_base is not None
        lines.append(
            "    private static readonly double[] ClassBase = ["
            + ", ".join(f64(v) for v in class_base)
            + "];"
        )

    lines += [
        "",
        "    private static readonly int[]    Feature   = ["
        + ", ".join(map(str, feature))
        + "];",
        thr_decl,
        "    private static readonly int[]    Left      = ["
        + ", ".join(map(str, left_arr))
        + "];",
        "    private static readonly int[]    Right     = ["
        + ", ".join(map(str, right_arr))
        + "];",
    ]

    if xgboost:
        lines.append(
            "    private static readonly int[]    Missing   = ["
            + ", ".join(map(str, missing_child))
            + "];"
        )
    elif lgb_has_missing:
        lines.append(
            "    private static readonly int[]    Default   = ["
            + ", ".join(map(str, missing_child))
            + "];"
        )
        lines.append(
            "    private static readonly byte[]   MissType  = ["
            + ", ".join(map(str, missing_type))
            + "];"
        )

    lines += [
        "    private static readonly double[] Value     = ["
        + ", ".join(f64(v) for v in value)
        + "];",
        "    private static readonly int[]    Roots     = ["
        + ", ".join(map(str, roots))
        + "];",
        "",
        "    // Left[node] == -1 means leaf node",
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    private static double Eval(int node, ReadOnlySpan<double> f)",
        "    {",
        "        while (true)",
        "        {",
        "            if (Left[node] < 0)",
        "                return Value[node];",
        "            double v = f[Feature[node]];",
    ]

    if xgboost:
        # XGBoost: NaN follows the trained default direction (Missing[node])
        lines += [
            "            if (double.IsNaN(v))",
            "                node = Missing[node];",
            "            else if ((float)v < Threshold[node])",
            "                node = Left[node];",
            "            else",
            "                node = Right[node];",
        ]
    elif lgb_has_missing:
        # Full LightGBM missing-value semantics (see MissType comment above)
        lines += [
            "            int m = MissType[node];",
            "            if (double.IsNaN(v))",
            "            {",
            "                if (m == 2) { node = Default[node]; continue; }",
            "                v = 0.0;  // missing_type None/Zero: NaN acts as 0",
            "            }",
            "            if (m == 1 && v >= -1e-35 && v <= 1e-35)",
            "            {",
            "                node = Default[node];",
            "                continue;",
            "            }",
            "            if (v <= Threshold[node])",
            "                node = Left[node];",
            "            else",
            "                node = Right[node];",
        ]
    else:
        # LightGBM missing_type None everywhere: NaN is treated as 0.0
        lines += [
            "            if (double.IsNaN(v)) v = 0.0;",
            "            if (v <= Threshold[node])",
            "                node = Left[node];",
            "            else",
            "                node = Right[node];",
        ]

    lines += [
        "        }",
        "    }",
        "",
    ]

    lines += _cs_predict_methods(mode, base_score, num_class, has_class_base)
    lines.append("}")

    _write_file(output, "\n".join(lines) + "\n")


def _cs_predict_methods(
    mode: str, base_score: float, _num_class: int, has_class_base: bool = False
) -> list[str]:
    init = "BaseScore" if base_score != 0.0 else "0"

    if mode == "regression":
        return [
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double Predict(ReadOnlySpan<double> f)",
            "    {",
            f"        double s = {init};",
            "        for (int i = 0; i < TreeCount; i++)",
            "            s += Eval(Roots[i], f);",
            "        return s;",
            "    }",
        ]

    if mode == "binary":
        return [
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double PredictScore(ReadOnlySpan<double> f)",
            "    {",
            f"        double s = {init};",
            "        for (int i = 0; i < TreeCount; i++)",
            "            s += Eval(Roots[i], f);",
            "        return s;",
            "    }",
            "",
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double PredictProbability(ReadOnlySpan<double> f)",
            "    {",
            "        double x = PredictScore(f);",
            "        return x >= 0",
            "            ? 1.0 / (1.0 + Math.Exp(-x))",
            "            : Math.Exp(x) / (1.0 + Math.Exp(x));",
            "    }",
            "",
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static int Predict(ReadOnlySpan<double> f)",
            "        => PredictProbability(f) >= 0.5 ? 1 : 0;",
        ]

    # multiclass
    scores_init = (
        "        double[] s = (double[])ClassBase.Clone();"
        if has_class_base
        else "        double[] s = new double[NumClasses];"
    )
    return [
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    public static double[] PredictScores(ReadOnlySpan<double> f)",
        "    {",
        scores_init,
        "        for (int c = 0; c < NumClasses; c++)",
        "            for (int i = 0; i < TreesPerClass; i++)",
        "                s[c] += Eval(Roots[c + i * NumClasses], f);",
        "        return s;",
        "    }",
        "",
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    public static double[] Predict(ReadOnlySpan<double> f)",
        "    {",
        "        double[] s = PredictScores(f);",
        "        double max = s[0];",
        "        for (int i = 1; i < s.Length; i++) if (s[i] > max) max = s[i];",
        "        double sum = 0;",
        "        for (int i = 0; i < s.Length; i++) { s[i] = Math.Exp(s[i] - max); sum += s[i]; }",
        "        for (int i = 0; i < s.Length; i++) s[i] /= sum;",
        "        return s;",
        "    }",
        "",
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    public static int PredictClass(ReadOnlySpan<double> f)",
        "    {",
        "        double[] p = Predict(f);",
        "        int best = 0;",
        "        for (int i = 1; i < p.Length; i++) if (p[i] > p[best]) best = i;",
        "        return best;",
        "    }",
    ]


def _write_cs_catboost(
    trees: list[Any],
    scale: float,
    bias: float,
    class_name: str,
    output: str,
    mode: str = "regression",
    _num_class: int = 1,
    nan_true_by_feature: dict[int, bool] | None = None,
) -> None:
    nan_true_by_feature = nan_true_by_feature or {}

    split_features: list[int] = []
    split_borders: list[float] = []
    split_nan_true: list[int] = []  # 1 when NaN sets the split bit ("AsTrue")
    tree_split_offset: list[int] = []
    tree_depth: list[int] = []
    leaf_values: list[float] = []
    tree_leaf_offset: list[int] = []

    for t in trees:
        tree_split_offset.append(len(split_features))
        splits = t["splits"]
        tree_depth.append(len(splits))
        for s in splits:
            fidx = int(s["float_feature_index"])
            split_features.append(fidx)
            split_borders.append(float(s["border"]))
            split_nan_true.append(1 if nan_true_by_feature.get(fidx) else 0)
        tree_leaf_offset.append(len(leaf_values))
        for v in t["leaf_values"]:
            leaf_values.append(float(v))

    # NaN comparing false already matches "AsFalse"/"AsIs"; the extra array is
    # only needed when some feature routes NaN to the true side.
    any_nan_true = any(split_nan_true)

    lines: list[str] = [
        "using System.Runtime.CompilerServices;",
        "",
        f"public static class {class_name}",
        "{",
        f"    private const int    TreeCount = {len(trees)};",
        f"    private const double Scale     = {f64(scale)};",
        f"    private const double Bias      = {f64(bias)};",
        "",
        "    private static readonly int[]    SplitFeature    = ["
        + ", ".join(map(str, split_features))
        + "];",
        "    private static readonly double[] SplitBorder     = ["
        + ", ".join(f64(v) for v in split_borders)
        + "];",
    ]

    if any_nan_true:
        lines.append(
            "    private static readonly byte[]   SplitNanTrue    = ["
            + ", ".join(map(str, split_nan_true))
            + "];"
        )

    lines += [
        "    private static readonly int[]    TreeSplitOffset = ["
        + ", ".join(map(str, tree_split_offset))
        + "];",
        "    private static readonly int[]    TreeDepth       = ["
        + ", ".join(map(str, tree_depth))
        + "];",
        "    private static readonly double[] LeafValues      = ["
        + ", ".join(f64(v) for v in leaf_values)
        + "];",
        "    private static readonly int[]    TreeLeafOffset  = ["
        + ", ".join(map(str, tree_leaf_offset))
        + "];",
        "",
        "    // bit l set when feature[split[l]] > border[split[l]]",
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    private static double EvalTree(int treeIdx, ReadOnlySpan<double> f)",
        "    {",
        "        int leafIdx     = 0;",
        "        int splitOffset = TreeSplitOffset[treeIdx];",
        "        int depth       = TreeDepth[treeIdx];",
        "        for (int l = 0; l < depth; l++)",
        "        {",
        "            int s = splitOffset + l;",
    ]

    if any_nan_true:
        lines += [
            "            double v = f[SplitFeature[s]];",
            "            bool bit = double.IsNaN(v) ? SplitNanTrue[s] == 1 : v > SplitBorder[s];",
            "            if (bit)",
            "                leafIdx |= (1 << l);",
        ]
    else:
        lines += [
            "            if (f[SplitFeature[s]] > SplitBorder[s])",
            "                leafIdx |= (1 << l);",
        ]

    lines += [
        "        }",
        "        return LeafValues[TreeLeafOffset[treeIdx] + leafIdx];",
        "    }",
        "",
    ]

    if mode == "regression":
        lines += [
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double Predict(ReadOnlySpan<double> f)",
            "    {",
            "        double s = 0;",
            "        for (int i = 0; i < TreeCount; i++)",
            "            s += EvalTree(i, f);",
            "        return Bias + Scale * s;",
            "    }",
        ]
    elif mode == "binary":
        lines += [
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double PredictScore(ReadOnlySpan<double> f)",
            "    {",
            "        double s = 0;",
            "        for (int i = 0; i < TreeCount; i++)",
            "            s += EvalTree(i, f);",
            "        return Bias + Scale * s;",
            "    }",
            "",
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static double PredictProbability(ReadOnlySpan<double> f)",
            "    {",
            "        double x = PredictScore(f);",
            "        return x >= 0",
            "            ? 1.0 / (1.0 + Math.Exp(-x))",
            "            : Math.Exp(x) / (1.0 + Math.Exp(x));",
            "    }",
            "",
            "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
            "    public static int Predict(ReadOnlySpan<double> f)",
            "        => PredictProbability(f) >= 0.5 ? 1 : 0;",
        ]

    lines.append("}")
    _write_file(output, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Go writers
#
# Leaf nodes: left[i] == -1 (sentinel; same logic as C# version)
# XGBoost: threshold is []float32; cast feature to float32 before comparing
# Package name = class_name lowercased
# ---------------------------------------------------------------------------


def _write_go_node_array(
    nodes: list[_Node],
    roots: list[int],
    class_name: str,
    output: str,
    base_score: float = 0.0,
    xgboost: bool = False,
    mode: str = "regression",
    num_class: int = 1,
    class_base: list[float] | None = None,
) -> None:
    pkg = class_name.lower()
    has_class_base = class_base is not None and any(v != 0.0 for v in class_base)

    feature: list[int] = []
    threshold: list[float] = []
    left_arr: list[int] = []
    right_arr: list[int] = []
    value: list[float] = []
    missing_child: list[int] = []  # child index taken when the feature is NaN
    missing_type: list[int] = []  # LightGBM: 0 = None, 1 = Zero, 2 = NaN

    for n in nodes:
        if n["leaf"]:
            feature.append(0)
            threshold.append(0.0)
            left_arr.append(-1)
            right_arr.append(0)
            value.append(n["value"])
            missing_child.append(0)
            missing_type.append(0)
        else:
            feature.append(n["feature"])
            threshold.append(n["threshold"])
            left_arr.append(n["left"])
            right_arr.append(n["right"])
            value.append(0.0)
            if xgboost:
                missing_child.append(n["left"] if n["missing_left"] else n["right"])
                missing_type.append(0)
            else:
                missing_child.append(n["left"] if n["default_left"] else n["right"])
                missing_type.append(n["missing"])

    # LightGBM: extra arrays only needed when some node routes missing values
    lgb_has_missing = any(t != 0 for t in missing_type)

    if xgboost:
        thr_type = "float32"
        thr_vals = ", ".join(go_f32(v) for v in threshold)
    else:
        thr_type = "float64"
        thr_vals = ", ".join(go_f64(v) for v in threshold)

    lines: list[str] = [f"package {pkg}", ""]

    # math is always needed now (math.IsNaN in eval)
    lines += ['import "math"', ""]

    # const block (gofmt aligns '=' within a block)
    if mode == "multiclass":
        const_lines = [
            f"\ttreeCount     = {len(roots)}",
            f"\tnumClasses    = {num_class}",
            "\ttreesPerClass = treeCount / numClasses",
        ]
    else:
        const_lines = [f"\ttreeCount = {len(roots)}"]
    if base_score != 0.0:
        const_lines.append(f"\tbaseScore = {go_f64(base_score)}")

    lines += ["const ("] + const_lines + [")", ""]

    class_base_line: list[str] = []
    if has_class_base:
        assert class_base is not None
        class_base_line = [
            "\tclassBase = []float64{"
            + ", ".join(go_f64(v) for v in class_base)
            + "}"
        ]

    # var block
    var_lines = class_base_line + [
        "\tfeature   = []int{" + ", ".join(map(str, feature)) + "}",
        "\tthreshold = []" + thr_type + "{" + thr_vals + "}",
        "\tleft      = []int{" + ", ".join(map(str, left_arr)) + "}",
        "\tright     = []int{" + ", ".join(map(str, right_arr)) + "}",
    ]
    if xgboost:
        var_lines.append(
            "\tmissing   = []int{" + ", ".join(map(str, missing_child)) + "}"
        )
    elif lgb_has_missing:
        var_lines.append(
            "\tdefaultTo = []int{" + ", ".join(map(str, missing_child)) + "}"
        )
        var_lines.append(
            "\tmissType  = []uint8{" + ", ".join(map(str, missing_type)) + "}"
        )
    var_lines += [
        "\tvalue     = []float64{" + ", ".join(go_f64(v) for v in value) + "}",
        "\troots     = []int{" + ", ".join(map(str, roots)) + "}",
    ]

    lines += ["//nolint:gochecknoglobals", "var ("] + var_lines + [")", ""]

    # eval function
    lines += [
        "// left[node] == -1 means leaf node",
        "func eval(node int, f []float64) float64 {",
        "\tfor {",
        "\t\tif left[node] < 0 {",
        "\t\t\treturn value[node]",
        "\t\t}",
        "\t\tv := f[feature[node]]",
    ]

    if xgboost:
        # XGBoost: NaN follows the trained default direction (missing[node])
        lines += [
            "\t\tswitch {",
            "\t\tcase math.IsNaN(v):",
            "\t\t\tnode = missing[node]",
            "\t\tcase float32(v) < threshold[node]:",
            "\t\t\tnode = left[node]",
            "\t\tdefault:",
            "\t\t\tnode = right[node]",
            "\t\t}",
        ]
    elif lgb_has_missing:
        # Full LightGBM missing-value semantics:
        #   missType 0 (None): NaN acts as 0.0, then normal comparison
        #   missType 1 (Zero): NaN and |v| <= 1e-35 go to defaultTo[node]
        #   missType 2 (NaN) : NaN goes to defaultTo[node]
        lines += [
            "\t\tm := missType[node]",
            "\t\tif math.IsNaN(v) {",
            "\t\t\tif m == 2 {",
            "\t\t\t\tnode = defaultTo[node]",
            "\t\t\t\tcontinue",
            "\t\t\t}",
            "\t\t\tv = 0.0 // missing_type None/Zero: NaN acts as 0",
            "\t\t}",
            "\t\tif m == 1 && v >= -1e-35 && v <= 1e-35 {",
            "\t\t\tnode = defaultTo[node]",
            "\t\t\tcontinue",
            "\t\t}",
            "\t\tif v <= threshold[node] {",
            "\t\t\tnode = left[node]",
            "\t\t} else {",
            "\t\t\tnode = right[node]",
            "\t\t}",
        ]
    else:
        # LightGBM missing_type None everywhere: NaN is treated as 0.0
        lines += [
            "\t\tif math.IsNaN(v) {",
            "\t\t\tv = 0.0",
            "\t\t}",
            "\t\tif v <= threshold[node] {",
            "\t\t\tnode = left[node]",
            "\t\t} else {",
            "\t\t\tnode = right[node]",
            "\t\t}",
        ]

    lines += [
        "\t}",
        "}",
        "",
    ]

    lines += _go_predict_methods(mode, base_score, num_class, has_class_base)
    _write_file(output, "\n".join(lines) + "\n")


def _write_go_catboost(
    trees: list[Any],
    scale: float,
    bias: float,
    class_name: str,
    output: str,
    mode: str = "regression",
    _num_class: int = 1,
    nan_true_by_feature: dict[int, bool] | None = None,
) -> None:
    pkg = class_name.lower()
    nan_true_by_feature = nan_true_by_feature or {}

    split_features: list[int] = []
    split_borders: list[float] = []
    split_nan_true: list[int] = []  # 1 when NaN sets the split bit ("AsTrue")
    tree_split_offset: list[int] = []
    tree_depth: list[int] = []
    leaf_values: list[float] = []
    tree_leaf_offset: list[int] = []

    for t in trees:
        tree_split_offset.append(len(split_features))
        splits = t["splits"]
        tree_depth.append(len(splits))
        for s in splits:
            fidx = int(s["float_feature_index"])
            split_features.append(fidx)
            split_borders.append(float(s["border"]))
            split_nan_true.append(1 if nan_true_by_feature.get(fidx) else 0)
        tree_leaf_offset.append(len(leaf_values))
        for v in t["leaf_values"]:
            leaf_values.append(float(v))

    # NaN comparing false already matches "AsFalse"/"AsIs"; the extra array is
    # only needed when some feature routes NaN to the true side.
    any_nan_true = any(split_nan_true)
    needs_math = mode in ("binary", "multiclass") or any_nan_true

    lines: list[str] = [f"package {pkg}", ""]

    if needs_math:
        lines += ['import "math"', ""]

    var_lines = [
        "\tsplitFeature    = []int{" + ", ".join(map(str, split_features)) + "}",
        "\tsplitBorder     = []float64{"
        + ", ".join(go_f64(v) for v in split_borders)
        + "}",
    ]
    if any_nan_true:
        var_lines.append(
            "\tsplitNanTrue    = []uint8{" + ", ".join(map(str, split_nan_true)) + "}"
        )
    var_lines += [
        "\ttreeSplitOffset = []int{" + ", ".join(map(str, tree_split_offset)) + "}",
        "\ttreeDepth       = []int{" + ", ".join(map(str, tree_depth)) + "}",
        "\tleafValues      = []float64{"
        + ", ".join(go_f64(v) for v in leaf_values)
        + "}",
        "\ttreeLeafOffset  = []int{" + ", ".join(map(str, tree_leaf_offset)) + "}",
    ]

    lines += [
        "const (",
        f"\ttreeCount = {len(trees)}",
        f"\tscale     = {go_f64(scale)}",
        f"\tbias      = {go_f64(bias)}",
        ")",
        "",
        "//nolint:gochecknoglobals",
        "var (",
    ] + var_lines + [
        ")",
        "",
        "// bit l set when feature[split[l]] > border[split[l]]",
        "func evalTree(treeIdx int, f []float64) float64 {",
        "\tleafIdx := 0",
        "\tsplitOffset := treeSplitOffset[treeIdx]",
        "\tdepth := treeDepth[treeIdx]",
        "\tfor l := 0; l < depth; l++ {",
        "\t\ts := splitOffset + l",
    ]

    if any_nan_true:
        lines += [
            "\t\tv := f[splitFeature[s]]",
            "\t\tbit := v > splitBorder[s]",
            "\t\tif math.IsNaN(v) {",
            "\t\t\tbit = splitNanTrue[s] == 1",
            "\t\t}",
            "\t\tif bit {",
            "\t\t\tleafIdx |= (1 << l)",
            "\t\t}",
        ]
    else:
        lines += [
            "\t\tif f[splitFeature[s]] > splitBorder[s] {",
            "\t\t\tleafIdx |= (1 << l)",
            "\t\t}",
        ]

    lines += [
        "\t}",
        "\treturn leafValues[treeLeafOffset[treeIdx]+leafIdx]",
        "}",
        "",
    ]

    if mode == "regression":
        lines += [
            "// Predict returns the regression score for feature vector f.",
            "func Predict(f []float64) float64 {",
            "\ts := 0.0",
            "\tfor i := 0; i < treeCount; i++ {",
            "\t\ts += evalTree(i, f)",
            "\t}",
            "\treturn bias + scale*s",
            "}",
        ]
    elif mode == "binary":
        lines += [
            "// PredictScore returns the raw logit for feature vector f.",
            "func PredictScore(f []float64) float64 {",
            "\ts := 0.0",
            "\tfor i := 0; i < treeCount; i++ {",
            "\t\ts += evalTree(i, f)",
            "\t}",
            "\treturn bias + scale*s",
            "}",
            "",
            "// PredictProbability returns sigmoid(PredictScore(f)).",
            "func PredictProbability(f []float64) float64 {",
            "\tx := PredictScore(f)",
            "\tif x >= 0 {",
            "\t\treturn 1.0 / (1.0 + math.Exp(-x))",
            "\t}",
            "\treturn math.Exp(x) / (1.0 + math.Exp(x))",
            "}",
            "",
            "// Predict returns 0 or 1 (threshold 0.5).",
            "func Predict(f []float64) int {",
            "\tif PredictProbability(f) >= 0.5 {",
            "\t\treturn 1",
            "\t}",
            "\treturn 0",
            "}",
        ]

    _write_file(output, "\n".join(lines) + "\n")


def _go_predict_methods(
    mode: str, base_score: float, _num_class: int, has_class_base: bool = False
) -> list[str]:
    init = "baseScore" if base_score != 0.0 else "0.0"

    if mode == "regression":
        return [
            "// Predict returns the regression score for feature vector f.",
            "func Predict(f []float64) float64 {",
            f"\ts := {init}",
            "\tfor _, root := range roots {",
            "\t\ts += eval(root, f)",
            "\t}",
            "\treturn s",
            "}",
        ]

    if mode == "binary":
        return [
            "// PredictScore returns the raw logit for feature vector f.",
            "func PredictScore(f []float64) float64 {",
            f"\ts := {init}",
            "\tfor _, root := range roots {",
            "\t\ts += eval(root, f)",
            "\t}",
            "\treturn s",
            "}",
            "",
            "// PredictProbability returns sigmoid(PredictScore(f)).",
            "func PredictProbability(f []float64) float64 {",
            "\tx := PredictScore(f)",
            "\tif x >= 0 {",
            "\t\treturn 1.0 / (1.0 + math.Exp(-x))",
            "\t}",
            "\treturn math.Exp(x) / (1.0 + math.Exp(x))",
            "}",
            "",
            "// Predict returns 0 or 1 (threshold 0.5).",
            "func Predict(f []float64) int {",
            "\tif PredictProbability(f) >= 0.5 {",
            "\t\treturn 1",
            "\t}",
            "\treturn 0",
            "}",
        ]

    # multiclass
    scores_init = ["\ts := make([]float64, numClasses)"]
    if has_class_base:
        scores_init.append("\tcopy(s, classBase)")
    return [
        "// PredictScores returns raw per-class scores.",
        "func PredictScores(f []float64) []float64 {",
    ] + scores_init + [
        "\tfor c := 0; c < numClasses; c++ {",
        "\t\tfor i := 0; i < treesPerClass; i++ {",
        "\t\t\ts[c] += eval(roots[c+i*numClasses], f)",
        "\t\t}",
        "\t}",
        "\treturn s",
        "}",
        "",
        "// Predict returns softmax probabilities over all classes.",
        "func Predict(f []float64) []float64 {",
        "\ts := PredictScores(f)",
        "\tmax := s[0]",
        "\tfor _, v := range s {",
        "\t\tif v > max { max = v }",
        "\t}",
        "\tsum := 0.0",
        "\tfor i := range s {",
        "\t\ts[i] = math.Exp(s[i] - max)",
        "\t\tsum += s[i]",
        "\t}",
        "\tfor i := range s { s[i] /= sum }",
        "\treturn s",
        "}",
        "",
        "// PredictClass returns the class index with the highest probability.",
        "func PredictClass(f []float64) int {",
        "\tp := Predict(f)",
        "\tbest := 0",
        "\tfor i, v := range p {",
        "\t\tif v > p[best] { best = i }",
        "\t\t_ = v",
        "\t}",
        "\treturn best",
        "}",
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_file(path: str, content: str) -> None:
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        _ = fh.write(content)


def _print_summary(
    library: str,
    mode: str,
    lang: str,
    n_trees: int,
    n_nodes: int,
    output: str,
    extra: str = "",
) -> None:
    print(f"Library : {library}")
    print(f"Mode    : {mode}")
    print(f"Lang    : {lang}")
    print(f"Trees   : {n_trees}" + (f"  {extra}" if extra else ""))
    if n_nodes:
        print(f"Nodes   : {n_nodes}")
    print(f"Output  : {output}")


EXTENSIONS: dict[str, str] = {
    ".json": "xgboost",
    ".txt": "lightgbm",
    ".cbm": "catboost",
}
LIBRARY_NAMES: dict[str, str] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def _get_exporter(library: str, lang: str) -> Callable[[str, str, str], None]:
    fns = {
        "xgboost": export_xgb,
        "lightgbm": export_lgbm,
        "catboost": export_cb,
    }
    fn = fns[library]
    return lambda path, output, cls: fn(path, output, cls, lang=lang)


def detect_library(model_path: str) -> str | None:
    return EXTENSIONS.get(os.path.splitext(model_path)[1].lower())


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_interactive() -> None:
    print("=== Boosting Model Exporter ===\n")

    model_path = input("Model file path: ").strip()
    if not os.path.exists(model_path):
        print(f"Error: file not found: {model_path}")
        sys.exit(1)

    library = detect_library(model_path)
    if library:
        ext = os.path.splitext(model_path)[1]
        print(f"Inferred library: {LIBRARY_NAMES[library]}  (from {ext} extension)")
    else:
        print("Could not infer library from extension.")
        print("  1) XGBoost  (.json)\n  2) LightGBM (.txt)\n  3) CatBoost (.cbm)")
        choice = input("Choose [1/2/3]: ").strip()
        library = {"1": "xgboost", "2": "lightgbm", "3": "catboost"}.get(choice)
        if library is None:
            print("Invalid choice.")
            sys.exit(1)

    lang_input = (
        input("Output language [cs/go] (default: cs): ").strip().lower() or "cs"
    )
    if lang_input not in ("cs", "go"):
        print("Invalid language. Choose cs or go.")
        sys.exit(1)

    if lang_input == "cs":
        name_prompt = "C# class name [Model]: "
        default_name = "Model"
    else:
        name_prompt = "Go package name [model]: "
        default_name = "model"

    class_name = input(name_prompt).strip() or default_name

    if lang_input == "cs":
        default_output = f"{class_name}.cs"
    else:
        default_output = f"{class_name}.go"

    output = input(f"Output file [{default_output}]: ").strip() or default_output

    print()
    _get_exporter(library, lang_input)(model_path, output, class_name)
    print("\nDone.")


def run_cli(args: argparse.Namespace) -> None:
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: file not found: {model_path}")
        sys.exit(1)

    library = detect_library(model_path)
    if library:
        print(f"Inferred library: {LIBRARY_NAMES[library]}")
    else:
        print("Error: cannot infer library from extension (.json / .txt / .cbm).")
        sys.exit(1)

    lang = args.lang
    default_name = "Model" if lang == "cs" else "model"
    class_name = args.class_name or default_name
    default_ext = ".cs" if lang == "cs" else ".go"
    output = args.output or f"{class_name}{default_ext}"

    _get_exporter(library, lang)(model_path, output, class_name)
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a boosting model to a C# or Go static class."
    )
    _ = parser.add_argument("model_path", nargs="?", help="Path to model file")
    _ = parser.add_argument(
        "-c",
        "--class-name",
        default="",
        help="Class/package name (default: Model / model)",
    )
    _ = parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output file (default: <name>.cs or <name>.go)",
    )
    _ = parser.add_argument(
        "--lang",
        choices=["cs", "go"],
        default="cs",
        help="Output language: cs (default) or go",
    )

    args = parser.parse_args()
    if args.model_path:
        run_cli(args)
    else:
        run_interactive()
