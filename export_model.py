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


def _flatten_lgb(node: dict[str, Any], nodes: list[_Node]) -> int:
    idx = len(nodes)
    if "leaf_value" in node:
        nodes.append({"leaf": True, "value": float(node["leaf_value"])})
        return idx
    entry: _Node = {
        "leaf": False,
        "feature": node["split_feature"],
        "threshold": float(node["threshold"]),
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
    with open(model_path, encoding="utf-8") as f:
        model_json: dict[str, Any] = json.load(f)

    lmp: dict[str, Any] = model_json["learner"]["learner_model_param"]
    base_score = float(str(lmp["base_score"]).strip("[]"))

    num_class = int(str(lmp.get("num_class", "0")))
    objective = str(model_json["learner"]["objective"]["name"])

    if num_class > 1:
        mode = "multiclass"
    elif "logistic" in objective or "binary" in objective:
        mode = "binary"
    else:
        mode = "regression"

    raw_trees: list[Any] = model_json["learner"]["gradient_booster"]["model"]["trees"]

    nodes: list[_Node] = []
    roots: list[int] = []
    for t in raw_trees:
        roots.append(_flatten_xgb_flat(t, nodes))

    # multiclass XGBoost auto base_score is per-class and folded in; omit it
    effective_base = base_score if mode != "multiclass" else 0.0

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
        )

    _print_summary(
        "XGBoost",
        mode,
        lang,
        len(raw_trees),
        len(nodes),
        output,
        extra=f"base_score={base_score}",
    )


def _flatten_xgb_flat(tree: dict[str, Any], nodes: list[_Node]) -> int:
    lc: Any = tree["left_children"]
    rc: Any = tree["right_children"]
    si: Any = tree["split_indices"]
    sc: Any = tree["split_conditions"]
    bw: Any = tree["base_weights"]

    def visit(nid: int) -> int:
        pos = len(nodes)
        if lc[nid] == -1:
            nodes.append({"leaf": True, "value": float(bw[nid])})
        else:
            entry: _Node = {
                "leaf": False,
                "feature": int(si[nid]),
                "threshold": float(sc[nid]),
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

    if lang == "go":
        _write_go_catboost(trees, scale, bias, class_name, output, mode=mode)
    else:
        _write_cs_catboost(trees, scale, bias, class_name, output, mode=mode)

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
) -> None:
    feature: list[int] = []
    threshold: list[float] = []
    left_arr: list[int] = []
    right_arr: list[int] = []
    value: list[float] = []

    for n in nodes:
        if n["leaf"]:
            feature.append(0)
            threshold.append(0.0)
            left_arr.append(-1)
            right_arr.append(0)
            value.append(n["value"])
        else:
            feature.append(n["feature"])
            threshold.append(n["threshold"])
            left_arr.append(n["left"])
            right_arr.append(n["right"])
            value.append(0.0)

    if xgboost:
        thr_decl = (
            "    private static readonly float[]  Threshold = ["
            + ", ".join(f32_lit(v) for v in threshold)
            + "];"
        )
        eval_cmp = "            if ((float)f[Feature[node]] < Threshold[node])"
    else:
        thr_decl = (
            "    private static readonly double[] Threshold = ["
            + ", ".join(f64(v) for v in threshold)
            + "];"
        )
        eval_cmp = "            if (f[Feature[node]] <= Threshold[node])"

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
        eval_cmp,
        "                node = Left[node];",
        "            else",
        "                node = Right[node];",
        "        }",
        "    }",
        "",
    ]

    lines += _cs_predict_methods(mode, base_score, num_class)
    lines.append("}")

    _write_file(output, "\n".join(lines) + "\n")


def _cs_predict_methods(mode: str, base_score: float, _num_class: int) -> list[str]:
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
    return [
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    public static double[] PredictScores(ReadOnlySpan<double> f)",
        "    {",
        "        double[] s = new double[NumClasses];",
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
) -> None:
    split_features: list[int] = []
    split_borders: list[float] = []
    tree_split_offset: list[int] = []
    tree_depth: list[int] = []
    leaf_values: list[float] = []
    tree_leaf_offset: list[int] = []

    for t in trees:
        tree_split_offset.append(len(split_features))
        splits = t["splits"]
        tree_depth.append(len(splits))
        for s in splits:
            split_features.append(int(s["float_feature_index"]))
            split_borders.append(float(s["border"]))
        tree_leaf_offset.append(len(leaf_values))
        for v in t["leaf_values"]:
            leaf_values.append(float(v))

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
        "    // bit l set when feature[split[l]] >= border[split[l]]",
        "    [MethodImpl(MethodImplOptions.AggressiveInlining)]",
        "    private static double EvalTree(int treeIdx, ReadOnlySpan<double> f)",
        "    {",
        "        int leafIdx     = 0;",
        "        int splitOffset = TreeSplitOffset[treeIdx];",
        "        int depth       = TreeDepth[treeIdx];",
        "        for (int l = 0; l < depth; l++)",
        "        {",
        "            int s = splitOffset + l;",
        "            if (f[SplitFeature[s]] >= SplitBorder[s])",
        "                leafIdx |= (1 << l);",
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
) -> None:
    pkg = class_name.lower()
    needs_math = mode in ("binary", "multiclass")

    feature: list[int] = []
    threshold: list[float] = []
    left_arr: list[int] = []
    right_arr: list[int] = []
    value: list[float] = []

    for n in nodes:
        if n["leaf"]:
            feature.append(0)
            threshold.append(0.0)
            left_arr.append(-1)
            right_arr.append(0)
            value.append(n["value"])
        else:
            feature.append(n["feature"])
            threshold.append(n["threshold"])
            left_arr.append(n["left"])
            right_arr.append(n["right"])
            value.append(0.0)

    if xgboost:
        thr_type = "float32"
        thr_vals = ", ".join(go_f32(v) for v in threshold)
        eval_cmp = "\t\tif float32(f[feature[node]]) < threshold[node] {"
    else:
        thr_type = "float64"
        thr_vals = ", ".join(go_f64(v) for v in threshold)
        eval_cmp = "\t\tif f[feature[node]] <= threshold[node] {"

    lines: list[str] = [f"package {pkg}", ""]

    if needs_math:
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

    # var block
    lines += [
        "//nolint:gochecknoglobals",
        "var (",
        "\tfeature   = []int{" + ", ".join(map(str, feature)) + "}",
        "\tthreshold = []" + thr_type + "{" + thr_vals + "}",
        "\tleft      = []int{" + ", ".join(map(str, left_arr)) + "}",
        "\tright     = []int{" + ", ".join(map(str, right_arr)) + "}",
        "\tvalue     = []float64{" + ", ".join(go_f64(v) for v in value) + "}",
        "\troots     = []int{" + ", ".join(map(str, roots)) + "}",
        ")",
        "",
    ]

    # eval function
    lines += [
        "// left[node] == -1 means leaf node",
        "func eval(node int, f []float64) float64 {",
        "\tfor {",
        "\t\tif left[node] < 0 {",
        "\t\t\treturn value[node]",
        "\t\t}",
        eval_cmp,
        "\t\t\tnode = left[node]",
        "\t\t} else {",
        "\t\t\tnode = right[node]",
        "\t\t}",
        "\t}",
        "}",
        "",
    ]

    lines += _go_predict_methods(mode, base_score, num_class)
    _write_file(output, "\n".join(lines) + "\n")


def _write_go_catboost(
    trees: list[Any],
    scale: float,
    bias: float,
    class_name: str,
    output: str,
    mode: str = "regression",
    _num_class: int = 1,
) -> None:
    pkg = class_name.lower()
    needs_math = mode in ("binary", "multiclass")

    split_features: list[int] = []
    split_borders: list[float] = []
    tree_split_offset: list[int] = []
    tree_depth: list[int] = []
    leaf_values: list[float] = []
    tree_leaf_offset: list[int] = []

    for t in trees:
        tree_split_offset.append(len(split_features))
        splits = t["splits"]
        tree_depth.append(len(splits))
        for s in splits:
            split_features.append(int(s["float_feature_index"]))
            split_borders.append(float(s["border"]))
        tree_leaf_offset.append(len(leaf_values))
        for v in t["leaf_values"]:
            leaf_values.append(float(v))

    lines: list[str] = [f"package {pkg}", ""]

    if needs_math:
        lines += ['import "math"', ""]

    lines += [
        "const (",
        f"\ttreeCount = {len(trees)}",
        f"\tscale     = {go_f64(scale)}",
        f"\tbias      = {go_f64(bias)}",
        ")",
        "",
        "//nolint:gochecknoglobals",
        "var (",
        "\tsplitFeature    = []int{" + ", ".join(map(str, split_features)) + "}",
        "\tsplitBorder     = []float64{"
        + ", ".join(go_f64(v) for v in split_borders)
        + "}",
        "\ttreeSplitOffset = []int{" + ", ".join(map(str, tree_split_offset)) + "}",
        "\ttreeDepth       = []int{" + ", ".join(map(str, tree_depth)) + "}",
        "\tleafValues      = []float64{"
        + ", ".join(go_f64(v) for v in leaf_values)
        + "}",
        "\ttreeLeafOffset  = []int{" + ", ".join(map(str, tree_leaf_offset)) + "}",
        ")",
        "",
        "// bit l set when feature[split[l]] >= border[split[l]]",
        "func evalTree(treeIdx int, f []float64) float64 {",
        "\tleafIdx := 0",
        "\tsplitOffset := treeSplitOffset[treeIdx]",
        "\tdepth := treeDepth[treeIdx]",
        "\tfor l := 0; l < depth; l++ {",
        "\t\ts := splitOffset + l",
        "\t\tif f[splitFeature[s]] >= splitBorder[s] {",
        "\t\t\tleafIdx |= (1 << l)",
        "\t\t}",
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


def _go_predict_methods(mode: str, base_score: float, _num_class: int) -> list[str]:
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
    return [
        "// PredictScores returns raw per-class scores.",
        "func PredictScores(f []float64) []float64 {",
        "\ts := make([]float64, numClasses)",
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
        "\t\tif v > max {",
        "\t\t\tmax = v",
        "\t\t}",
        "\t}",
        "\tsum := 0.0",
        "\tfor i := range s {",
        "\t\ts[i] = math.Exp(s[i] - max)",
        "\t\tsum += s[i]",
        "\t}",
        "\tfor i := range s {",
        "\t\ts[i] /= sum",
        "\t}",
        "\treturn s",
        "}",
        "",
        "// PredictClass returns the class index with the highest probability.",
        "func PredictClass(f []float64) int {",
        "\tp := PredictScores(f)",
        "\tbest := 0",
        "\tfor i := range p {",
        "\t\tif p[i] > p[best] {",
        "\t\t\tbest = i",
        "\t\t}",
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
    # LF newlines so output is identical on Windows and Unix (gofmt requires LF)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
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
