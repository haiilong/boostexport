# boostexport

Export trained XGBoost, LightGBM, or CatBoost models into self-contained static classes for native inference in C# or Go - no Python runtime, no model files, no extra dependencies at prediction time.

The generated code uses a data-array evaluator (feature/threshold/left/right arrays + a small traversal loop) rather than giant `if/else` trees. It is exact: outputs match the original Python model to within floating-point rounding, including missing-value (NaN) routing.

Background reading: this project grew out of the blog series
[Building an ML Inference API](https://haiilong.github.io/blog/building-ml-inference-part-4),
which walks through the "trees are already if/else" idea and why the
data-array form beats generated branches.

---

## How it works

Each boosting model is a sum of decision trees. The exporter reads the trained model, flattens every tree into parallel arrays, and writes a static class with a `Predict` method that walks those arrays at runtime.

| Library   | Format  | Tree type              | Comparison                                  |
|-----------|---------|------------------------|---------------------------------------------|
| LightGBM  | `.txt`  | Asymmetric             | `double <=`                                 |
| XGBoost   | `.json` | Asymmetric             | `float <` (DMatrix uses float32 internally) |
| CatBoost  | `.cbm`  | Symmetric (oblivious)  | `double >` (strictly greater than border)   |

Supported tasks: **regression**, **binary classification**, **multiclass classification** - for all three libraries.

### Missing values (NaN)

The generated code reproduces each library's missing-value routing exactly:

- **LightGBM** - honours each split's `missing_type` (`None`: NaN is treated
  as `0.0`; `Zero`: NaN and `|v| <= 1e-35` follow the trained default
  direction; `NaN`: NaN follows the default direction).
- **XGBoost** - NaN follows each split's trained `default_left` direction.
- **CatBoost** - NaN follows the feature's `nan_value_treatment`
  (`AsFalse` / `AsTrue`, from `nan_mode="Min"` / `"Max"`).

### Supported objectives

The exporter reproduces the output transform for these objectives:

| Library  | Raw score (regression/ranking) | Sigmoid (binary) | Softmax (multiclass) |
|----------|--------------------------------|------------------|----------------------|
| LightGBM | `regression`, `regression_l1`, `huber`, `fair`, `quantile`, `mape`, `lambdarank`, `rank_xendcg` | `binary` | `multiclass` |
| XGBoost  | `reg:squarederror`, `reg:squaredlogerror`, `reg:absoluteerror`, `reg:quantileerror`, `reg:pseudohubererror`, `rank:*` | `binary:logistic`, `reg:logistic` | `multi:softprob`, `multi:softmax` |
| CatBoost | `RMSE`, `MAE`, `Quantile`, `MAPE`, `Huber`, `Expectile` | `Logloss`, `CrossEntropy` | `MultiClass` |

Objectives that need an output transform the generated code does not
implement - e.g. Poisson/Gamma/Tweedie (`exp()` link), LightGBM
`multiclassova`, XGBoost `binary:logitraw`, CatBoost `MultiClassOneVsAll` -
are **rejected at export time** with an explanation, rather than silently
producing wrong numbers. Unknown objectives export with a warning and return
raw tree sums; verify against the original model before deploying.

The exporter also fails fast on model features it cannot reproduce:
categorical splits (all three libraries), LightGBM linear trees
(`linear_tree=true`), non-`gbtree` XGBoost boosters, and CatBoost
non-symmetric grow policies.

---

## Dependencies

### Python (exporter)

```
pip install xgboost lightgbm catboost scikit-learn numpy pandas
```

### C# (generated code)

The generated code targets .NET 8 or later (C# 12 collection expressions and
implicit global usings). The demo test project targets .NET 10.

Download from <https://aka.ms/dotnet/download>.

### Go (generated code)

Any supported Go release. Generated files are `gofmt`-clean and have no
external dependencies.

Download from <https://go.dev/dl/>.

---

## Usage

### Interactive

```bash
python export_model.py
```

Prompts for:
1. Model file path - library inferred from extension (`.json` / `.txt` / `.cbm`), or asked if unknown
2. Output language - `cs` (default) or `go`
3. Class/package name - defaults to `Model` (C#) or `model` (Go)
4. Output file - defaults to `<Name>.cs` or `<Name>.go`

### CLI - C#

```bash
python export_model.py model_lgb.txt  -c LgbmModel -o LgbmModel.cs
python export_model.py model_xgb.json -c XgbModel  -o XgbModel.cs
python export_model.py model_cb.cbm   -c CbModel   -o CbModel.cs
```

### CLI - Go

```bash
python export_model.py model_lgb.txt  -c lgbm -o lgbm/model.go --lang go
python export_model.py model_xgb.json -c xgb  -o xgb/model.go  --lang go
python export_model.py model_cb.cbm   -c cb   -o cb/model.go   --lang go
```

### Generated API - C#

**Regression**
```csharp
double prediction = MyModel.Predict(features);
```

**Binary classification**
```csharp
double score       = MyModel.PredictScore(features);       // raw logit
double probability = MyModel.PredictProbability(features); // sigmoid(score)
int    label       = MyModel.Predict(features);            // 0 or 1
```

**Multiclass classification**
```csharp
double[] scores        = MyModel.PredictScores(features); // raw per-class
double[] probabilities = MyModel.Predict(features);       // softmax
int      label         = MyModel.PredictClass(features);
```

Drop the generated `.cs` file into any .NET project. No additional packages needed.

### Generated API - Go

**Regression**
```go
prediction := mypkg.Predict(features)
```

**Binary classification**
```go
score       := mypkg.PredictScore(features)       // raw logit
probability := mypkg.PredictProbability(features) // sigmoid(score)
label       := mypkg.Predict(features)            // 0 or 1
```

**Multiclass classification**
```go
scores        := mypkg.PredictScores(features) // raw per-class
probabilities := mypkg.Predict(features)       // softmax []float64
label         := mypkg.PredictClass(features)
```

Each model is its own package. Drop the generated `.go` file into any module. No external dependencies needed.

---

## Demo / test suite

The `demo/` folder contains an end-to-end verification matrix: regression,
binary, multiclass, and NaN-regression models for all three libraries
(12 models total), exported to both languages and compared row-by-row
against the Python predictions. The same pipeline runs in GitHub Actions
on every push, retraining from scratch so new library releases that change
the model formats are caught early.

### 1 - Train the model matrix

```bash
cd demo
python create_models.py
```

Creates 12 models, 4 datasets (`data*.csv`), and 4 expected-prediction files
(`predictions*.csv`).

### 2 - Verify Python predictions reload correctly

```bash
python demo_predict.py
```

### 3 - Export all models

```bash
python generate_cs.py   # -> DotnetModelTests/*.cs
python generate_go.py   # -> GoModelTests/<pkg>/model.go
```

### 4 - Verify C# and Go match Python

```bash
cd ..
dotnet test demo/DotnetModelTests
```

```bash
cd demo/GoModelTests
go test ./...
gofmt -l .   # generated code is gofmt-clean
```

The C# suite runs 18 tests (regression, binary score + probability,
multiclass probabilities + labels, NaN routing); the Go suite covers the
same matrix.

---

## Repo layout

```
boostexport/
  export_model.py            # the exporter - main tool
  README.md
  .github/workflows/ci.yml   # train -> export -> dotnet test + go test
  demo/
    create_models.py         # train the 12-model matrix
    demo_predict.py          # verify Python reload
    generate_cs.py           # export all models to C#
    generate_go.py           # export all models to Go
    model_*.{json,txt,cbm}   # saved models (regression, _bin, _multi, _nan)
    data*.csv                # datasets
    predictions*.csv         # expected outputs from Python
    DotnetModelTests/
      DotnetModelTests.csproj
      ModelVerifyTests.cs    # xunit: full matrix vs Python predictions
      *Model.cs              # generated
    GoModelTests/
      go.mod
      models_test.go         # go test: full matrix vs Python predictions
      <pkg>/model.go         # generated (one package per model)
```

---

## Notes

- **XGBoost float32 precision**: XGBoost converts features to `float32`
  internally (via `DMatrix`) and also accumulates leaf values in float32.
  The generated code casts features to `float`/`float32` before threshold
  comparisons but sums in float64, so XGBoost outputs match to about `1e-5`
  rather than exactly. LightGBM and CatBoost evaluate in float64 end-to-end
  and match to within `1e-9` (usually bit-exact).
- **XGBoost base_score**: stored in the objective's output space; the
  exporter applies the logit transform for logistic objectives and per-class
  base margins for multiclass (XGBoost >= 2.0 auto-estimates these from the
  training labels, so they are rarely the neutral defaults).
- **Feature vector**: pass features in training order. `NaN` is the missing
  value; there is no length check in the generated code, so a short vector
  throws an index error and a long one is silently truncated by unused slots.
- **Categorical features**: not supported in any library - the exporter
  rejects such models at export time. Encode categoricals numerically before
  training if you want to export.
- **Line endings**: LightGBM `.txt` model files must keep LF endings
  (enforced via `.gitattributes`); LightGBM cannot parse CRLF model files.
- **Tolerances**: the demo suites assert `1e-9` for LightGBM/CatBoost,
  `1e-4` for XGBoost, `1e-6` for probabilities. Verify with your own data
  before deploying.
