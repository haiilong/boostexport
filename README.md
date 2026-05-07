# boostexport

Export trained XGBoost, LightGBM, or CatBoost models into self-contained static classes for native inference in C# or Go - no Python runtime, no model files, no extra dependencies at prediction time.

The generated code uses a data-array evaluator (feature/threshold/left/right arrays + a small traversal loop) rather than giant `if/else` trees. It is exact: outputs match the original Python model to within floating-point rounding.

---

## How it works

Each boosting model is a sum of decision trees. The exporter reads the trained model, flattens every tree into parallel arrays, and writes a static class with a `Predict` method that walks those arrays at runtime.

| Library   | Format  | Tree type              | Comparison                                  |
|-----------|---------|------------------------|---------------------------------------------|
| LightGBM  | `.txt`  | Asymmetric             | `double <=`                                 |
| XGBoost   | `.json` | Asymmetric             | `float <` (DMatrix uses float32 internally) |
| CatBoost  | `.cbm`  | Symmetric (oblivious)  | `double >=`                                 |

Supported tasks: **regression**, **binary classification**, **multiclass classification**.

---

## Dependencies

### Python (exporter)

```
pip install xgboost lightgbm catboost scikit-learn numpy
```

### C# (generated code + verification)

.NET SDK 8 or later. Generated code uses C# 12 collection expressions and .NET 8 implicit GlobalUsings. If you use earlier version, you can manually fix your output codes (should be very straightforward).

Download from <https://aka.ms/dotnet/download>.

### Go (generated code + verification)

Any version of Go would do.

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
double[] probabilities = MyModel.Predict(features);      // softmax
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
probabilities := mypkg.Predict(features)      // softmax []float64
label         := mypkg.PredictClass(features)
```

Each model is its own package. Drop the generated `.go` file into any module. No external dependencies needed.

---

## Demo

The `demo/` folder contains a complete end-to-end example using regression on a synthetic dataset.

### 1 - Train and save models

```bash
cd demo
python create_models.py
```

Creates `model_xgb.json`, `model_lgb.txt`, `model_cb.cbm`, `data.csv`, `predictions_train.csv`.

### 2 - Verify Python predictions reload correctly

```bash
python demo_predict.py
```

### 3a - Export all three models to C#

```bash
python generate_cs.py
```

Writes `DotnetModelTests/LgbmModel.cs`, `DotnetModelTests/XgbModel.cs`, `DotnetModelTests/CbModel.cs`.

### 4a - Verify C# predictions match Python

```bash
cd ..
dotnet test demo/ModelTests
```

Expected output:

```
Passed ModelVerifyTests.LgbmModel_MatchesPython
Passed ModelVerifyTests.CbModel_MatchesPython
Passed ModelVerifyTests.XgbModel_MatchesPython
```

### 3b - Export all three models to Go

```bash
python generate_go.py
```

Writes `GoDotnetModelTests/lgbm/model.go`, `GoDotnetModelTests/xgb/model.go`, `GoDotnetModelTests/cb/model.go`.

### 4b - Verify Go predictions match Python

```bash
cd GoModelTests
go test ./...
```

Expected output:

```
LgbmModel  max diff: 0.00e+00  (200 rows)
XgbModel   max diff: 6.79e-05  (200 rows)
CbModel    max diff: 0.00e+00  (200 rows)
ok  gomodeltests
```

---

## Repo layout

```
boostexport/
  export_model.py          # the exporter - main tool
  README.md
  demo/
    create_models.py       # train regression models
    demo_predict.py        # verify Python reload
    generate_cs.py         # export all 3 models to C#
    generate_go.py         # export all 3 models to Go
    model_xgb.json         # saved XGBoost model
    model_lgb.txt          # saved LightGBM model
    model_cb.cbm           # saved CatBoost model
    data.csv               # 200-row synthetic dataset
    predictions_train.csv  # predictions from all 3 models
    DotnetModelTests/
      ModelTests.csproj
      ModelVerifyTests.cs  # xunit: 200 rows, tolerance 1e-4
      LgbmModel.cs         # generated
      XgbModel.cs          # generated
      CbModel.cs           # generated
    GoDotnetModelTests/
      go.mod
      models_test.go       # go test: 200 rows, tolerance 1e-4
      lgbm/model.go        # generated
      xgb/model.go         # generated
      cb/model.go          # generated
```

---

## Notes

- **XGBoost float32 precision**: XGBoost converts all features to `float32` internally (via `DMatrix`). The generated C# and Go code casts features to `float`/`float32` before threshold comparisons to match this behaviour exactly.
- **Binary/multiclass**: Tree traversal is identical to regression. The exporter adds sigmoid (binary) or softmax (multiclass) output transforms automatically when it detects the objective from the model file.
- **CatBoost multiclass**: Not yet supported. CatBoost multiclass uses vector-valued leaf nodes with a different JSON layout.
- **Tolerance**: Generated predictions match Python to within `1e-4` for regression (verified on 200 rows). Binary and multiclass outputs follow the same tree traversal so the same precision applies; verify with your own test data before deploying.
- **C# version**: Requires .NET 8+ for implicit global usings and collection expressions.
- **Go version**: No external dependencies. Each exported model is a self-contained package.
