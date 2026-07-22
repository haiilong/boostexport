package gomodeltests_test

import (
	"encoding/csv"
	"math"
	"os"
	"strconv"
	"testing"

	"gomodeltests/cb"
	"gomodeltests/cbbin"
	"gomodeltests/cbmulti"
	"gomodeltests/cbnan"
	"gomodeltests/lgbm"
	"gomodeltests/lgbmbin"
	"gomodeltests/lgbmmulti"
	"gomodeltests/lgbmnan"
	"gomodeltests/xgb"
	"gomodeltests/xgbbin"
	"gomodeltests/xgbmulti"
	"gomodeltests/xgbnan"
)

// LightGBM and CatBoost evaluate in float64 end-to-end, so the generated
// code matches Python almost exactly. XGBoost accumulates leaf values in
// float32 internally, so its tolerance is wider.
const (
	tolExact     = 1e-9
	tolXgb       = 1e-4
	tolProba     = 1e-6
	featureCount = 10
)

func loadCSV(t *testing.T, path string) ([][]float64, map[string]int) {
	t.Helper()
	fh, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer fh.Close()
	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	cols := make(map[string]int, len(rows[0]))
	for i, h := range rows[0] {
		cols[h] = i
	}
	out := make([][]float64, len(rows)-1)
	for i, r := range rows[1:] {
		out[i] = make([]float64, len(r))
		for j, s := range r {
			v, err := strconv.ParseFloat(s, 64) // "NaN" parses to NaN
			if err != nil {
				t.Fatalf("%s row %d col %d: %v", path, i, j, err)
			}
			out[i][j] = v
		}
	}
	return out, cols
}

func features(rows [][]float64) [][]float64 {
	out := make([][]float64, len(rows))
	for i, r := range rows {
		out[i] = r[:featureCount]
	}
	return out
}

func verify(t *testing.T, label string, X [][]float64, expected [][]float64,
	col int, predict func([]float64) float64, tol float64) {
	t.Helper()
	maxDiff := 0.0
	for i, f := range X {
		diff := math.Abs(predict(f) - expected[i][col])
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			t.Errorf("%s row %d: got %.10g, want %.10g, diff %.2e",
				label, i, predict(f), expected[i][col], diff)
		}
	}
	t.Logf("%s: %d rows OK, max diff %.2e", label, len(X), maxDiff)
}

// ------------------------------------------------------------------
// Regression
// ------------------------------------------------------------------

func TestRegression(t *testing.T) {
	data, _ := loadCSV(t, "../data.csv")
	preds, cols := loadCSV(t, "../predictions_train.csv")
	X := features(data)
	verify(t, "regression/lgb", X, preds, cols["pred_lgb"], lgbm.Predict, tolExact)
	verify(t, "regression/xgb", X, preds, cols["pred_xgb"], xgb.Predict, tolXgb)
	verify(t, "regression/cb", X, preds, cols["pred_cb"], cb.Predict, tolExact)
}

// ------------------------------------------------------------------
// Binary classification: raw score and probability
// ------------------------------------------------------------------

func TestBinaryScore(t *testing.T) {
	data, _ := loadCSV(t, "../data_binary.csv")
	preds, cols := loadCSV(t, "../predictions_binary.csv")
	X := features(data)
	verify(t, "binary-score/lgb", X, preds, cols["score_lgb"], lgbmbin.PredictScore, tolExact)
	verify(t, "binary-score/xgb", X, preds, cols["score_xgb"], xgbbin.PredictScore, tolXgb)
	verify(t, "binary-score/cb", X, preds, cols["score_cb"], cbbin.PredictScore, tolExact)
}

func TestBinaryProbability(t *testing.T) {
	data, _ := loadCSV(t, "../data_binary.csv")
	preds, cols := loadCSV(t, "../predictions_binary.csv")
	X := features(data)
	verify(t, "binary-proba/lgb", X, preds, cols["proba_lgb"], lgbmbin.PredictProbability, tolProba)
	verify(t, "binary-proba/xgb", X, preds, cols["proba_xgb"], xgbbin.PredictProbability, tolProba)
	verify(t, "binary-proba/cb", X, preds, cols["proba_cb"], cbbin.PredictProbability, tolProba)
}

// ------------------------------------------------------------------
// Multiclass classification: per-class probabilities + argmax label
// ------------------------------------------------------------------

func TestMulticlassProbabilities(t *testing.T) {
	data, _ := loadCSV(t, "../data_multi.csv")
	preds, cols := loadCSV(t, "../predictions_multi.csv")
	X := features(data)
	models := map[string]func([]float64) []float64{
		"lgb": lgbmmulti.Predict,
		"xgb": xgbmulti.Predict,
		"cb":  cbmulti.Predict,
	}
	for lib, predict := range models {
		for c := 0; c < 3; c++ {
			cc := c
			verify(t, "multi-proba/"+lib, X, preds, cols["proba_"+lib+"_"+strconv.Itoa(c)],
				func(f []float64) float64 { return predict(f)[cc] }, tolProba)
		}
	}
}

func TestMulticlassLabel(t *testing.T) {
	data, _ := loadCSV(t, "../data_multi.csv")
	X := features(data)
	models := map[string]struct {
		proba func([]float64) []float64
		label func([]float64) int
	}{
		"lgb": {lgbmmulti.Predict, lgbmmulti.PredictClass},
		"xgb": {xgbmulti.Predict, xgbmulti.PredictClass},
		"cb":  {cbmulti.Predict, cbmulti.PredictClass},
	}
	for lib, m := range models {
		for i, f := range X {
			p := m.proba(f)
			best := 0
			for j := range p {
				if p[j] > p[best] {
					best = j
				}
			}
			if got := m.label(f); got != best {
				t.Errorf("%s row %d: PredictClass=%d, argmax=%d", lib, i, got, best)
			}
		}
	}
}

// ------------------------------------------------------------------
// Regression with missing values (NaN routing)
// ------------------------------------------------------------------

func TestNanRegression(t *testing.T) {
	data, _ := loadCSV(t, "../data_nan.csv")
	preds, cols := loadCSV(t, "../predictions_nan.csv")
	X := features(data)

	hasNaN := false
	for _, r := range X {
		for _, v := range r {
			if math.IsNaN(v) {
				hasNaN = true
			}
		}
	}
	if !hasNaN {
		t.Fatal("data_nan.csv contains no NaN values; test would be vacuous")
	}

	verify(t, "nan/lgb", X, preds, cols["pred_lgb"], lgbmnan.Predict, tolExact)
	verify(t, "nan/xgb", X, preds, cols["pred_xgb"], xgbnan.Predict, tolXgb)
	verify(t, "nan/cb", X, preds, cols["pred_cb"], cbnan.Predict, tolExact)
}
