package gomodeltests_test

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"

	"gomodeltests/cb"
	"gomodeltests/lgbm"
	"gomodeltests/xgb"
)

const (
	dataFile  = "../data.csv"
	predFile  = "../predictions_train.csv"
	tolerance = 1e-4
)

type row struct {
	features []float64
	predLgb  float64
	predXgb  float64
	predCb   float64
}

func loadData(t *testing.T) []row {
	t.Helper()

	fh, err := os.Open(dataFile)
	if err != nil {
		t.Fatalf("open data.csv: %v", err)
	}
	defer fh.Close()
	dataRows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		t.Fatalf("read data.csv: %v", err)
	}

	ph, err := os.Open(predFile)
	if err != nil {
		t.Fatalf("open predictions_train.csv: %v", err)
	}
	defer ph.Close()
	predRows, err := csv.NewReader(ph).ReadAll()
	if err != nil {
		t.Fatalf("read predictions_train.csv: %v", err)
	}

	// data.csv header: f0, f1, ..., f9, label  (features cols 0-9)
	// predictions_train.csv header: label, pred_xgb, pred_lgb, pred_cb
	var rows []row
	for i := 1; i < len(dataRows); i++ {
		dr := dataRows[i]
		pr := predRows[i]

		feats := make([]float64, 10)
		for j := 0; j < 10; j++ {
			feats[j], _ = strconv.ParseFloat(dr[j], 64)
		}
		predXgb, _ := strconv.ParseFloat(pr[1], 64)
		predLgb, _ := strconv.ParseFloat(pr[2], 64)
		predCb, _ := strconv.ParseFloat(pr[3], 64)

		rows = append(rows, row{feats, predLgb, predXgb, predCb})
	}
	return rows
}

func TestLgbmModel_MatchesPython(t *testing.T) {
	rows := loadData(t)
	maxDiff := 0.0
	for i, r := range rows {
		got := lgbm.Predict(r.features)
		diff := math.Abs(got - r.predLgb)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			t.Errorf("row %d: got %.6f, want %.6f, diff %.2e", i, got, r.predLgb, diff)
		}
	}
	fmt.Printf("LgbmModel  max diff: %.2e  (%d rows)\n", maxDiff, len(rows))
}

func TestXgbModel_MatchesPython(t *testing.T) {
	rows := loadData(t)
	maxDiff := 0.0
	for i, r := range rows {
		got := xgb.Predict(r.features)
		diff := math.Abs(got - r.predXgb)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			t.Errorf("row %d: got %.6f, want %.6f, diff %.2e", i, got, r.predXgb, diff)
		}
	}
	fmt.Printf("XgbModel   max diff: %.2e  (%d rows)\n", maxDiff, len(rows))
}

func TestCbModel_MatchesPython(t *testing.T) {
	rows := loadData(t)
	maxDiff := 0.0
	for i, r := range rows {
		got := cb.Predict(r.features)
		diff := math.Abs(got - r.predCb)
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tolerance {
			t.Errorf("row %d: got %.6f, want %.6f, diff %.2e", i, got, r.predCb, diff)
		}
	}
	fmt.Printf("CbModel    max diff: %.2e  (%d rows)\n", maxDiff, len(rows))
}
