using System.Globalization;
using Xunit.Abstractions;

namespace DotnetModelTests;

public class ModelVerifyTests(ITestOutputHelper output)
{
    // LightGBM and CatBoost evaluate in float64 end-to-end, so the generated
    // code matches Python almost exactly. XGBoost accumulates leaf values in
    // float32 internally, so its tolerance is wider.
    private const double TolExact = 1e-9;
    private const double TolXgb = 1e-4;
    private const double TolProba = 1e-6;

    private const int FeatureCount = 10;

    private static double Parse(string s) =>
        double.Parse(s, CultureInfo.InvariantCulture); // handles "NaN" too

    private static double[][] LoadFeatures(string path)
    {
        var lines = File.ReadAllLines(path);
        return [.. lines.Skip(1).Select(l => l.Split(',').Take(FeatureCount).Select(Parse).ToArray())];
    }

    private static double[] LoadColumn(string path, string col)
    {
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        int idx = Array.IndexOf(header, col);
        Assert.True(idx >= 0, $"column {col} not found in {path}");
        return [.. lines.Skip(1).Select(l => Parse(l.Split(',')[idx]))];
    }

    private void Verify(string label, double[] expected, Func<double[], double> predict,
        double[][] X, double tolerance)
    {
        double maxDiff = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double actual = predict(X[i]);
            double diff = Math.Abs(actual - expected[i]);
            if (diff > maxDiff) maxDiff = diff;
            Assert.True(diff < tolerance,
                $"{label} row {i}: expected={expected[i]:G17}, actual={actual:G17}, diff={diff:G6}");
        }
        output.WriteLine($"{label}: {X.Length} rows OK, max diff = {maxDiff:G6}");
    }

    // ------------------------------------------------------------------
    // Regression
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void Regression_MatchesPython(string lib)
    {
        var X = LoadFeatures("data.csv");
        var expected = LoadColumn("predictions_train.csv", $"pred_{lib}");
        Func<double[], double> predict = lib switch
        {
            "lgb" => f => LgbmModel.Predict(f),
            "xgb" => f => XgbModel.Predict(f),
            _ => f => CbModel.Predict(f),
        };
        Verify($"regression/{lib}", expected, predict, X, lib == "xgb" ? TolXgb : TolExact);
    }

    // ------------------------------------------------------------------
    // Binary classification: raw score and probability
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void BinaryScore_MatchesPython(string lib)
    {
        var X = LoadFeatures("data_binary.csv");
        var expected = LoadColumn("predictions_binary.csv", $"score_{lib}");
        Func<double[], double> predict = lib switch
        {
            "lgb" => f => LgbmBinModel.PredictScore(f),
            "xgb" => f => XgbBinModel.PredictScore(f),
            _ => f => CbBinModel.PredictScore(f),
        };
        Verify($"binary-score/{lib}", expected, predict, X, lib == "xgb" ? TolXgb : TolExact);
    }

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void BinaryProbability_MatchesPython(string lib)
    {
        var X = LoadFeatures("data_binary.csv");
        var expected = LoadColumn("predictions_binary.csv", $"proba_{lib}");
        Func<double[], double> predict = lib switch
        {
            "lgb" => f => LgbmBinModel.PredictProbability(f),
            "xgb" => f => XgbBinModel.PredictProbability(f),
            _ => f => CbBinModel.PredictProbability(f),
        };
        Verify($"binary-proba/{lib}", expected, predict, X, TolProba);
    }

    // ------------------------------------------------------------------
    // Multiclass classification: per-class probabilities + argmax label
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void MulticlassProbabilities_MatchPython(string lib)
    {
        var X = LoadFeatures("data_multi.csv");
        Func<double[], double[]> predict = lib switch
        {
            "lgb" => f => LgbmMultiModel.Predict(f),
            "xgb" => f => XgbMultiModel.Predict(f),
            _ => f => CbMultiModel.Predict(f),
        };
        for (int c = 0; c < 3; c++)
        {
            int cc = c;
            var expected = LoadColumn("predictions_multi.csv", $"proba_{lib}_{c}");
            Verify($"multi-proba{c}/{lib}", expected, f => predict(f)[cc], X, TolProba);
        }
    }

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void MulticlassLabel_MatchesArgmax(string lib)
    {
        var X = LoadFeatures("data_multi.csv");
        Func<double[], double[]> proba = lib switch
        {
            "lgb" => f => LgbmMultiModel.Predict(f),
            "xgb" => f => XgbMultiModel.Predict(f),
            _ => f => CbMultiModel.Predict(f),
        };
        Func<double[], int> label = lib switch
        {
            "lgb" => f => LgbmMultiModel.PredictClass(f),
            "xgb" => f => XgbMultiModel.PredictClass(f),
            _ => f => CbMultiModel.PredictClass(f),
        };
        foreach (var row in X)
        {
            var p = proba(row);
            int best = 0;
            for (int i = 1; i < p.Length; i++) if (p[i] > p[best]) best = i;
            Assert.Equal(best, label(row));
        }
    }

    // ------------------------------------------------------------------
    // Regression with missing values (NaN routing)
    // ------------------------------------------------------------------

    [Theory]
    [InlineData("lgb")]
    [InlineData("xgb")]
    [InlineData("cb")]
    public void NanRegression_MatchesPython(string lib)
    {
        var X = LoadFeatures("data_nan.csv");
        Assert.Contains(X, row => row.Any(double.IsNaN)); // data really has NaNs
        var expected = LoadColumn("predictions_nan.csv", $"pred_{lib}");
        Func<double[], double> predict = lib switch
        {
            "lgb" => f => LgbmNanModel.Predict(f),
            "xgb" => f => XgbNanModel.Predict(f),
            _ => f => CbNanModel.Predict(f),
        };
        Verify($"nan/{lib}", expected, predict, X, lib == "xgb" ? TolXgb : TolExact);
    }
}
