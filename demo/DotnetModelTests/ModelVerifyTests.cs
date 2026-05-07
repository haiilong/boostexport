using Xunit.Abstractions;

namespace DotnetModelTests;

public class ModelVerifyTests(ITestOutputHelper output)
{
    private const double Tolerance = 1e-4;

    private static double[][] LoadFeatures(string path)
    {
        var lines = File.ReadAllLines(path);
        return [.. lines.Skip(1).Select(l => l.Split(',').Take(10).Select(double.Parse).ToArray())];

    }

    private static double[] LoadColumn(string path, string col)
    {
        var lines = File.ReadAllLines(path);
        var header = lines[0].Split(',');
        int idx = Array.IndexOf(header, col);
        return [.. lines.Skip(1).Select(l => double.Parse(l.Split(',')[idx]))];
    }

    private void Verify(string label, double[] expected, Func<double[], double> predict, double[][] X)
    {
        double maxDiff = 0;
        for (int i = 0; i < X.Length; i++)
        {
            double actual = predict(X[i]);
            double diff = Math.Abs(actual - expected[i]);
            if (diff > maxDiff) maxDiff = diff;
            Assert.True(diff < Tolerance,
                $"{label} row {i}: expected={expected[i]:G17}, actual={actual:G17}, diff={diff:G6}");
        }
        output.WriteLine($"{label}: {X.Length} rows OK, max diff = {maxDiff:G6}");
    }

    [Fact]
    public void LgbmModel_MatchesPython()
    {
        var X = LoadFeatures("data.csv");
        var expected = LoadColumn("predictions_train.csv", "pred_lgb");
        Verify("LgbmModel", expected, f => LgbmModel.Predict(f), X);
    }

    [Fact]
    public void XgbModel_MatchesPython()
    {
        var X = LoadFeatures("data.csv");
        var expected = LoadColumn("predictions_train.csv", "pred_xgb");
        Verify("XgbModel", expected, f => XgbModel.Predict(f), X);
    }

    [Fact]
    public void CbModel_MatchesPython()
    {
        var X = LoadFeatures("data.csv");
        var expected = LoadColumn("predictions_train.csv", "pred_cb");
        Verify("CbModel", expected, f => CbModel.Predict(f), X);
    }
}
