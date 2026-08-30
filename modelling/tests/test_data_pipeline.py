import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from modelling.data_pipeline import (
    prepare_experiment_frame,
    validate_experiment_frame,
)
from modelling.sentiment import aggregate_monthly_sentiment
from modelling.sentiment import add_sentiment_lags


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-01", periods=14, freq="MS")
        self.macro = pd.DataFrame({
            "Date": self.dates,
            "CPIAUCSL": np.linspace(250.0, 263.0, len(self.dates)),
            "UNRATE": np.linspace(4.0, 5.3, len(self.dates)),
            "FEDFUNDS": np.linspace(1.5, 0.2, len(self.dates)),
            "INDPRO": np.linspace(100.0, 106.5, len(self.dates)),
            "MCOILWTICO": np.linspace(55.0, 68.0, len(self.dates)),
        })

    def test_article_aggregation_preserves_source_and_distribution(self):
        articles = pd.DataFrame({
            "article_id": [1, 2, 3],
            "news_date": ["2020-01-02", "2020-01-20", "2020-02-01"],
            "sentiment_score": [-0.8, 0.4, 0.1],
            "sentiment_label": ["negative", "positive", "neutral"],
        })
        monthly = aggregate_monthly_sentiment(articles, source_code="NYT")

        self.assertEqual(monthly.loc[0, "News_Source"], "NYT")
        self.assertAlmostEqual(monthly.loc[0, "News_Sentiment"], -0.2)
        self.assertAlmostEqual(monthly.loc[0, "Negative_Share"], 0.5)
        self.assertEqual(monthly.loc[0, "Article_Count"], 2)

    def test_prepare_frame_creates_lags_without_lookahead(self):
        news = pd.DataFrame({
            "Date": self.dates,
            "News_Source": "NYT",
            "News_Sentiment": np.arange(len(self.dates), dtype=float),
            "Sentiment_Median": 0.0,
            "Sentiment_Std": 0.1,
            "Negative_Share": 0.5,
            "Neutral_Share": 0.1,
            "Positive_Share": 0.4,
            "Article_Count": 10,
        })
        prepared = prepare_experiment_frame(self.macro, news, source_code="NYT")

        self.assertTrue(pd.isna(prepared.loc[0, "News_Sentiment_Lag1"]))
        self.assertEqual(
            prepared.loc[4, "News_Sentiment_Lag1"],
            prepared.loc[3, "News_Sentiment"],
        )
        self.assertEqual(prepared.loc[0, "Experiment_News_Source"], "NYT")

    def test_validation_reports_complete_monthly_coverage(self):
        news = pd.DataFrame({
            "Date": self.dates,
            "News_Source": "NYT",
            "News_Sentiment": 0.0,
            "Sentiment_Median": 0.0,
            "Sentiment_Std": 0.1,
            "Negative_Share": 0.5,
            "Neutral_Share": 0.1,
            "Positive_Share": 0.4,
            "Article_Count": 10,
        })
        prepared = prepare_experiment_frame(self.macro, news, source_code="NYT")
        coverage = validate_experiment_frame(prepared)

        self.assertEqual(coverage["row_count"], 14)
        self.assertEqual(coverage["missing_calendar_months"], [])
        self.assertEqual(coverage["months_with_sentiment"], 14)

    def test_sentiment_lags_use_only_previous_months(self):
        data = pd.DataFrame({
            "News_Sentiment": [0.1, 0.2, -0.3],
        })
        lagged, features = add_sentiment_lags(data, lags=2)

        self.assertEqual(features, ["Sentiment_Lag1", "Sentiment_Lag2"])
        self.assertTrue(pd.isna(lagged.loc[0, "Sentiment_Lag1"]))
        self.assertEqual(lagged.loc[2, "Sentiment_Lag1"], 0.2)
        self.assertEqual(lagged.loc[2, "Sentiment_Lag2"], 0.1)


if __name__ == "__main__":
    unittest.main()
