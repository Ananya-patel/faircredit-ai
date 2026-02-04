from sklearn.model_selection import train_test_split

from src.data_pipeline.collectors import AlternativeDataCollector, DataGenerationConfig
from src.data_pipeline.feature_engineers import AlternativeFeatureEngineer
from src.data_pipeline.cleaners import DataCleaner
from src.data_pipeline.label_generator import CreditLabelGenerator
from src.models.fair_classifiers import FairnessConstrainedClassifier
from src.evaluation.bias_detectors import BiasDetector


def main() -> None:
    # Pipeline
    collector = AlternativeDataCollector(DataGenerationConfig())
    raw_df = collector.collect()

    engineer = AlternativeFeatureEngineer()
    features_df = engineer.transform(raw_df)

    cleaner = DataCleaner()
    X_clean = cleaner.clean(features_df)

    labeler = CreditLabelGenerator()
    X, y = labeler.generate(X_clean)

    sensitive_attr = (X["network_creditworthiness"] > 0.5).astype(int)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_attr, test_size=0.25, random_state=42, stratify=y
    )

    model = FairnessConstrainedClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    detector = BiasDetector()
    results = detector.run(model, X_test, y_pred, s_test)

    print("BIAS DETECTION RESULTS")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
