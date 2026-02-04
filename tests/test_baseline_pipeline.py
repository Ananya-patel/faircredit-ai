from sklearn.model_selection import train_test_split

from src.data_pipeline.collectors import AlternativeDataCollector, DataGenerationConfig
from src.data_pipeline.feature_engineers import AlternativeFeatureEngineer
from src.data_pipeline.cleaners import DataCleaner
from src.data_pipeline.label_generator import CreditLabelGenerator
from src.models.fair_classifiers import FairnessConstrainedClassifier


def main() -> None:
    # Data pipeline
    collector = AlternativeDataCollector(DataGenerationConfig())
    raw_df = collector.collect()

    engineer = AlternativeFeatureEngineer()
    features_df = engineer.transform(raw_df)

    cleaner = DataCleaner()
    clean_df = cleaner.clean(features_df)

    labeler = CreditLabelGenerator()
    X, y = labeler.generate(clean_df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Baseline model
    model = FairnessConstrainedClassifier()
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print(metrics)


if __name__ == "__main__":
    main()
