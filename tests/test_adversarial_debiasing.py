import numpy as np
from sklearn.model_selection import train_test_split

from src.data_pipeline.collectors import AlternativeDataCollector, DataGenerationConfig
from src.data_pipeline.feature_engineers import AlternativeFeatureEngineer
from src.data_pipeline.cleaners import DataCleaner
from src.data_pipeline.label_generator import CreditLabelGenerator
from src.models.adversarial_debiasing import AdversarialDebiasingModel


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
        X.values,
        y.values,
        sensitive_attr.values,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = AdversarialDebiasingModel(
        input_dim=X_train.shape[1],
        lambda_adv=1.0,
    )

    model.fit(X_train, y_train, s_train, epochs=40)

    y_pred = model.predict(X_test)

    print("Sample predictions:", y_pred[:10])


if __name__ == "__main__":
    main()
