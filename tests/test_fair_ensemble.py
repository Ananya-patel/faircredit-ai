from sklearn.model_selection import train_test_split

from src.data_pipeline.collectors import AlternativeDataCollector, DataGenerationConfig
from src.data_pipeline.feature_engineers import AlternativeFeatureEngineer
from src.data_pipeline.cleaners import DataCleaner
from src.data_pipeline.label_generator import CreditLabelGenerator
from src.models.fair_classifiers import FairnessConstrainedClassifier
from src.models.adversarial_debiasing import AdversarialDebiasingModel
from src.models.ensemble_methods import FairEnsemble, GroupCalibrator


def main() -> None:
    # Data pipeline
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

    # Baseline model
    baseline = FairnessConstrainedClassifier()
    baseline.fit(X_train, y_train)

    # Debiased model
    debiased = AdversarialDebiasingModel(input_dim=X_train.shape[1])
    debiased.fit(X_train.values, y_train.values, s_train.values, epochs=30)

    # Ensemble
    ensemble = FairEnsemble(weight_baseline=0.5)
    ensemble.fit(baseline, debiased)

    ensemble_preds = ensemble.predict(X_test)
    print("Ensemble sample preds:", ensemble_preds[:10])

    # Group calibration
    calibrator = GroupCalibrator()
    calibrator.fit(X_train, y_train, s_train)

    calibrated_probs = calibrator.predict_proba(X_test, s_test)
    print("Calibrated probs sample:", calibrated_probs[:10])


if __name__ == "__main__":
    main()
