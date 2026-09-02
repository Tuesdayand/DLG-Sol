from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..label_firewall import bind_labels_for_scoring
from ..metrics import paired_rmse_bootstrap, regression_metrics


@dataclass(frozen=True)
class EvaluationScore:
    metrics: dict[str, dict]
    paired_rmse: dict[str, dict]


def score_evaluation_predictions(predictions: pd.DataFrame, labels: pd.DataFrame, seed: int = 20260821, replicates: int = 10000) -> EvaluationScore:
    expected = {"record_id", "descriptor_prediction", "aug2_head4", "dlg_sol"}
    if set(predictions.columns) != expected:
        raise ValueError("assembled prediction columns differ from the scoring contract")
    if set(labels.columns) != {"record_id", "logS"}:
        raise ValueError("scored-label columns differ from the scoring contract")
    bound = bind_labels_for_scoring(predictions, labels, ["descriptor_prediction", "aug2_head4", "dlg_sol"])
    observed = bound["logS"]
    metrics = {name: asdict(regression_metrics(observed, bound[column])) for name, column in (("descriptor", "descriptor_prediction"), ("neural", "aug2_head4"), ("dlg_sol", "dlg_sol"))}
    paired = {
        "dlg_sol_minus_descriptor": asdict(paired_rmse_bootstrap(observed, bound["dlg_sol"], bound["descriptor_prediction"], seed=int(seed), replicates=int(replicates))),
        "dlg_sol_minus_neural": asdict(paired_rmse_bootstrap(observed, bound["dlg_sol"], bound["aug2_head4"], seed=int(seed) + 1, replicates=int(replicates))),
    }
    return EvaluationScore(metrics, paired)
