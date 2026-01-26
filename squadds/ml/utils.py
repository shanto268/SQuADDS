from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor


def characterize_shape_function(
    ebm: ExplainableBoostingRegressor,
    term_idx: int,
) -> dict[str, Any]:
    """
    Characterize the shape of a learned function (monotonic, saturating, linear, etc.).

    Args:
        ebm: Trained EBM model
        term_idx: Index of the term to analyze

    Returns:
        Dictionary with shape characteristics
    """
    term_name = ebm.term_names_[term_idx]

    # Handle interaction terms separately or skip logic for them
    if " x " in term_name or " & " in term_name:
        return {"term": term_name, "type": "interaction", "details": "See heatmap"}

    # EBM stores bins/scores differently depending on version/internal structure
    # Standard access for univariate terms usually involves lookups on the additive terms
    # But ebm.term_scores_ might be the easiest way if available and populated
    # In the tutorial it used ebm.term_scores_[term_idx], but let's verify if that's standard.
    # Actually, often we need to look at bin predictions.
    # However, following the tutorial implementation which relied on term_scores_ (feature contribution for samples)
    # OR it might have been plotting data.
    # Let's stick closer to the tutorial's logic which used `ebm.term_scores_[term_idx]`
    # BUT wait, `term_scores_` usually requires running predict_and_contrib.
    # If the tutorial accessed `ebm.term_scores_`, it implies it might have been a custom attribute or
    # the result of an explanation object.
    # Let's check the tutorial code again.
    # Tutorial: "term_data = ebm.term_scores_[term_idx]"
    # Standard EBM sklearn API doesn't have `term_scores_` as a direct attribute after fit.
    # It has `term_scores_` only if specifically computed or maybe it's `term_importances_`.
    # Hmmm. Let's look at the tutorial code I read:
    # 1658: "term_data = ebm.term_scores_[term_idx]"
    # This suggests that in the context of the tutorial, `ebm` might have been an object that HAD this.
    # Or maybe it meant `ebm.explain_global().data(term_idx)`.

    # To be safe and generic, let's assume we can pass the specific score values (y-axis of the shape function)
    # OR we access the model graphs directly.
    # ebm.term_scores_ is NOT standard.
    # Let's try to extract the graph data properly from the model.
    #

    # For now, I will implement it assuming we can extract the graph data.
    # shape (n_bins, )

    try:
        # This is the robust way to get the shape function for a term
        data = ebm.explain_global(name=term_name).data(term_idx)
        # data['scores'] holds the y-values
        term_data = np.array(data["scores"])
    except Exception:
        # Fallback if the above API fails or isn't available
        return {"term": term_name, "type": "unknown", "error": "Could not extract shape data"}

    # Analyze the shape
    differences = np.diff(term_data)

    # Check monotonicity
    is_monotonic_increasing = np.all(differences >= -1e-10)
    is_monotonic_decreasing = np.all(differences <= 1e-10)

    # Check for saturation (flattening at ends)
    n = len(differences)
    if n == 0:
        return {"term": term_name, "type": "constant", "range": (0, 0), "effect_size": 0}

    early_slope = np.mean(np.abs(differences[: n // 4])) if n >= 4 else 0
    late_slope = np.mean(np.abs(differences[-n // 4 :])) if n >= 4 else 0
    mid_slope = np.mean(np.abs(differences[n // 4 : 3 * n // 4])) if n >= 4 else np.mean(np.abs(differences))

    early_slope < 0.1 * mid_slope if mid_slope > 0 else False
    saturates_late = late_slope < 0.1 * mid_slope if mid_slope > 0 else False

    # Check linearity (constant slope)
    if len(differences) > 2:
        slope_variation = np.std(differences) / (np.mean(np.abs(differences)) + 1e-10)
        is_approximately_linear = slope_variation < 0.3
    else:
        is_approximately_linear = True

    # Determine overall shape type
    if is_monotonic_increasing:
        if is_approximately_linear:
            shape_type = "linear increasing"
        elif saturates_late:
            shape_type = "saturating (asymptotic) increase"
        else:
            shape_type = "nonlinear increasing"
    elif is_monotonic_decreasing:
        if is_approximately_linear:
            shape_type = "linear decreasing"
        elif saturates_late:
            shape_type = "saturating (asymptotic) decrease"
        else:
            shape_type = "nonlinear decreasing"
    else:
        shape_type = "non-monotonic"

    return {
        "term": term_name,
        "type": shape_type,
        "range": (float(np.min(term_data)), float(np.max(term_data))),
        "effect_size": float(np.max(term_data) - np.min(term_data)),
    }


def get_top_features_from_ebm(
    ebm: ExplainableBoostingRegressor,
    design_params: list[str],
    threshold: float = 0.05,
) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Extract the most important features from an EBM for use in symbolic regression.

    Args:
        ebm: Trained EBM model or result dictionary (if it has term_names and term_importances)
        design_params: List of valid design parameter names (feature whitelist)
        threshold: Minimum relative importance (0.0 to 1.0) to include a feature

    Returns:
        Tuple of (base_features, interaction_pairs)
        - base_features: List of single feature names
        - interaction_pairs: List of tuples like [('cross_length', 'ground_spacing'), ...]
    """
    if hasattr(ebm, "term_names_"):
        term_names = ebm.term_names_
        # Handle term_importances (could be property or method or with _)
        if hasattr(ebm, "term_importances_"):
            term_importances = ebm.term_importances_
        elif hasattr(ebm, "term_importances"):
            term_importances = ebm.term_importances()
        else:
            # Fallback: try exp
            try:
                explanation = ebm.explain_global()
                term_importances = explanation.data()["scores"]
            except Exception as err:
                raise ValueError("Could not extract term importances from EBM model") from err
    elif isinstance(ebm, dict):
        term_names = ebm.get("term_names", [])
        term_importances = ebm.get("term_importances", [])
    else:
        raise ValueError("ebm argument must be an ExplainableBoostingRegressor or a dict with term_names/importances")

    total_importance = sum(term_importances)

    base_features = set()
    interaction_pairs = []

    for name, importance in zip(term_names, term_importances):
        relative_importance = importance / total_importance if total_importance > 0 else 0
        if relative_importance >= threshold:
            # Check for interaction terms (contain ' x ' or ' & ')
            if " x " in name:
                features = name.split(" x ")
                if len(features) == 2:
                    f1, f2 = features[0].strip(), features[1].strip()
                    if f1 in design_params and f2 in design_params:
                        interaction_pairs.append((f1, f2))
                        # Also include base features
                        base_features.add(f1)
                        base_features.add(f2)
            elif " & " in name:
                features = name.split(" & ")
                if len(features) == 2:
                    f1, f2 = features[0].strip(), features[1].strip()
                    if f1 in design_params and f2 in design_params:
                        interaction_pairs.append((f1, f2))
                        base_features.add(f1)
                        base_features.add(f2)
            else:
                # Single feature
                if name in design_params:
                    base_features.add(name)

    return list(base_features), interaction_pairs


def prepare_features_with_interactions(
    X: pd.DataFrame,
    base_features: list[str],
    interaction_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Create a dataframe with base features AND interaction (polynomial) features.

    Args:
        X: Original dataframe with design parameters
        base_features: List of single feature names to include
        interaction_pairs: List of tuples like [('cross_length', 'ground_spacing'), ...]

    Returns:
        DataFrame with base features + interaction features
    """
    # Ensure we only select columns that exist in X
    valid_base = [f for f in base_features if f in X.columns]
    result = X[valid_base].copy()

    # Add interaction features
    for f1, f2 in interaction_pairs:
        if f1 in X.columns and f2 in X.columns:
            interaction_name = f"{f1}_x_{f2}"
            result[interaction_name] = X[f1] * X[f2]

    return result
