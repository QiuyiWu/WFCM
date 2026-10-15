"""Reproducible simulation code for the WFCM revision."""

from .competitors import (
    ClusteringResult,
    DEFAULT_METHODS,
    fcm_membership,
    fit_competitor,
    fit_competitors,
    polynomial_fuzzifier,
    polynomial_membership,
)
from .data_generators import ScenarioData, factorial_scenarios, generate_scenario
from .generalized_mm import GeneralizedMMResult, fit_generalized_mm

from .wfcm_core import (
    FitResult,
    ISContext,
    WFCMParams,
    WFCMParameterization,
    align_to_reference,
    build_proposal,
    energy,
    fit_wfcm,
    membership,
    sample_wfcm_1d,
    weighted_xie_beni,
)

__all__ = [
    "ClusteringResult",
    "DEFAULT_METHODS",
    "FitResult",
    "GeneralizedMMResult",
    "ISContext",
    "WFCMParams",
    "WFCMParameterization",
    "align_to_reference",
    "build_proposal",
    "energy",
    "factorial_scenarios",
    "fcm_membership",
    "fit_competitor",
    "fit_competitors",
    "fit_generalized_mm",
    "fit_wfcm",
    "generate_scenario",
    "membership",
    "polynomial_fuzzifier",
    "polynomial_membership",
    "sample_wfcm_1d",
    "ScenarioData",
    "weighted_xie_beni",
]
