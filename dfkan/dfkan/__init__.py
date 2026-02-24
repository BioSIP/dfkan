# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author:       Andrés Ortiz, 2026
# Group:        BioSiP Research Group
# License:      MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

from .KAN_shared_poly7 import (
    DualFlexKANLinear,
    DualFlexKAN,
    KANFeatureAttention,
    DualFlexKANAttentionLayer,
    SelectiveAttentiveKAN,
)
from .KAN_activation import (
    StandardActivation,
    ActivationFunction,
    PolynomialActivation,
    GegenbauerActivation,
    JacobiActivation,
    BSplineActivation,
    FastBSplineActivation,
    RBFActivation,
    LegendreActivation,
    OptimizedLegendreActivation,
    ChebyshevActivation,
    SineActivation,
    RationalActivation,
    WaveletActivation,
)
from .KAN_initialization_v2 import (
    KANInitializer,
    InitializationAnalyzer,
    smart_kan_initialization,
    initialize_model_layerwise,
)
from .save_load_utils import (
    save_object,
    load_object,
    save_multiple,
    load_multiple,
    save_experiment,
    list_experiments,
    save_model_auto,
    load_model_auto,
    extract_data_from_loader,
    extract_kan_config,
)
from .KAN_explainability_importance import (
    KANExplainability,
    quick_explain,
    plot_aggregated_importance_heatmap,
    plot_saliency_map,
    plot_importance_distribution,
)
from .KAN_explainability_interaction_analysis import KANInteractionAnalysis

__version__ = "0.1.0"
__author__ = "dfkan"
