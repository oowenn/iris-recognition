"""Paper section 3.4: Fisher projection, rotation templates, nearest centers."""
from dataclasses import dataclass
from typing import Any
from PipelineConfig import METRICS, ROTATION_ANGLES


@dataclass
class MatcherModel:
    """State learned only from training data; store exact preprocessing too.

    projection: (1536,d); feature_mean: (1536,); class_labels: sorted (108,).
    centers: (108,7,d), rotation axis ordered as ROTATION_ANGLES.
    Project queries and training vectors with the SAME centering convention.
    """
    projection: Any
    feature_mean: Any
    class_labels: Any
    centers: Any


def fit_matcher(train_features, train_labels, *, choices):
    """Return MatcherModel from (324,7,1536) training features and (324,) labels.

    Team convention: fit Fisher on zero-angle training vectors only, then project
    each rotation and average the three training samples per eye/rotation.
    This makes seven centers per eye. Paper does not spell out rotation
    augmentation during Fisher fitting; record this choice explicitly.
    Standard Fisher dimension <= classes-1 = 107 (paper used 200 with more classes).
    Raw within-class scatter is singular: rank <= 324-108 = 216. TODO: choose
    documented PCA+Fisher or SVD method; use effective ranks and training data only.
    """
    raise NotImplementedError("Implement training-only Fisher model and seven centers")


def score_queries(model, query_features, metric):
    """Return (distances, class_labels) with distances shape (n_queries,108).

    query_features: (n_queries,1536), extracted at zero angle. Project once;
    compare each query with each class's seven rotated centers, taking minimum.
    L1=sum(abs(a-b)); L2=sum((a-b)**2), following equation (8), no square root;
    cosine=1-dot(a,b)/(norm(a)*norm(b)). Smaller is always closer.
    TODO: implement and specify zero-norm behavior and deterministic ties.
    Never roll or rotate coordinates AFTER Fisher projection.
    """
    raise NotImplementedError("Implement distances and minimum over rotations")
