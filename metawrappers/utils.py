def adjust_feature_bounds(min_features, max_features, X):
    if min_features < 1:
        min_features = 1
    if max_features in (-1, None) or max_features > X.shape[1]:
        max_features = X.shape[1]
    if max_features < min_features:
        max_features = min_features
    return min_features, max_features
