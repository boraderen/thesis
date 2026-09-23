import pandas as pd
from minisom import MiniSom
from sklearn.decomposition import PCA

def standardize(feature_matrix: pd.DataFrame, method: str, exclude: list = ["current_act", "past_acts", "act_set"]) -> pd.DataFrame:
    # column groups in exclude are ignored, by default they are set to kairo's standard features but can adjusted for manual usage
    # supported methods are zscore and minmax to 0-1
    
    columns = [c for c in feature_matrix.columns if not c.startswith(tuple(exclude))]
    values = feature_matrix[columns]

    if method == "zscore":
        scaled = (values - values.mean()) / values.std()
    elif method == "minmax":
        scaled = (values - values.min()) / (values.max() - values.min())
    else:
        raise ValueError(f"Unknown method: {method}, pick zscore or minmax")

    # a column that never changes has no spread, the division leaves it empty
    scaled = scaled.fillna(0.0)

    result = feature_matrix.copy()
    result[columns] = scaled

    return result

def compute_pca(feature_matrix: pd.DataFrame, num_components: int = None) -> PCA:
    # return PCA object
    pca = PCA(n_components=num_components if num_components else explained_var)
    components = pca.fit_transform(feature_matrix)

    names = [f"pc_{i + 1}" for i in range(pca.n_components_)]

    compressed_features = pd.DataFrame(
        components,
        index=feature_matrix.index,
        columns=names,
    )

    # how much variance every component carries, and how much every feature
    # contributes to it, which is what makes a component readable
    pca_stats = {
        "num_components": int(pca.n_components_),
        "explained_var": float(pca.explained_variance_ratio_.sum()),
        "explained_var_per_component": pd.Series(pca.explained_variance_ratio_, index=names),
        "loadings": pd.DataFrame(pca.components_.T, index=feature_matrix.columns, columns=names),
    }

    return pca

def apply_pca(feature_matrix: pd.DataFrame, pca: PCA, cut_component: int) -> pd.DataFrame:
    # return compressed dataframe based on cut_component
    pass

def compute_som(df: pd.DataFrame, size: tuple[int, int] = (5, 5), distance: str = "euclidean") -> pd.DataFrame:
    # every neuron of the grid becomes one cluster, so a 5x5 map can hold 25 of them.
    # supported distances are euclidean, cosine, manhattan and chebyshev

    data = df.to_numpy()

    som = MiniSom(
        x=size[0],
        y=size[1],
        input_len=data.shape[1],
        activation_distance=distance,
        random_seed=0,
    )

    # start the neurons off on random rows, then show every row once in random order.
    # the fixed seed above keeps the same run giving the same grid positions
    som.random_weights_init(data)
    som.train(data, num_iteration=len(data), random_order=True)

    # the winning neuron of a row is its cluster, (i, j) is where it sits on the
    # grid, so rows landing next to each other are rows the map found similar
    positions = [som.winner(x) for x in data]

    return pd.DataFrame(
        positions,
        index=df.index,
        columns=["i", "j"],
    )

def compute_kmeans():
    pass

def compute_dbcan():
    pass
