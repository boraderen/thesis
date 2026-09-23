import pandas as pd
from minisom import MiniSom
from sklearn.decomposition import PCA

# carried along with every feature matrix, never part of the maths
META = ["case:concept:name", "time:timestamp"]

def standardize(feature_matrix: pd.DataFrame, method: str, exclude: list = ["current_act", "past_acts", "act_set"]) -> pd.DataFrame:
    # column groups in exclude are ignored, by default they are set to kairo's standard features but can adjusted for manual usage
    # supported methods are zscore and minmax to 0-1
    # the case id and timestamp come along with every feature matrix, never scaled

    columns = [
        c for c in feature_matrix.columns
        if c not in META and not c.startswith(tuple(exclude))
    ]
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
    # num_components is how many components are fitted, None fits all of them.
    # where to actually cut is decided afterwards, by looking at their variances
    # the meta columns are only there for reference, they never take part in the maths
    pca = PCA(n_components=num_components)
    pca.fit(feature_matrix.drop(columns=META, errors="ignore"))

    return pca

def apply_pca(feature_matrix: pd.DataFrame, pca: PCA, cut_component: int) -> pd.DataFrame:
    # return compressed dataframe based on cut_component
    # the components come out ordered by variance, so cutting keeps the strongest ones
    features = feature_matrix.drop(columns=META, errors="ignore")
    components = pca.transform(features)[:, :cut_component]

    names = [f"pc_{i + 1}" for i in range(components.shape[1])]

    compressed = pd.DataFrame(
        components,
        index=feature_matrix.index,
        columns=names,
    )

    # the meta columns travel on with the components, the way they do with the features
    meta = [c for c in META if c in feature_matrix.columns]

    return pd.concat([feature_matrix[meta], compressed], axis=1)

def compute_som(df: pd.DataFrame, size: tuple[int, int] = (5, 5), distance: str = "euclidean") -> MiniSom:
    # every neuron of the grid becomes one cluster, so a 5x5 map can hold 25 of them.
    # supported distances are euclidean, cosine, manhattan and chebyshev

    # the meta columns are only carried along for reference, the map trains without them
    data = df.drop(columns=META, errors="ignore").to_numpy()

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

    return som

def get_som_winners(df: pd.DataFrame, som: MiniSom) -> pd.DataFrame:
    # should return df of two columns giving for each row the winning grid coordinates
    data = df.drop(columns=META, errors="ignore").to_numpy()

    winners = pd.DataFrame(
        [som.winner(x) for x in data],
        index=df.index,
        columns=["i", "j"],
    )

    return winners

def compute_kmeans():
    pass

def compute_dbcan():
    pass
