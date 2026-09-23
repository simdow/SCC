def matrices(vectors):
    import numpy as np
    from scipy.spatial.transform import Rotation
    result = np.broadcast_to(np.eye(4), (len(vectors), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_euler('yzx', vectors[:, 3:], degrees=True).as_matrix()
    result[:, :3, 3] = vectors[:, :3]
    return result

def metrics(prediction, truth):
    import numpy as np
    from scipy.spatial.transform import Rotation
    (p, g) = ([np.eye(4)], [np.eye(4)])
    for (pi, gi) in zip(prediction, truth):
        p.append(p[-1] @ pi)
        g.append(g[-1] @ gi)
    (p, g) = (np.asarray(p), np.asarray(g))
    errors = np.linalg.norm(p[:, :3, 3] - g[:, :3, 3], axis=1)
    length = np.linalg.norm(truth[:, :3, 3], axis=1).sum()
    assert length > 1e-08
    bias = (prediction[:, :3, 3] - truth[:, :3, 3]).mean(0)
    return dict(FDR=float(100 * errors[-1] / length), ATE=float(np.sqrt(np.mean(errors ** 2))), FD=float(errors[-1]), RTE=float(np.linalg.norm(prediction[:, :3, 3] - truth[:, :3, 3], axis=1).mean()), RRE=float(np.rad2deg((Rotation.from_matrix(prediction[:, :3, :3]).inv() * Rotation.from_matrix(truth[:, :3, :3])).magnitude()).mean()), path_length=float(length), path_ratio=float(np.linalg.norm(prediction[:, :3, 3], axis=1).sum() / length), translation_bias=bias.tolist())

def window_starts(length):
    assert length >= 5
    starts = list(range(0, length - 4, 4))
    if starts[-1] != length - 5:
        starts.append(length - 5)
    return starts
