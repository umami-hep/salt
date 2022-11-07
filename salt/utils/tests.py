def get_scale_dict(jet_features: int, track_features: int):
    jet_vars = ["pt", "eta"]
    track_vars = [f"test_{i}" for i in range(track_features)]
    sd: dict = {}
    sd["jets"] = [{"name": n, "scale": 1, "shift": 1} for n in jet_vars]
    sd["tracks_loose"] = {n: {"scale": 1, "shift": 1} for n in track_vars}
    return sd
