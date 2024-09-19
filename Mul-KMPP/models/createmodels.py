from .mul-muppl import Mul_KMPP


def create_model(cfg, device, pn_weights=None, y0_weights=None):
    return Mul_KMPP(cfg, device, pn_weights=pn_weights, y0_weights=y0_weights)

