from ..attacks import AttackMethods

STRENGTH_ALL = [0.2, 0.5, 0.8, 1.0]

ATTACK_ALL = {
    AttackMethods.COMB_DIST_PHOTO: STRENGTH_ALL,
    AttackMethods.COMB_DIST_GEOMETRIC: STRENGTH_ALL,
    AttackMethods.COMB_DIST_DEGRAD: STRENGTH_ALL,
    AttackMethods.COMB_DIST_ALL: STRENGTH_ALL,
    AttackMethods.ADV_EMB_CLIP: STRENGTH_ALL,
    AttackMethods.ADV_EMB_KLVAE_8: STRENGTH_ALL,
    AttackMethods.ADV_EMB_CLIP: STRENGTH_ALL,
    AttackMethods.ADV_EMB_RESNET_18: STRENGTH_ALL,
    AttackMethods.ADV_EMB_SDXL_VAE: STRENGTH_ALL,
    AttackMethods.ADV_SURROGATE: STRENGTH_ALL,
}