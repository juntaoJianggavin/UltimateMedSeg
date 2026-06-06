"""Ultrasound foundation encoders (USF-MAE, UltraDINO, UltraFedFM)."""
for _stem in ('usfmae_encoder', 'ultradino_encoder', 'ultrafedfm_encoder'):
    try:
        __import__(f"{__name__}.{_stem}")
    except ImportError:
        pass
