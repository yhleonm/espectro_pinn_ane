# Parsers para datos topográficos y cartográficos
from .srtm_reader import SRTMReader, download_srtm_tile

try:
    from .vec_parser import VecParser
except Exception:
    VecParser = None
