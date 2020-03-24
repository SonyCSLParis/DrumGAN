
from .pgan_config import _C

#Starting scale
_C.dataType = 'audio'

_C.sampleRate = 16000
_C.output_shape = [1, 1, 16000]

_C.n_mlp = 8
_C.noise_injection = True
_C.style_mixing = True