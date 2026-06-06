"""Loss functions."""

from . import ce_loss
from . import dice_loss
from . import edge_loss
from . import focal_loss
from . import tversky_loss
from . import lovasz_loss
from . import boundary_loss
from . import compound_loss
from . import deep_supervision_loss
from . import hausdorff_loss
from . import nsd_loss
from . import contrastive_loss
from . import wasserstein_dice_loss
from . import el_loss
from . import kl_loss

# Semi-supervised methods now live in medseg/semi/ (not as criterion classes)
# — the ssl4mis_losses.py shim was removed since none of its 23 registered
# loss classes had a forward() signature compatible with the criterion path.
