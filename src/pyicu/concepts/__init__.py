from .concept import *
from .item import *

CONCEPT_CLASSES = {
    "num_cncpt": NumConcept,
    "fct_cncpt": FctConcept,
    "lgl_cncpt": LglConcept,
    "rec_cncpt": RecConcept
}

ITEM_CLASSES = {
    "sel_itm": SelItem,
    "rgx_itm": RgxItem,
    "col_itm": ColItem,
    "fun_itm": FunItem
}