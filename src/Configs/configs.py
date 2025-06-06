
import numpy as np
from typing import Optional, Any


__all__ = ['Config']

class Config:
    """
    Simple configuration container for the pipeline steps.
    Any attribute not set will default to None (or be read via get_attr).
    """
    def __init__(
        self,
        do_ffc: bool = True,
        do_gc: bool = True,
        do_wb: bool = True,
        do_cc: bool = True,
        save: bool = False,
        check_saturation : bool = True,
        save_path: Optional[str] = None,
        REF_ILLUMINANT: Optional[np.ndarray] = None,
        FFC_kwargs: Optional[Any] = None,
        GC_kwargs: Optional[Any] = None,
        WB_kwargs: Optional[Any] = None,
        CC_kwargs: Optional[Any] = None,
    ) -> None:
        self.do_ffc = do_ffc
        self.do_gc = do_gc
        self.do_wb = do_wb
        self.do_cc = do_cc
        self.save = save
        self.save_path = save_path
        self.REF_ILLUMINANT = REF_ILLUMINANT
        self.FFC_kwargs = FFC_kwargs
        self.GC_kwargs = GC_kwargs
        self.WB_kwargs = WB_kwargs
        self.CC_kwargs = CC_kwargs
        self.check_saturation = check_saturation

