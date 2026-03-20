from __future__ import annotations
import copy
import logging
import numpy as np
import re
from q2mm import constants as co

logger = logging.getLogger(__name__)

# Row of mm3.fld where comments start.
COM_POS_START = 96
# Row where standard 3 columns of parameters appear.
P_1_START = 23
P_1_END = 33
P_2_START = 34
P_2_END = 44
P_3_START = 45
P_3_END = 55


class ParamError(Exception):
    pass


class ParamFE(Exception):
    pass


class ParamBE(Exception):
    pass


class Param:
    """
     A single parameter of a force field (FF). TODO rework this to match Google style docstrings
     for later sphinx autodocumentation.

     :var _allowed_range: Stored as None if not set, else it's set to True or
       False depending on :func:`allowed_range`.
    :type _allowed_range: None, 'both', 'pos', 'neg'

     :ivar ptype: Parameter type can be one of the following: ae, af, be, bf, df,
       imp1, imp2, sb, or q.
     :type ptype: string

     Attributes
     ----------
     d1 : float
          First derivative of parameter with respect to penalty function.
     d2 : float
          Second derivative of parameter with respect to penalty function.
     step : float
            Step size used during numerical differentiation.
     ptype : {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}
     value : float
             Value of the parameter.
    """

    __slots__ = ["_allowed_range", "_step", "_value", "d1", "d2", "ptype", "simp_var"]

    def __init__(self, d1: float = None, d2: float = None, ptype=None, value: float = None):
        """_summary_

        Args:
            d1 (float, optional): First derivative of parameter with respect to penalty function. Defaults to None.
            d2 (float, optional): Second derivative of parameter with respect to penalty function. Defaults to None.
            ptype (_type_, optional): Parameter type {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}. Defaults to None.
            value (float, optional): Value of the parameter. Defaults to None.
        """
        self._allowed_range = None
        self._step = None
        self._value = None
        self.d1 = d1
        self.d2 = d2
        self.ptype = ptype
        self.simp_var = None
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.ptype}]({self.value:7.4f})"

    @property
    def allowed_range(self) -> List[float]:
        """Returns the allowed range of values for the parameter based on its parameter type (ptype).

        Returns:
            List[float]: [minimum_value, maximum_value]
        """
        if self._allowed_range is None and self.ptype is not None:
            if self.ptype in ["q", "df", "bf", "af"]:
                # bf/af allow negative for transition-state force fields (TSFF)
                self._allowed_range = [-float("inf"), float("inf")]
            else:
                self._allowed_range = [0.0, float("inf")]
        return self._allowed_range

    @property
    def step(self):
        """TODO Google style
        Returns a float for the current step size that should be used. If
        _step is a string, return float(_step) * value. If
        _step is a float, simply return that.

        Not sure how well the check for a step size of zero works.
        """
        if self._step is None:
            try:
                self._step = co.STEPS[self.ptype]
            except KeyError:
                logger.warning(f"{self} doesn't have a default step size and none provided!")
                raise
        if isinstance(self._step, str):
            return float(self._step) * self.value
        else:
            return self._step

    @step.setter
    def step(self, x):
        self._step = x

    @property
    def value(self):
        if self.ptype == "ae" and self._value > 180.0:
            self._value = 180.0 - abs(180 - self._value)
        return self._value

    @value.setter
    def value(self, value):
        """TODO Google style
        When you try to give the parameter a value, make sure that's okay.
        """
        if self.value_in_range(value):
            self._value = value

    def convert_and_set(self, value: float, units=None):
        """Converts force constant value in kJ/molA to the correct units based on FF units and parameter type.

        Note: This should only be used for force constants, not equilibrium bond lengths or angles or charges.

        Args:
            value (float): New value for the parameter
            units (str, optional): units to convert to for FF, must be in constants.py. Defaults to None.
        """
        if value is None:
            return
        if units == co.MM3FF:
            self.value = (
                value / co.MM3_STR
            )  #  Uses the conversion factor specific to MM3.fld, Notes on this in box TODO: Remove in a later commit and note commit # in documentation
            # self.value = value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG**2)  if self.ptype == 'bf' else value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG)
            # self.value = value * co.AU_TO_MDYNA  if self.ptype == 'bf' else value * co.AU_TO_MDYN_ANGLE
            # self.value = value * 10**6  if self.ptype == 'bf' else value * co.KJMOLA_TO_MDYN
            # self.value = (
            #     value / co.MDYNA_TO_KJMOLA2
            #     if self.ptype == "bf"
            #     else value * co.KJMOLA_TO_MDYN
            # )
        elif (
            units == co.AMBERFF
        ):  # TODO Amber conversion factor is unknown, ask David Case because it is not just units.
            self.value = (
                value * co.HARTREE_TO_KCALMOL / (co.BOHR_TO_ANG**2)
                if self.ptype == "bf"
                else value * co.HARTREE_TO_KCALMOL
            )
        elif units == co.TINKERFF:
            raise NotImplementedError()
        else:
            raise Exception(
                "Only MM3, AMBER, and Tinker type force fields have defined units and conversions for parameters in Q2MM."
            )

    def value_in_range(self, value):
        """TODO

        Args:
            value (_type_): _description_

        Raises:
            ParamBE: _description_
            ParamFE: _description_
            ParamError: _description_

        Returns:
            _type_: _description_
        """
        if self.allowed_range[0] <= value <= self.allowed_range[1]:
            return True
        elif value == self.allowed_range[0] - 0.1:
            raise ParamBE(f"{str(self)} Backward Error. Forward Derivative only")
        elif value == self.allowed_range[1] + 0.1:
            raise ParamFE(f"{str(self)} Forward Error. Backward Derivative only")
        elif value == self.allowed_range[1] or value == self.allowed_range[0]:
            return True
        else:
            raise ParamError(
                f"{str(self)} isn't allowed to have a value of {value}! "
                f"({self.allowed_range[0]} <= x <= {self.allowed_range[1]})"
            )

    def value_at_limits(self):
        """TODO"""
        # Checks if the parameter is at the limits of
        # its allowed range. Should only be run at the
        # end of an optimization to warn users they should
        # consider whether this is ok.
        if self.value == min(self.allowed_range):
            logger.warning(
                f"{str(self)} is equal to its lower limit of {self.value}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate."
            )
        if self.value == max(self.allowed_range):
            logger.warning(
                f"{str(self)} is equal to its upper limit of {self.value}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate."
            )


# TODO: MF - I see little reason to have ParamMM3 or ParAMBER, having a singular Param object
# should suffice by using a Param.units or Param.ff_type variable. Perhaps consider this in future
# refactoring efforts.


# Need a general index scheme/method/property to compare the equalness of two
# parameters, rather than having to rely on some expression that compares
# ff_row and ff_col.
# MF - I agree, a __equal__ would be nice, but its use would require a refactor so I recommend for future.
class ParamMM3(Param):
    """
    Adds information to Param that is specific to MM3* parameters. TODO
    """

    __slots__ = ["atom_labels", "atom_types", "ff_col", "ff_row", "mm3_label"]

    def __init__(
        self,
        atom_labels=None,
        atom_types=None,
        ff_col=None,
        ff_row=None,
        mm3_label=None,
        d1=None,
        d2=None,
        ptype=None,
        value=None,
    ):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.ff_col = ff_col
        self.ff_row = ff_row
        self.mm3_label = mm3_label
        super().__init__(ptype=ptype, value=value)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.ptype}][{self.ff_row},{self.ff_col}]({self.value})"

    def __str__(self):
        return f"{self.__class__.__name__}[{self.ptype}][{self.ff_row},{self.ff_col}]({self.value})"

    def convert_and_set(self, value):
        return super().convert_and_set(value, units=co.MM3FF)


class ParAMBER(Param):
    """
    Adds information to Param that is specific to AMBER parameters. TODO
    """

    __slots__ = ["atom_labels", "atom_types", "ff_col", "ff_row", "mm3_label"]

    def __init__(
        self,
        atom_labels=None,
        atom_types=None,
        ff_col=None,
        ff_row=None,
        mm3_label=None,
        d1=None,
        d2=None,
        ptype=None,
        value=None,
    ):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.ff_col = ff_col
        self.ff_row = ff_row
        self.mm3_label = mm3_label
        super().__init__(ptype=ptype, value=value)

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.ptype}][{self.ff_row},{self.ff_col}]({self.value})"

    def __str__(self):
        return f"{self.__class__.__name__}[{self.ptype}][{self.ff_row},{self.ff_col}]({self.value})"

    def convert_and_set(self, value):
        return super().convert_and_set(value, units=co.AMBERFF)
