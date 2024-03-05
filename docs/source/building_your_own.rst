Bulding Your Own
================

PyAnomaly is highly configurable and customizable, and you can easily add new firm characteristics or functions.
When you make modifications, do not change the original source directly. Rather, add new modules (files) and define subclasses if necessary.
This is because the library can be updated in the future, and if you change the original source, you will lose the changes you made when you
update the library.

If you want some of your modules to be added to PyAnomaly, please contact us.

Coding Rule
------------

There are a few coding rules we follow and we suggest you follow them when developing your own library based on PyAnomaly.

1. Methods for firm characteristics
    - A method (function) that generates a firm characteristic should have a name of the form `c_[characteristic name]`.
    - Write a single-line docstring that describes the characteristic and its author(s) (year).
      The docstring content will be displayed when ``Panel.show_available_functions()`` is called.
    - Do not make a method that generate multiple firm characteristics. For each firm characteristic, make one associated
      method. If for some reason, it is efficient to generate multiple firm characteristics in one method,
      make a private method for it and then make a method for each firm characteristic that calls the private method.
      Below is an example of generating earnings persistence and predictability. ``_ni_ar1_ivol()`` generates both
      persistence and predictability and ``c_ni_ar1()`` and ``c_ni_ivol()`` call ``_ni_ar1_ivol()`` to generate
      persistence and predictability, respectively.

    .. code-block::

        def _ni_ar1_ivol(self):
            """Earnings persistence and predictability"""

            fa = self.data
            n = 5  # window size

            reg = pd.DataFrame(index=fa.index)
            reg['ni_at'] = fa.ib / fa.at_
            reg['ni_at_lag1'] = self.shift(reg.ni_at)

            beta, _, ivol = self.rolling_beta(reg, n, n)
            fa['ni_ar1'] = beta[:, 1]
            fa['ni_ivol'] = ivol

        def c_ni_ar1(self):
            """Earnings persistence. Francis et al. (2004)"""

            if 'ni_ar1' not in self.data:
                self._ni_ar1_ivol()

            return self.data['ni_ar1']

        def c_ni_ivol(self):
            """Earnings predictability. Francis et al. (2004)"""

            if 'ni_ivol' not in self.data:
                self._ni_ar1_ivol()

            return self.data['ni_ivol']

2. Temporary variables
    If you need to add temporary variables to ``Panel.data`` attribute, use a name starting with underscore, e.g., '_temp'.
    Any column whose name starts with '_' will be considered a temporary variable and deleted during postprocessing.

3. Date format
    For date arguments, PyAnomaly uses a string of 'yyyy-mm-dd' format.

4. Naming convention
    PyAnomaly uses UpperCamelCase for class names and lowercase_separated_by_underscores for other names (functions,
    methods, and attributes).

