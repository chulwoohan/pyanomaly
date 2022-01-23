Changelog
=========

v0.9 - 2022.01.15
-----------------

Initial version.

v0.923 - 2022.01.16
--------------------

Multiprocessing in ``datatools.populate()`` has been updated to increase the speed.


v0.930 - 2022.01.17
--------------------

The trend factor of Han, Zhou, and Zhu (2016) has been added. We thank Guofu Zhou for this suggestion.


v0.931 - 2022.01.23
--------------------

A bug of not returning the result in FUNDA.c_ebitda_mev has been fixed.

A new characteristic method for Enterprise multiple (Loughran and Wellman, 2011), c_enterprise_multiple,
has been added to FUNDA, as the previous one (c_ebitda_mev) that implements JKP's SAS code uses a different definition
from the original definition. This new method uses the original definition.


