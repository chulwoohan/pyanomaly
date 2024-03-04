==========
PyAnomaly
==========

PyAnomaly is a comprehensive python library for asset pricing research with a focus on firm characteristic and factor generation.
It covers the majority of the firm characteristics published in the literature and contains various analytic tools that are
commonly used in asset pricing research, such as quantile portfolio construction, factor regression, and cross-sectional regression.
The purpose of PyAnomaly is *NOT* to generate firm characteristics in a fixed manner. Rather, we aim to build
a package that can serve as a standard library for asset pricing research and help reduce *non-standard errors*\ [5]_.

The current list of the firm characteristics supported by PyAnomaly can be found in `Coverage`_.
PyAnomaly is a live project and we plan to add more firm characteristics and functionalities going forward. We also welcome contributions
from other scholars.

PyAnomaly is very efficient, comprehensive, and flexible.

    Efficiency
        PyAnomaly can generate over 200 characteristics from 1950 under one hour (tested on a desktop with
        12th Gen Intel(R) Core(TM) i9-12900KS and 128GB RAM and 1GB network). Once the data have been downloaded from
        WRDS, the processing time is under 15 minutes.
        To achieve this, PyAnomaly utilizes numba, multiprocessing, and asyncio packages when possible, but not too
        heavily to maximize readability of the code.
        The library is slower on the first run due to caching of numba jitted functions.

    Comprehensiveness
        PyAnomaly supports over 200 firm characteristics published in the literature. It covers most characteristics in
        Green et al. (2017)\ [2]_ and Jensen et al. (2021)\ [4]_, except those that use IBES data. It also provides
        various tools for asset pricing research.

    Flexibility
        PyAnomaly adopts the object-oriented programming design philosophy and is easy to customize or add functionalities.
        This means users can easily change the definition of an existing characteristic, add a new characteristic, or
        change configurations to run the program. For instance, a user can choose whether to update annual accounting
        variables quarterly (using Compustat.fundq) or annually (using Compustat.funda), or whether
        to use the latest market equity or the year-end market equity when generating firm characteristics.


Main Features
=============

* Efficient data download from WRDS using asynco.
* Over 200 firm characteristics generation. You can choose which firm characteristics to generate.
* Factor models

    * Fama-French 3-factor model
    * Fama-French 5-factor model
    * Hou-Xue-Zhang 4-factor model
    * Stambaugh-Yuan 4-factor model

* Analytics

    * Cross-sectional regression
    * 1-D sort
    * 2-D sort
    * Rolling regression
    * Quantile portfolio
    * Long-short portfolio
    * Portfolio performance analysis

* Data tools

    * Filtering
    * Winsorizing
    * Trimming
    * Grouping
    * Population
    * And more...


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

v1.0 - 2024.02.28 (Major Update)
--------------------------------

There are several important updates in this version and some functions are not backward compatible.
See the examples in :ref:`cookbook` for changes.

**Major updates**

- Performance upgrade

    The library is now significantly faster and more memory efficient.

- ``panel.FCPanel``

    The ``Panel`` class has been divided into two classes: ``Panel`` class that serves as the base class for panel
    data analysis and ``FCPanel`` class that inherits ``Panel`` and serves as the base class for firm characteristics
    generation. ``FUNDA``, ``FUNDQ``, ``CRSPM``, ``CRSPD``, and ``Merge`` now inherit ``FCPanel`` instead of ``Panel``.

- ``characteristics.CRSPDRaw``

    Previously, ``CRSPD.data`` contained daily crspd data and ``CRSPD.chars`` contained monthly firm characteristics.
    In the new version, a new class ``CRSPDRaw`` handles daily crspd data and is a member of ``CRSPD``.
    ``CRSPDRaw.data`` contains daily crspd data and ``CRSPD.data`` contains monthly firm characteristics.

- Factor models

    Two new factor models, Fama-French 5-factor and Stambough and Yuan 4-factor models, have been added.

- CRSP-Compustat link

    If a use don't have WRDS subscription for ccmxpf_linktable, PyAnomlay will create a link table internally and use it
    to map permno and gvkey. Compared to using ccmxpf_linktable, about 13% of gvkey's are different when using the
    internal link table ('crsp_comp_linktable').

**Minor updates**

- Default log directory has been added as ``config.log_dir``.
- Float datatype can be configured to float32 using ``set_config(float_type='float32')``.
- New file format, parquet, has been added. To change the file format to parquet,
  do ``set_config(file_format='parquet')``. The default file format is pickle.
- ``log.set_log_path()`` has been revised so that it can create a log file automatically from a file name.
- ``datatools.classify()`` has been revised so that if the characteristic is a binary variable, the class is either
  0 or (number of quantiles - 1). In the previous version, the class was not deterministic.
- ``jkp.py`` has been renamed as ``factors.py``.
- ``analytics.rolling_beta()`` has been renamed as ``numba_support.rolling_regression()``.
- ``panel.Panel.rolling_beta()`` has been renamed as ``panel.Panel.rolling_regression()``.
- Input arguments have been changed in the following functions.

    - ``datatools.classify()``
    - ``datatools.trim()``
    - ``datatools.filter()``
    - ``datatools.winsorize()``

- A new argument `fname` has been added to ``load_data()`` of ``FUNDA``, ``FUNDQ``, ``CRSPM``, and ``CRSPD``.
  If funda, fundq, crspm, and crspd data are modified (e.g., cleansed) and saved with different file names,
  those names can be given to read data from those modified data files.

- ``mapping.xlsx``: New columns, original sample start date (sample_start) and original sample end date (sample_end),
  have been added.

**New functions**

    - ``analytics.grs_test()``: GRS (Gibbons, Ross, and Shanken, 1989) test.
    - ``config.set_config()``: Set library configuration.
    - ``config.get_config()``: Get library configuration.
    - ``datatools.apply_to_groups()``: Group data and apply a function to each group.
    - ``datatools.apply_to_groups_jit()``: Group data and apply a function to each group (jitted version).
    - ``datatools.apply_to_groups_reduce_jit()``: Group data and apply a reduce function to each group (jitted version).
    - ``numba_support.roll_sum()``: Rolling sum.
    - ``numba_support.roll_mean()``: Rolling mean.
    - ``numba_support.roll_std()``: Rolling standard deviation.
    - ``numba_support.roll_var()``: Rolling variance.
    - ``numba_support.rank()``: Rank.
    - ``numba_support.bivariate_regression()``: Bivariate regression.
    - ``numba_support.regression()``: Multivariate regression.
    - ``numba_support.rolling_regression()``: Rolling regression.
    - ``panel.Panel.apply_to_ids()``: Apply a function to each id group.
    - ``panel.Panel.apply_to_dates()``: Apply a function to each date group.
    - ``wrdsdata.WRDS.create_crsp_comp_linktable()``: Create a CRSP-Compustat link table using cusip.
    - ``wrdsdata.WRDS.add_gvkey_to_crsp_cusip()``: Add gvkey to m(d)sf and identify primary stocks using internal link table.

**Deprecated functions**

    - ``characteristics.FUNDA.convert_to_monthly()``: Use ``Panel.populate()`` instead.
    - ``characteristics.FUNDQ.convert_to_monthly()``: Use ``Panel.populate()`` instead.
    - ``datatools.filter_n()``.
    - ``datatools.groupby_apply()``: Use ``datatools.apply_to_groups()``, ``datatools.apply_to_groups_jit()``, or
      ``datatools.apply_to_groups_reduce_jit()``.
    - ``datatools.groupby_apply_np()``: Use ``datatools.apply_to_groups()``, ``datatools.apply_to_groups_jit()``, or
      ``datatools.apply_to_groups_reduce_jit()``.
    - ``datatools.rolling_apply()``: Use ``datatools.apply_to_groups()``, ``datatools.apply_to_groups_jit()``, or
      ``datatools.apply_to_groups_reduce_jit()``.
    - ``datatools.rolling_apply_np()``: Use ``datatools.apply_to_groups()``, ``datatools.apply_to_groups_jit()``, or
      ``datatools.apply_to_groups_reduce_jit()``.

**Bug fix**

    - ``characteristic.FUNDA.c_currat()``: A bug of not returning the result has been fixed.
    - ``characteristics.FUNDQ.c_ni_inc8q()``: In the previous version, dibq (difference of ibq) was set to nan in the
      first 4 quarters. This made some valid ni_inc8q in the first 12 quarters become nan. In the new version,
      we set all nan values of dibq to 0 before calculating ni_inc8q and ni_inc8q is set to nan if dibq is nan.
      The revised logic does not lose valid ni_inc8q in the first 12 quarters.
    - ``characteristic.CRSPD.zero_trades_21d()``: Fixed dividing by 0 when the total turnover is 0.
      When counting the number of days in a month, only the days when turnover is not nan are counted. Before, all days
      were counted.
    - ``characteristic.CRSPD.c_zero_trades_126d()``: Fixed dividing by 0 when the total turnover is 0.
    - ``characteristic.CRSPD.c_zero_trades_252d()``: Fixed dividing by 0 when the total turnover is 0.
    - ``characteristic.CRSPD.c_rmax5_21d()``: A bug when there are only a few distinct return values in a month has been
      fixed.
      Suppose the return is positive in two days and 0 in the other days. Previously, rmax5_21d was the mean of the
      two positive returns. In the new version, it is the mean of the two positive returns and three 0 returns.
      Also, if days of valid returns (not nan) are fewer than or equal to 5, the result is nan.
    - ``characteristic.Merge.age()``: In the previous version, age was the max of (funda history, crspm history).
      This logic can make the age decrease when funda history is missing: if funda data exists from 2000.01 to 2020.12
      and crsp data from 2001.01 to 2022.12, the age will decrease in 2021.01. The logic has been revised so that the
      age doesn't decrease when funda data is missing.
    - ``panel.Panel.rolling()``: When `lag` > 0, shifted rows were not properly removed. This bug has been fixed.

