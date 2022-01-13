.. _about:

About
============

PyAnomaly is a comprehensive python library for asset pricing research with a focus on firm characteristic and factor generation. 
It covers the majority of the firm characteristics published in the literature and contains various analytic tools that are 
commonly used in asset pricing research, such as quantile portfolio construction, factor regression, and cross-sectional regression.
The purpose of PyAnomaly is *NOT* to generate firm characteristics in a fixed manner. Rather, we aim to build
a package that can serve as a standard library for asset pricing research and help reduce *non-standard errors* [5]_.

The current list of firm characteristics supported by PyAnomaly can be found in `Coverage`_.
PyAnomaly is a live project and we plan to add more firm characteristics and functionalities going forward. We also welcome contributions
from other scholars.

PyAnomaly is very efficient, comprehensive, and flexible.

    Efficiency
        PyAnomaly can generate over 200 characteristics from 1950 in around one hour including the time to download data from WRDS.
        To achieve this, PyAnomaly utilizes numba, multiprocessing, and asyncio packages when possible, but not too heavily to maximize readability of the code.

    Comprehensiveness
        PyAnomaly supports over 200 firm characteristics published in the literature. It covers most characteristics in
        Green et al. (2017)\ [2]_ and Jensen et al. (2021)\ [4]_, except those that use IBES data. It also provides
        various tools for asset pricing research.

    Flexibility
        PyAnomaly adopts the object-oriented programming design philosophy and is easy to customize or add functionalities.
        This means users can easily change the definition of an existing characteristic, add a new characteristic, or
        change configurations to run the program. For instance, a user can choose whether to update annual accounting
        variables quarterly (using Compustat.fundq) or annually (using Compustat.funda), or whether
        to use the latest market equity or the year-end market equity, when generating firm characteristics.


Main Features
=============

* Efficient data download from WRDS using asynco.
* Over 200 firm characteristics generation. You can choose which firm characteristics to generate.
* Fama-French 3-factor and Hou-Zhou-Zhang 4-factor portfolios.
* Analytics

    * Cross-section regression
    * 1-D sort
    * 2-D sort
    * Rolling regression
    * Quantile portfolio
    * Long-short portfolio
    * Portfolio performance analysis

* Data tools

    * Data filtering
    * Winsorizing
    * Trimming
    * Data population


Coverage
============

Markets
-------

PyAnomaly currently supports analysis of the firms listed in the US stock market.

Firm Characteristics
--------------------
The table below lists firm characteristics that are currently supported by PyAnomaly. The characteristics without a function
are not yet available but may be added in the future. For a mapping between the functions and the firm characteristics in
Chen and Zimmermann (2020)\ [1]_, Green et al. (2017)\ [2]_, Hou et al. (2020)\ [3]_, and Jensen et al. (2021)\ [4]_,
refer to the `mapping file`_,

.. csv-table::
    :widths: 5, 45, 15, 5, 5, 25
    :header: "", "Description", "Author(s)", "Year", "Journal", "Function"
    :file: characteristics.csv

Structure
============

PyAnomaly consists of various modules and the core modules you are likely to use are as follows.
The full list of the modules and their details can be found in the API documentation (:ref:`pyanomaly`).

* ``wrdsdata.py``: ``WRDS``, a class to handle data downloading from WRDS is defined here.
* ``panel.py``: ``Panel``, a base class to handle panel data is defined here.
* ``characteristics.py``: Classes to generate firm characteristics are defined here. These classes are derived from ``Panel``.

    * ``FUNDA``: A class to generate firm characteristics from funda.
    * ``FUNDQ``: A class to generate firm characteristics from fundq.
    * ``CRSPM``: A class to generate firm characteristics from crspm.
    * ``CRSPD``: A class to generate firm characteristics from crspd.
    * ``Merge``: A class to generate firm characteristics from a merged dataset of funda, fundq, crspm, and crspd.
* ``analytics.py``: A module that defines functions for analytics, such as 1-D sort, cross-sectional regression, and time-series regression.
* ``datatools.py``: A module that defines functions for data handling, such as data filtering, trimming, and winsorizing.


System Requirement
==================

Recommendation
    - Disc space: minimum 100 GB
    - Memory: minimum 64 GB

The minimum system requirement depends on the configuration, e.g., what characteristics to generate or the sample period.

Disc space
    The raw data downloaded from WRDS take up about 27GB of the disc space. The final output file can take
    up to 15GB if all characteristics are generated and the raw data are saved together.
    The size of the output file can be significantly reduced if only the firm characteristics are saved (less than 5 GB).
    In general, 100GB should be sufficient in all types of tasks and even when interim results are saved.

Memory
    Generating firm characteristics from daily data such as crspd consumes a significant amount of memory. The memory
    usage can be as much as 50 GB at a peak. This does not mean you need a physical
    memory of this size. Most OS will use Paging File to allocate some of the disc space as memory,
    although using Paging File will increase the running time.


Comparison to Other Sources
================================

PyAnomaly benefits greatly from the SAS codes of Green et al. (2017) and Jensen et al. (2021).
We generally follow the SAS codes and validate our code against them, but when their implementation is
significantly different from the original definition, we try to follow the original definition.
We also found several mistakes in these codes. For those mistakes we found and the differences between our implementation
and theirs, we make a note in the `mapping file`_ and comments in the code.
The SAS code of Jensen et al. (2021) has been updated several times
while we develop PyAnomaly and some

Comparison to the SAS code of Jensen et al. (2021)
--------------------------------------------------

PyAnomaly can be configured so that it replicates JKP's SAS code as closely as possible.
However, Having said that...

* Market equity

* Merging FUNDA with FUNDQ


The code is mainly based on the SAS codes of GHZ and JKP: when the implementation of a firm characteristic is significantly different between the two sources, we implement both implementations.


References
==========

.. [1] `Chen, A.Y. and Zimmermann, T., 2020. Open source cross-sectional asset pricing. Critical Finance Review, Forthcoming. <https://cfr.pub/forthcoming/papers/chen2021open.pdf>`__
.. [2] `Green, J., Hand, J.R. and Zhang, X.F., 2017. The characteristics that provide independent information about average US monthly stock returns. The Review of Financial Studies, 30(12), pp.4389-4436. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2262374>`__
.. [3] `Hou, K., Xue, C. and Zhang, L., 2020. Replicating anomalies. The Review of Financial Studies, 33(5), pp.2019-2133. <http://theinvestmentcapm.com/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf>`__
.. [4] `Jensen, T.I., Kelly, B.T. and Pedersen, L.H., 2021. Is There A Replication Crisis In Finance? Journal of Finance, Forthcoming. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514>`__
.. [5] `Menkveld, A.J., Dreber, A., Holzmeister, F., Huber, J., Johannesson, M., Kirchler, M., Neusüss, S., Razen, M. and Weitzel, U., 2021. Non-standard errors. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3961574>`__

Useful Links
============

- PyAnomaly repository: https://github.com/chulwoohan/pyanomaly
- JKP's SAS code: https://github.com/bkelly-lab/ReplicationCrisis
- openassetpricing: https://www.openassetpricing.com/
- GHZ' SAS code: https://sites.google.com/site/jeremiahrgreenacctg/home

.. _PyAnomaly repository: https://github.com/chulwoohan/pyanomaly
.. _mapping file: https://github.com/chulwoohan/pyanomaly/blob/master/mapping.xlsx
.. _CZ's openassetpricing: https://www.openassetpricing.com/
.. _GHZ' SAS code: https://sites.google.com/site/jeremiahrgreenacctg/home
.. _JKP's SAS code: https://github.com/bkelly-lab/ReplicationCrisis


Glossary
=============

* **crspd**: CRSP daily data created from dsf, dsenames, and dseall.
* **crspm**: CRSP monthly data created from msf, msenames, and mseall.
* **funda**: Compustat annual accounting data created from funda.
* **fundq**: Compustat quarterly accounting data created from fundq.

* **CZ**: Either the paper or the R/Stata code of Chen and Zimmermann (2020).
* **GHZ**: Either the paper or the SAS code of Green, Hand, and Zhang (2017).
* **HXZ**: Hou, Xue, and Zhang (2020).
* **JKP**: Either the paper or the SAS code of Jensen, Kelly, and Pedersen (2021).

Featured In
============

