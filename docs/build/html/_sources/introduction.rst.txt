.. _about:

About
============

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


Coverage
============

Markets
-------

PyAnomaly currently supports analysis of the firms listed in the US stock market.

Firm Characteristics
--------------------
The table below lists firm characteristics that are currently supported by PyAnomaly. The characteristics without a function
are not yet available but may be added in the future. For a mapping between the functions and the firm characteristics in
Chen and Zimmermann (2022)\ [1]_, Green et al. (2017)\ [2]_, Hou et al. (2020)\ [3]_, and Jensen et al. (2023)\ [4]_,
refer to the `mapping file`_,

.. csv-table::
    :widths: 5, 45, 15, 5, 5, 25
    :header: "", "Description", "Author(s)", "Year", "Journal", "Function"
    :file: characteristics.csv

Structure
============

PyAnomaly consists of various modules and the core modules you are likely to use are as follows.
The full list of the modules and their details can be found in :ref:`API <pyanomaly>`.

* ``wrdsdata.py``: ``WRDS``, a class for downloading WRDS data, is defined here.
* ``panel.py``: ``Panel``, a base class for panel data analysis, and ``FCPanel``, a base class for firm characteristic generation, are defined here.
* ``characteristics.py``: Classes for firm characteristic generation are defined here. These classes are derived from ``FCPanel``.

    * ``FUNDA``: Class to generate firm characteristics from funda.
    * ``FUNDQ``: Class to generate firm characteristics from fundq.
    * ``CRSPM``: Class to generate firm characteristics from crspm.
    * ``CRSPD``: Class to generate firm characteristics from crspd.
    * ``Merge``: Class to generate firm characteristics from a merged dataset of funda, fundq, crspm, and crspd.
* ``analytics.py``: Analytic functions such as 1-D sort, cross-sectional regression, and time-series regression are defined here.
* ``datatools.py``: Data handling functions such as grouping, trimming, and winsorization are defined here.


System Requirement
==================

Recommendation
    - Disc space: minimum 50 GB
    - Memory: minimum 64 GB

The minimum system requirement depends on the configuration, e.g., which characteristics to generate or the sample period.

Disc space
    The raw data downloaded from WRDS (until 2022.12) take up about 29 GB (if pickle file format is used) or
    3.5 GB (if parquet file format is used) of the disc space. The final output file can take
    up to 14 GB (pickle) or 7 GB (parquet) if all the firm characteristics are generated and the raw data are saved
    together. The sizes can be almost halved if float32 is used instead of float64
    (``set_config(float_type='float32')``). The size of the output file can also be significantly reduced if only the
    firm characteristics are saved (less than 5 GB). In general, 100 GB should be sufficient for all types of tasks and
    even when interim results are saved.

Memory
    Generating firm characteristics from daily data such as crspd consumes a significant amount of memory.
    During firm characteristics generation, the memory usage can reach as high as 43 GB (31 GB if float32 is used).
    This, however, does not mean a physical memory of this size is required. Most OS will use Paging File to allocate
    some of the disc space as memory, although using Paging File will increase the running time.

Note on file format
    A parquet file can be significantly smaller than a pickle file of the same data,
    especially when there are many duplicate values in columns.
    However, it tends to be slower to read and write and takes significantly more memory in some cases for unknown
    reasons (almost three times of the memory consumed by a pickle file of the same data).
    If parquet did not have a memory issue, we would have chosen it as the default file format.
    If you do not encounter the same issue in your environment, you may choose to use parquet to save disc space.
    To change the file format to parquet, do ``set_config(file_format='parquet')``.

Comparison to Other Sources
================================

PyAnomaly benefits greatly from the SAS codes of Green et al. (2017) and Jensen et al. (2023), and also from the papers
and documentations of Hou et al. (2020) and Chen and Zimmermann (2022).
We generally follow the SAS codes of JKP and GHZ and validate our code against them. Nevertheless, if their implementation is
significantly different from the original definition, we try to follow the original definition.
When the implementation of a firm characteristic is significantly different between the two sources,
we implement both implementations using different function names.
We also find several mistakes in these codes. The mistakes we find and the differences between our implementation
and theirs are documented in the `mapping file`_ and comments in the code.
The SAS code of Jensen et al. (2021) has been updated several times
while we develop PyAnomaly and some of our comments may no longer be valid.

Comparison to the SAS code of Jensen et al. (2021)
--------------------------------------------------

PyAnomaly can be configured so that it replicates JKP's SAS code as closely as possible (``set_config(replicate_jkp=True``).
However, there are a few key differences that make our results differ from theirs.

Market equity
    JKP use not only CRSP's msf but also Compustat's secm and secd to calculate market equity,
    and (roughly speaking) choose the maximum market equity among those calculated from different sources.
    We only use the price and shares outstanding from CRSP to calculate the market equity.

Merging FUNDA with FUNDQ
    JKP quarterly-update annual accounting variables using comp.fundq. More specifically,
    JKP create same characteristics in funda and fundq separately and merge them. On the other hand, we merge the raw data
    first and then generate characteristics. Since some variables in funda are not available in fundq, e.g., ebitda,
    JKP synthesize those unavailable variables from other variables and create characteristics, even when they are
    available in funda. We prefer to merge funda with fundq at the raw data level and create characteristics from
    the merged data.

Share code filtering
    JKP do not filter data using CRSP share code (shrcd), whereas we only use ordinary common stocks
    (shrcd = 10, 11, or 12). We find that some stocks' shrcd changes over time. Therefore, this difference does not
    only affect the cross-section but also time-series. In PyAnomaly, filtering rules can easily be overriden by inheriting
    firm characteristic generation classes.


References
==========

.. [1] `Chen, A.Y. and Zimmermann, T., 2022. Open source cross-sectional asset pricing. Critical Finance Review, 27(2). <https://cfr.pub/forthcoming/papers/chen2021open.pdf>`__
.. [2] `Green, J., Hand, J.R. and Zhang, X.F., 2017. The characteristics that provide independent information about average US monthly stock returns. The Review of Financial Studies, 30(12). <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2262374>`__
.. [3] `Hou, K., Xue, C. and Zhang, L., 2020. Replicating anomalies. The Review of Financial Studies, 33(5). <http://theinvestmentcapm.com/uploads/1/2/2/6/122679606/houxuezhang2020rfs.pdf>`__
.. [4] `Jensen, T.I., Kelly, B.T. and Pedersen, L.H., 2023. Is There A Replication Crisis In Finance? Journal of Finance, 78(5). <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3774514>`__
.. [5] `Menkveld, A.J. el al., 2024. Non-standard errors. Journal of Finance, Forthcoming. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3961574>`__

Useful Links
============

- PyAnomaly repository: https://github.com/chulwoohan/pyanomaly
- JKP's SAS code: https://github.com/bkelly-lab/ReplicationCrisis
- Openassetpricing: https://www.openassetpricing.com/
- GHZ' SAS code: https://sites.google.com/site/jeremiahrgreenacctg/home


Glossary
=============

* **crspd**: CRSP daily data created from dsf, dsenames, and dseall.
* **crspm**: CRSP monthly data created from msf, msenames, and mseall.
* **funda**: Compustat annual accounting data created from funda.
* **fundq**: Compustat quarterly accounting data created from fundq.

* **CZ**: Either the paper or the R/Stata code of Chen and Zimmermann (2022).
* **GHZ**: Either the paper or the SAS code of Green, Hand, and Zhang (2017).
* **HXZ**: Hou, Xue, and Zhang (2020).
* **JKP**: Either the paper or the SAS code of Jensen, Kelly, and Pedersen (2023).


Contributors
============
* Chulwoo Han
* Jongho Kang
* Byeongguk Kang


Featured In
============

