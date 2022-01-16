==========
PyAnomaly
==========

PyAnomaly is a comprehensive python library for asset pricing research with a focus on firm characteristic and factor generation.
It covers the majority of the firm characteristics published in the literature and contains various analytic tools that are 
commonly used in asset pricing research, such as quantile portfolio construction, factor regression, and cross-sectional regression.
The purpose of PyAnomaly is *NOT* to generate firm characteristics in a fixed manner. Rather, we aim to build
a package that can serve as a standard library for asset pricing research and help reduce *non-standard errors*.

PyAnomaly is a live project and we plan to add more firm characteristics and functionalities going forward. We also welcome contributions
from other scholars.

PyAnomaly is very efficient, comprehensive, and flexible.

    Efficiency
        PyAnomaly can generate over 200 characteristics from 1950 in around one hour including the time to download data from WRDS.
        To achieve this, PyAnomaly utilizes numba, multiprocessing, and asyncio packages when possible, but not too heavily to maximize readability of the code.

    Comprehensiveness
        PyAnomaly supports over 200 firm characteristics published in the literature. It covers most characteristics in
        Green et al. (2017) and Jensen et al. (2021), except those that use IBES data. It also provides
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
* Fama-French 3-factor and Hou-Xue-Zhang 4-factor portfolios.
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

