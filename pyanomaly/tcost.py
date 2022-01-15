"""This module defines classes for transaction costs.
"""
import pandas as pd

from pyanomaly.globals import *


class TransactionCost:
    """Transaction cost class.

    Trnansaction cost can be set at the security level. It can also vary over time. See ``set_params()`` for details.

    Attributes:
        params: Transaction cost parameters. This can be a float number, dict, or DataFrame. See ``set_params()``.
    """

    def __init__(self, **kwargs):
        self.params = None
        if kwargs:
            self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """Set transaction cost parameters.

        Parameters can be set either by this method or by ``__init__()``.

        Args:
            kwargs: `kwargs` can have the following keywords:

                - 'cost': For a constant (scalar) transaction cost.
                - 'buy_fixed`, 'buy_linear`, 'buy_quad', 'sell_fixed', 'sell_linear', 'sell_quad': For
                  a quadratic transaction cost.
                - 'params': For a transaction cost that varies across securities (and over time).

        Below are some examples.

        For a constant transaction cost of 20 basis points:

        >>> tc = TransactionCost(cost=0.002)

        Asymmetric quadratic cost function:

            `cost = fixed + linear * Amount + quad * Amount^2`

            - To buy: fixed cost = $5, linear cost = 0.002, and quadratic cost = 0.001
            - To sell: fixed cost = 0, linear cost = 0.003, and quadratic cost = 0.001

        >>> tc = TransactionCost(buy_fixed=5, buy_linear=0.002, buy_quad=0.001, sell_linear=0.003, sell_quad=0.001)

            Only non-zero parameters need to be provided.

        Transaction costs that vary across securities:

            - Security 1 (id: 0001): 0.002, security 2 (id: 0002): 0.003

        .. code-block::

            >>> params = pd.DataFrame({'cost': [0.002, 0.003]}, index=pd.Index(['0001', '0002'], name='id'))
            >>> params
                  cost
            id
            0001 0.002
            0002 0.003
            >>> tc = TransactionCost(params=params)

        Transaction costs that vary across securities and dates:

            - Security 1 (id: 0001): 0.004 on '2000-03-31', 0.003 on '2000-04-30'
            - Security 2 (id: 0002): 0.005 on '2000-03-31', 0.004 on '2000-04-30'

        .. code-block::

            >>> dates = ['2000-03-31', '2000-04-30']
            >>> ids = ['0001', '0002']
            >>> params = pd.DataFrame(index=pd.MultiIndex.from_product([dates, ids], names=('date', 'id')))
            >>> params['cost'] = [0.004, 0.005, 0.003, 0.004]
            >>> params
                             cost
            date       id
            2000-03-31 0001 0.004
                       0002 0.005
            2000-04-30 0001 0.003
                       0002 0.004
            >>> tc = TransactionCost(params=params)

        The `params` DataFrame must have index = 'id' or 'date'/'id'. It can have columns such as 'buy_fixed' instead
        of 'cost' for a more complex transaction cost structure.

        """
        if 'cost' in kwargs:
            self.params = kwargs['cost']
        elif 'params' in kwargs:
            self.params = kwargs['params']
        else:
            self.params = dict(
                buy_fixed=0,
                buy_linear=0,
                buy_quad=0,
                sell_fixed=0,
                sell_linear=0,
                sell_quad=0,
            )
            for k in self.params:
                if k in kwargs:
                    self.params[k] = kwargs[k]

    @staticmethod
    def get_cost_(val, val0, params):
        """Get a quadratic transaction cost

        Args:
            val: Value after rebalancing.
            val0: Value before rebalancing.
            params (dict): Transaction cost parameters.

        Returns:
            Transaction cost.
        """

        if val > val0:  # buy
            return params['buy_fixed'] + params['buy_linear'] * (val - val0) + params['buy_quad'] * (val - val0) ** 2
        else: # sell
            return params['sell_fixed'] + params['sell_linear'] * (val0 - val) + params['sell_quad'] * (val0 - val) ** 2

    def get_cost(self, position):
        """Get transaction costs.

        Args:
            position: Portfolio positions. ``Portfolio`` object calls this function with the argument,
                ``Portfolio.position``, to get transaction costs. The `position` argument should have index = 'date' and
                columns = [`id`, 'val', 'val0'], where 'val' is the value after rebalancing and 'val0' is the value
                before rebalancing.

        Returns:
            Transaction costs. A vector with the same length as `position`.
        """

        if self.params is None:
            return 0
        elif isinstance(self.params, float):
            return self.params * np.abs(position.val - position.val0)
        elif isinstance(self.params, dict):
            vcostfcn = np.vectorize(TransactionCost.get_cost_)
            return vcostfcn(position.val, position.val0, self.params)
        else:
            if isinstance(self.params.index, pd.MultiIndex):  # date/id
                tmp = pd.merge(position[['id', 'val', 'val0']], self.params, how='left', on=['date', 'id'])
            else:  # id
                tmp = pd.merge(position[['id', 'val', 'val0']], self.params, how='left', on='id')

            if 'cost' in tmp:
                tmp['cost'] = tmp['cost'] * np.abs(tmp['val'] - tmp['val0'])
            else:
                tmp['buy_cost'] = 0
                if 'buy_fixed' in tmp:
                    tmp['buy_cost'] += tmp['buy_fixed']
                elif 'buy_linear' in tmp:
                    tmp['buy_cost'] += tmp['buy_linear'] * (tmp['val'] - tmp['val0'])
                elif 'buy_quad' in tmp:
                    tmp['buy_cost'] += tmp['buy_quad'] * (tmp['val'] - tmp['val0']) ** 2

                tmp['sell_cost'] = 0
                if 'sell_fixed' in tmp:
                    tmp['sell_cost'] += tmp['sell_fixed']
                elif 'sell_linear' in tmp:
                    tmp['sell_cost'] += tmp['sell_linear'] * (tmp['val0'] - tmp['val'])
                elif 'sell_quad' in tmp:
                    tmp['sell_cost'] += tmp['sell_quad'] * (tmp['val0'] - tmp['val']) ** 2

                tmp['cost'] = np.where(tmp['val'] > tmp['val0'], tmp['buy_cost'], tmp['sell_cost'])

            return tmp.cost.values


class TimeVaryingCost(TransactionCost):
    """Transaction costs that vary over time and across firm sizes.

    This class implements the time-varying transaction costs used in, e.g., Brandt et al. (2009), Hand and Green (2011),
    DeMiguel et al. (2020), and Han (2021). Transaction cost parameter `k` is defined as `k` = `y` * `z`,
    where `y` decreases linearly from 4.0 in 1974.01 to 1.0 in 2002.01 and remains at 1.0 thereafter, and
    `z` = 0.006 - 0.0025 * `nme`, where `nme` is the normalized market equity that has a value between 0 and 1.

    The maximum transaction cost is 240 basis points (the smallest firm before 1974) and the minimum transaction cost is
    35 basis points (the largest firm after 2002).

    We find this assumption is too conservative since the normalized me is sensitive to the largest company.
    In 1974, the mean of the normalized me is only 0.0045 and most firms have the transaction cost of 240 basis points,
    and in 2002, the mean of the normalized me is only 0.0059 and most firms have the transaction cost of 60 basis points.

    Using a logarithm of the market equity or capping the me of the largest firms may make more sense.
    """

    def __init__(self, me=None, normalize=True):
        super().__init__()
        if me is not None:
            self.set_params(me, normalize)

    def set_params(self, me, normalize=True):
        """Set transaction cost parameters.

        Args:
            me: DataFrame or Series of market equity with index = date/id.
            normalize: If True, normalize `me` so that its values are between 0 and 1.

        NOTE:
            If the input contains only a subset of all listed stocks, normalizing the market equity can result in
            over- or underestimation of the transaction costs. For example, if `me` contains only top 80% of the stocks,
            the transaction costs will be overestimated. Use all listed stocks in the market or normalize the market equity
            outside and set `normalize = False`.
        """

        date1 = pd.Timestamp('1974-01-01')
        date2 = pd.Timestamp('2002-01-01')
        dates = me.index.get_level_values(0)

        if isinstance(me, pd.Series):
            params = me.to_frame()
        else:
            params = me.copy()
        params.index = params.index.set_names(('date', 'id'))

        if normalize:
            params['nme'] = params.groupby('date').transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        else:
            params['nme'] = params.iloc[:, 0]
        params['y'] = 1 + (4 - 1) * (date2 - dates) / (date2 - date1)
        params.loc[dates < date1, 'y'] = 4.0
        params.loc[dates > date2, 'y'] = 1.0
        params['cost'] = params['y'] * (0.006 - 0.0025 * params['nme'])

        self.params = params[['cost']]

