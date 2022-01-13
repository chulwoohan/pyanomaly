""""This module defines `Portfolo` and `Portfolios` classes that are used for portfolio analysis.
"""

from pyanomaly.globals import *
from pyanomaly.tcost import TransactionCost

class Portfolio:
    """Portfolio class.

    This class makes a portfolio from positions and evaluate it.

    Position information is saved in `position` DataFrame and portfolio information is saved in `value` DataFrame.
    If positive weights don't add up to 1, (1 - sum(positive weights)) will be assumed to be invested in a
    risk free asset, and its information is saved in `fposition` DataFrame.
    The transaction cost is assumed to be 0 for risk-free assets.
    Once the portfolio is evaluated by calling ``Portfolio.eval()``, `performance` DataFrame is generated.

    Args:
        name: Portfolio name.
        position: Position DataFrame. It should have index = 'date' and columns = 'id' (security ID) 'ret' (return),
            and 'wgt' (portfolio weight). If it has other columns such as price, they will be kept in the `position`
            attribute. If it has 'rf' (risk-free rate) column, its values are used as risk-free rates.
        rf: Risk-free rate DataFrame with index = 'date' and columns = 'rf'. `rf` have priority over 'rf' column in
            `position`. If `rf` = None and `position` does not have 'rf' column, the risk-free is assumed to be 0.
        pfval0: Initial portfolio value. Default to 1.
        costfcn: `TransactionCost` class, a transaction cost function, or value. If a const transaction cost of 20
            basis points is assumed, this can be set to 0.002.
        keep_position: If False, position information (`position`) is deleted.

    Attributes:
        name: Portfolio name.
        position: Position DataFrame. Its index is `date` and has the following columns:

            - 'id': security ID provided by the user.
            - 'ret': Return between t-1 and t.
            - 'exret': Excess return over risk-free rate between t-1 and t.
            - 'wgt`: Weight ot the beginning of t.
            - 'val1': Value at t.
            - 'val': Value at the beginning of t.
            - 'val0': value at t-1.
            - 'cost': Transaction cost incurred at the beginning of t.
            - Other columns: Any columns that are in the input position data, such as 'permno'.

            In the description above, 'at the beginning of t` means at t-1 after rebalancing.

        value: Portfolio value DataFrame. Its index is `date` and has the following columns:

            - 'ret': Return between t-1 and t. This can be either net return or gross return depending on
              the evaluation method: see ``Portfolio.eval()``.
            - 'exret': Excess return over risk-free rate between t-1 and t. This can be either net excess return or
              gross excess return depending on the evaluation method.
            - 'val1': Value at t.
            - 'val': Value at the beginning of t.
            - 'cost': Transaction cost incurred at the beginning of t.
            - 'tover': Turnover incurred at the beginning of t. `tover` = sum(pos.val - pos.val0) / sum(pf.val),
              where pos.val is the position values and pf.val is the portfolio value.
            - 'netret': Return between t-1 and t, net of transaction cost
            - 'grossret': Gross return between t-1 and t.
            - 'netexret': Excess return between t-1 and t, net of transaction cost
            - 'grossexret': Excess gross return between t-1 and t.
            - 'lposition': Number of long positions.
            - 'sposition': Number of short positions.

            Columns added once the portfolio is evaluated by calling ``Portfolio.eval()``:

            - 'cumret': Cumulative return since the first date.
            - 'drawdown': Drawdown.
            - 'drawdur': Duration of daawdown in the frequency of data, e.g., if rebalanced monthly, 3 means 3 months.
            - 'drawstart': Beginning date of drawdown.
            - 'succdown': Successive down; down without any up during the period.
            - 'succdur': Duration of successive down.
            - 'succstart' Beginning date of successive down.

        fposition: Risk-free asset position DataFrame. Its index is 'date' and has the following columns:

            - 'ret': Return between t-1 and t.
            - 'wgt`: Weight ot the beginning of t.
            - 'val1': Value at t.
            - 'val': Value at the beginning of t.

        performance: Portfolio performance DataFrame. Its column is equal to the portfolio name and has the following
            indexes:

            - 'mean': Mean excess return over the evaluation period.
            - 'std': Standard deviation of the excess returns over the evaluation period.
            - 'sharpe': Sharpe ratio.
            - 'cum': : Cumulative return over the evaluation period.
            - 'mdd': Maximum drawdown.
            - 'mdd start': Maximum drawdown start date.
            - 'mdd end': Maximum drawdown end date.
            - 'msd': Maximum successive down.
            - 'msd start': Maximum successive down start date.
            - 'msd end': Maximum successive down end date.
            - 'turnover': Average turnover.
            - 'lposotion': Average number of long positions.
            - 'sposition': Average number of short positions.

        costfcn: `TransactionCost` object, a transaction cost function, or value. For example, if a const transaction
            cost of 20 basis points is assumed, `costfcn` can be simply set to 0.002.

            If `costfcn` is a `TransactionCost` object, ``TransactionCost.get_cost(position)`` is called to get
            transaction costs. `TransactionCost` allows transaction costs varying across time and securities.
            See ``pyanomaly.tcost`` module for more details.

            A transaction cost function should have arguments val (value after rebalancing) and val0
            (value before rebalancing). For example, if the transaction cost to buy (sell) is 20 (30) bps,
            a function can be defined as follows:

            .. code-block::

                def cost_fcn(val, val0):
                    return 0.002 if val > val0 else 0.003

    NOTE:
        If a position exists at t-1 but not at t, it will be added at t with 0 weight.
        This is to compute the transaction cost.

        'val', 'val1', 'val0' in `position` and `value` are calculated without taking transaction costs into account.
        For a value increase after transaction costs, refer to the cumulative return.
    """

    def __init__(self, name=None, position=None, rf=None, pfval0=1, costfcn=None, keep_position=False):
        self.name = name
        self.position = None
        self.fposition = None
        self.value = None
        self.performance = None
        self.costfcn = costfcn

        if position is not None:
            self.set_position(position, rf, pfval0, keep_position)

    @staticmethod
    def _add_sold_position(position):
        """Add sold positions.

        When a stock is excluded from the portfolio, a position with val1=0 should be added for
        correct transaction cost calculation.
        """

        dates = np.sort(position.index.get_level_values(0).unique())
        dates = pd.DataFrame(dates, columns=['date'])
        dates['date0'] = dates.date.shift(1)
        position0 = position.merge(dates, left_on='date', right_on='date0')
        position0 = position0.drop(columns=['date0']).set_index('date')
        position = position.merge(position0['id'], 'outer', on=['date', 'id'], sort=True)
        position = position.fillna(0)
        return position

    @staticmethod
    def _make_portfolio_from_weight(position, rf=None, pfval0=1):
        """Make a portfolio give position weights.
        """

        position = position.copy()
        dates = np.sort(position.index.get_level_values(0).unique())

        # Initialize risk-free position and portfolio.
        fposition = pd.DataFrame(0., columns=['ret', 'wgt', 'val1', 'val'], index=dates)
        if rf is not None:
            fposition['ret'] = rf
            position['rf'] = rf
            position['exret'] = position['ret'] - position['rf']
        elif 'rf' in position:
            fposition['ret'] = position['rf'].groupby('date').first()
            position['exret'] = position['ret'] - position['rf']
        else:
            position['exret'] = position['ret']

        portfolio = pd.DataFrame(0., columns=['ret', 'exret', 'val1', 'val'], index=dates)

        position['pwgt'] = np.where(position.wgt >= 0, position.wgt, 0)
        position['wgtret'] = position.wgt * position.ret
        position['wgtexret'] = position.wgt * position.exret

        gbd = position.groupby('date')

        fposition.wgt = 1 - gbd.pwgt.sum()

        portfolio.ret = gbd.wgtret.sum() + fposition.wgt * fposition.ret
        portfolio.exret = gbd.wgtexret.sum() + fposition.wgt * fposition.ret
        position.drop(columns=['pwgt', 'wgtret', 'wgtexret'], inplace=True)

        portfolio.val1 = pfval0 * (1 + portfolio.ret).cumprod()
        portfolio.val = portfolio.val1.shift(1)
        portfolio.val.values[0] = pfval0

        portfolio['lposition'] = position.loc[position['wgt'] > 0, 'wgt'].groupby('date').count().fillna(0)
        portfolio['lposition'].fillna(0, inplace=True)
        portfolio['sposition'] = position.loc[position['wgt'] < 0, 'wgt'].groupby('date').count().fillna(0)
        portfolio['sposition'].fillna(0, inplace=True)


        position['val'] = portfolio.val
        position['val'] *= position.wgt
        position['val1'] = position.val * (1 + position.ret)

        fposition.val = fposition.wgt * portfolio.val
        fposition.val1 = fposition.val * (1 + fposition.ret)
        fposition = fposition.fillna(0)

        position = Portfolio._add_sold_position(position)
        return portfolio, position, fposition

    @staticmethod
    def _add_transaction_cost(portfolio, position, fposition, costfcn=None):
        """Add transaction costs and turnover to position and portfolio.
        """

        position['val0'] = position.groupby('id').val1.shift(1).fillna(0)
        # Set the cost of the first date to 0
        date0 = portfolio.index.min()
        position.loc[date0, 'val0'] = position.loc[date0, 'val']
        position['dval'] = np.abs(position.val - position.val0)

        if costfcn is None:
            position['cost'] = 0
        elif callable(costfcn):
            vcostfcn = np.vectorize(costfcn)
            position['cost'] = vcostfcn(position.val, position.val0)
        elif isinstance(costfcn, TransactionCost):
            position['cost'] = costfcn.get_cost(position)
        else:  # constant transaction cost
            position['cost'] = costfcn * position.dval
        portfolio[['cost', 'tover']] = position.groupby('date')[['cost', 'dval']].sum()
        portfolio.tover = portfolio.tover / portfolio.val

        portfolio['netret'] = portfolio.ret - portfolio.cost / portfolio.val
        portfolio['grossret'] = portfolio.ret
        portfolio['netexret'] = portfolio.exret - portfolio.cost / portfolio.val
        portfolio['grossexret'] = portfolio.exret
        del position['dval']

        return portfolio, position, fposition

    def set_position(self, position, rf=None, pfval0=1, keep_position=False):
        """Set positions.

        This method sets `position` attribute from the input arguments.
        For the details of the input arguments, See the arguments of the initializer.
        """

        self.value, self.position, self.fposition = Portfolio._make_portfolio_from_weight(position, rf, pfval0)
        Portfolio._add_transaction_cost(self.value, self.position, self.fposition, self.costfcn)

        if not keep_position: self.position = None

    @staticmethod
    def from_portfolo_return(pfret, val=1):
        """Make a portfolio given portfolio returns.

        If portfolio returns are already known, this method can be used to construct a `Portfoliio` object
        and evaluate the portfolio.

        Args:
            pfret: DataFrame with index = 'date' and columns = 'ret'.

        Returns:
            `Portfolio` object.
        """

        pfval = pfret.copy()
        pfval['exret'] = pfval.ret
        pfval['val1'] = val * (1 + pfval.ret).cumprod()
        pfval['val'] = pfval.val1.shift(1)
        pfval.val.values[0] = val
        pfval['cost'] = 0
        pfval['tover'] = 0
        pfval['netret'] = pfval.ret
        pfval['grossret'] = pfval.ret
        pfval['netexret'] = pfval.exret
        pfval['grossexret'] = pfval.exret
        pfval['lposition'] = 1
        pfval['sposition'] = 0

        portfolio = Portfolio()
        portfolio.value = pfval
        return portfolio

    def add_position(self, position, rf=None):
        """Add new positions.

        This method assumes positions are added in order of time. All positions of a given date must be added together.
        The arguments have the same formats as those of the initializer.
        """

        if self.position is None:
            self.set_position(position, rf, keep_position=True)
            return

        val = self.value.val1[-1]  # val = val1 of last time
        portfolio, position, fposition = Portfolio._make_portfolio_from_weight(position, rf, val)

        date0 = self.position.index[-1]
        portfolio = pd.concat([self.value.loc[[date0]], portfolio], sort=False)
        position = pd.concat([self.position.loc[[date0]], position], sort=False)
        fposition = pd.concat([self.fposition.loc[[date0]], fposition], sort=False)
        portfolio, position, fposition = Portfolio._add_transaction_cost(portfolio, position,
                                                               fposition, self.costfcn)
        self.value = pd.concat([self.value, portfolio[portfolio.index > date0]])
        self.position = pd.concat([self.position, position[position.index > date0]])
        self.fposition = pd.concat([self.fposition, fposition[fposition.index > date0]])

    def copy(self, sdate=None, edate=None):
        """Copy this object for the given period.

        Args:
            sdate: Start date ('yyyy-mm-dd').
            edate: End date.

        Returns:
            `Portfolio` object.
        """
        pf = Portfolio()
        pf.name = self.name
        pf.value = self.value[sdate:edate].copy()
        pf.position = self.position[sdate:edate].copy()
        pf.fposition = self.fposition[sdate:edate].copy()
        pf.costfcn = self.costfcn

        return pf

    def set_return_type(self, consider_cost=True):
        """Determine which return (gross vs. net) to use for portfolio evaluation.

        If `consider_cost` is True, `value.ret` is set to net returns, otherwise, it is set to gross returns.

        Args:
            consider_cost: True to use returns net of transaction costs for evaluation.
        """

        if consider_cost:
            self.value.ret = self.value.netret
            self.value.exret = self.value.netexret
        else:
            self.value.ret = self.value.grossret
            self.value.exret = self.value.grossexret

    def returns(self, sdate=None, edate=None):
        """Get portfolio returns.

        Both `sdate` and `edate` are inclusive, i.e., the first return is the return over `sdate`-1 and `sdate`.

        Args:
            sdate: Start date.
            edate: End date.

        Returns:
            return Series with index = 'date'.
        """

        return self.value[sdate:edate].ret

    @staticmethod
    def _cum_returns(ret, logscale=True, zero_start=False):
        cumret = np.log(1 + ret).cumsum()
        if zero_start: cumret = cumret - cumret.iloc[0]
        if not logscale:
            cumret = np.exp(cumret) - 1
        return cumret

    def cum_returns(self, sdate=None, edate=None, logscale=True, zero_start=False):
        """Get portfolio cumulative returns.

        Both `sdate` and `edate` are inclusive, i.e., the first return is the return over `sdate`-1 and `sdate`.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale cumulative returns.
            zero_start: If True, the return at `sdate` is forced to 0.

        Returns:
            Cumulative return Series with index = 'date'.
        """

        return self._cum_returns(self.value[sdate:edate].ret, logscale, zero_start)

    def mean_return(self, sdate=None, edate=None):
        """Get portfolio mean return over the period.

        Args:
            sdate: Start date.
            edate: End date.

        Returns:
            Mean return.
        """

        return self.value[sdate:edate].ret.mean()

    def std_return(self, sdate=None, edate=None):
        """Get the standard deviation of the returns over the period.

        Args:
            sdate: Start date.
            edate: End date.

        Returns:
            Standard deviation of the returns.
        """
        return self.value[sdate:edate].ret.std()

    def cum_return(self, sdate=None, edate=None, logscale=True):
        """Get the cumulative return over the period.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale cumulative return.

        Returns:
            Cumulative return.
        """

        cumret = np.log(1 + self.value[sdate:edate].ret.values).sum()
        if not logscale:
            cumret = np.exp(cumret) - 1

        return cumret

    def sharpe_ratio(self, sdate=None, edate=None):
        """Get the Sharpe ratio over the period.

        If risk-free rates are not set, they are assumed to be 0.

        Args:
            sdate: Start date.
            edate: End date.

        Returns:
            Sharpe ratio.
        """

        er = self.value[sdate:edate].exret.values
        mean = er.mean()
        std = er.std()
        return mean / std if std != 0 else 0

    @staticmethod
    def _get_succdown(cumret):
        ret = np.diff(np.insert(cumret, 0, 0))
        succdown = np.zeros_like(cumret)
        duration = np.zeros_like(cumret)
        cumret0 = 0
        t0 = -1
        for t in range(len(cumret)):
            if ret[t] > 0:
                cumret0 = cumret[t]
                t0 = t
            succdown[t] = cumret0 - cumret[t]
            duration[t] = t - t0
        return succdown, duration

    @staticmethod
    def _succdown(cumret, logscale=True):
        value, duration = Portfolio._get_succdown(cumret.values)
        if not logscale: value = 1 - np.exp(-value)
        succdown = pd.DataFrame(np.transpose([value, duration]), columns=['value', 'duration'], index=cumret.index)
        idx = np.arange(len(duration)) - duration.astype(int)
        idx[idx < 0] = 0  # to prevent negative index (can happen when ret < 0 in the first month)
        succdown['start'] = cumret.index[idx]
        return succdown

    def succdown(self, sdate=None, edate=None, logscale=True):
        """Get the successive downs over the period.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale values.

        Returns:
            Successive downs: DataFrame with index = 'date' and columns = ['value', 'duration', 'start'].
        """

        return self._succdown(self.cum_returns(sdate, edate), logscale)

    def max_succdown(self, sdate=None, edate=None, logscale=True):
        """Get the maximum successive down over the period.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale value.

        Returns:
            Maximum successive down: Series with index = ['value', 'duration', 'start'].
        """

        succdown = self.succdown(sdate, edate, logscale)
        return succdown.loc[succdown.value.idxmax()]

    @staticmethod
    def _get_drawdown(cumret):
        drawdown = np.zeros_like(cumret)
        duration = np.zeros_like(cumret)
        cumret0 = 0
        t0 = -1
        for t in range(len(cumret)):
            if cumret[t] >= cumret0:
                cumret0 = cumret[t]
                t0 = t
            drawdown[t] = cumret0 - cumret[t]
            duration[t] = t - t0
        return drawdown, duration

    @staticmethod
    def _drawdown(cumret, logscale=True):
        value, duration = Portfolio._get_drawdown(cumret.values)
        if not logscale: value = 1 - np.exp(-value)
        drawdown = pd.DataFrame(np.transpose([value, duration]), columns=['value', 'duration'], index=cumret.index)
        idx = np.arange(len(duration)) - duration.astype(int)
        idx[idx < 0] = 0  # to prevent negative index (can happen when ret < 0 in the first month)
        drawdown['start'] = cumret.index[idx]
        return drawdown

    def drawdown(self, sdate=None, edate=None, logscale=True):
        """Get the drawdowns over the period.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale values.

        Returns:
            Drawdowns: DataFrame with index = 'date' and columns = ['value', 'duration', 'start'].
        """

        return self._drawdown(self.cum_returns(sdate, edate), logscale)

    def max_drawdown(self, sdate=None, edate=None, logscale=True):
        """Get the maximum drawdown over the period.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, return log-scale value.

        Returns:
            Maximum drawdown: Series with index = ['value', 'duration', 'start'].
        """

        drawdown = self.drawdown(sdate, edate, logscale)
        return drawdown.loc[drawdown.value.idxmax()]

    def eval(self, sdate=None, edate=None, logscale=True, annualize_factor=1, consider_cost=True, percentage=False):
        """Evaluate the portfolio over the period.

        This method evaluate the portfolio and create `performance` attribute. It also adds performance-related columns
        to `value` attribute.

        `performance`: DataFrame with columns = [`name`] and the following indexes:

            - 'mean': Mean excess return over the evaluation period.
            - 'std': Standard deviation of the excess returns over the evaluation period.
            - 'sharpe': Sharpe ratio.
            - 'cum': : Cumulative return over the evaluation period.
            - 'mdd': Maximum drawdown.
            - 'mdd start': Maximum drawdown start date.
            - 'mdd end': Maximum drawdown end date.
            - 'msd': Maximum successive down.
            - 'msd start': Maximum successive down start date.
            - 'msd end': Maximum successive down end date.
            - 'turnover': Average turnover.
            - 'lposotion': Average number of long positions.
            - 'sposition': Average number of short positions.

        Columns added to `value`:

            - 'cumret': Cumulative return since the first date.
            - 'drawdown': Drawdown.
            - 'drawdur': Duration of daawdown in the frequency of data, e.g., if rebalanced monthly, 3 means 3 months.
            - 'drawstart': Beginning date of drawdown.
            - 'succdown': Successive down; down without any up during the period.
            - 'succdur': Duration of successive down.
            - 'succstart' Beginning date of successive down.

        Args:
            sdate: Start date.
            edate: End date.
            logscale: If True, 'cum', 'mdd', 'msd' are in log-scale.
            annualize_factor: 'mean', 'std', 'sharpe', and 'turnover' are annualized by this factor, e.g.,
                `mean` is multiplied by `annualize_factor` and `std` by its square-root.
                If the data is monthly, the results can be annualized by setting `annualize_factor` = 12. Default to 1.
            consider_cost: If True, the results are calculated using net returns, otherwise, they are calculated
                using gross returns. If True (False), `value.ret` will be set to `value.netret` (`value.grossret`).
            percentage: If True, 'mean', 'std', 'cum', 'mdd', 'msd', and 'turnover' are multiplied by 100.
                Default to False.

        Returns:
            `performance`, `value`.
        """

        self.set_return_type(consider_cost)

        cumret = self.cum_returns(sdate, edate, logscale)
        if logscale:
            drawdown = self._drawdown(cumret, logscale=False)
            succdown = self._succdown(cumret, logscale=False)
        else:
            drawdown = self._drawdown(np.log(1 + cumret), logscale=False)
            succdown = self._succdown(np.log(1 + cumret), logscale=False)

        self.value['cumret'] = cumret
        self.value[['drawdown', 'drawdur', 'drawstart']] = drawdown
        self.value[['succdown', 'succdur', 'succstart']] = succdown

        portfolio = self.value[sdate:edate]
        r = portfolio.ret.values
        er = portfolio.exret.values
        # if self.fposition is not None:
        #     r -= self.fposition[sdate:edate].ret.values
        mean = er.mean()
        std = er.std()
        sharpe = mean / std if std != 0 else 0
        cum = cumret[-1]

        mdd = drawdown.loc[drawdown.value.idxmax()]
        msd = succdown.loc[succdown.value.idxmax()]
        dates = np.datetime_as_string(pd.to_datetime([mdd.start, mdd.name, msd.start, msd.name]), 'D')

        tover = portfolio.tover.mean()

        lposition = portfolio.lposition.mean()
        sposition = portfolio.sposition.mean()

        # Annualize
        mean *= annualize_factor
        std *= np.sqrt(annualize_factor)
        sharpe *= np.sqrt(annualize_factor)
        tover *= annualize_factor

        values = [mean, std, sharpe, cum, mdd.value, dates[0], dates[1], msd.value, dates[2], dates[3],
                  tover, lposition, sposition]
        index = ['mean', 'std', 'sharpe', 'cum', 'mdd', 'mdd start', 'mdd end', 'msd', 'msd start', 'msd end',
                 'turnover', 'lposition', 'sposition']
        self.performance = pd.DataFrame(values, index=index, columns=[self.name])
        if percentage:
            self.performance.loc[['mean', 'std', 'cum', 'mdd', 'msd', 'turnover']] *= 100
        return self.performance, self.value[sdate:edate]

    def eval_series(self, sdate=None, edate=None, logscale=True, annualize_factor=1, consider_cost=True, percentage=False):
        """Evaluate the portfolio repeatedly over the period.

        Evaluate the portfolio repeatedly for the period [`sdate`, `sdate`+1], [`sdate`, `sdate`+2], ...,
        [`sdate`, `edate`]. For the description of the arguments, see ``Portfolio.eval()``.

        Returns:
            Performance for each period: DataFrame with index values equal to the period end dates and columns
            equal to the indexes of `performance` attribute, i.e., a row with index t contains the performance up to t.
        """

        if sdate is None: sdate = self.value.index[0]
        if edate is None: edate = self.value.index[-1]
        perf_list = {}
        for date in self.value.index:
            if date <= sdate: continue
            if date > edate: break
            perf_list[date], _ = self.eval(sdate, date, logscale, annualize_factor, consider_cost, percentage)
        perf_list = pd.concat(perf_list, axis=1).transpose()
        return perf_list.droplevel(axis=0, level=1)  # drop portfolio name and keep dates only

    # def diff(self, bm):
    #     pfdiff = Portfolio()
    #     pfdiff.name = self.name + '-' + bm.name
    #     pfdiff.value = (self.value.ret - bm.value.ret).to_frame()
    #     return pfdiff


class Portfolios():
    """Class for a group of portfolios.

    This class can have several portfolios as its members and evaluate them together.
    This class facilitates portfolio comparison by evaluating them and saving the results in a single DataFrame.

    Args:
        portfolios: List or dict of `Portfolio` objects to add. If it is a dict, its keys are used as
            portfolio names. Portfolios can be added later using ``Portfolios.add()`` or ``Portfolios.set()``.

    Attributes:
        members (dict): Portfolio members. A `Portfolio` object is added to `members` with its name as the key.
            A member portfolio can be accessed by ``__getitem__()``:

            >>> pf1 = Portfolio('pf1')
            >>> portfolios = Portfolios()
            >>> portfolios.add(pf1)
            >>> pf1 = portfolios['pf1']  # This and the next line are equivalent.
            >>> pf1 = portfolios.members['pf1']

        value: Portfolios' values. This is a concatenated DataFrame of the `value` attributes of the members.
            It has the same structure as `Portfolio.value` except that its columns are two-level: the first level
            is the same as the columns of `Portfolio.value` and the second level is the member names.
        performance: Portfolios' performances. This is a concatenated DataFrame of the `performance` attributes of
            the members. It has the same structure as `Portfolio.performance` except that it has multiple columns,
            each corresponding to a member. The column names are the same as the portfolio names.
    """

    def __init__(self, portfolios=None):
        self.members = {}
        self.value = None
        self.performance = None
        if portfolios:
            self.set(portfolios)

    def __getitem__(self, alias):
        return self.members[alias]

    def add(self, portfolio, alias=None):
        """Add a portfolio.

        Args:
            portfolio (Portfolio): Portfolio to add.
            alias: Portfolio alias. If not None, this is used as the portfolio name.
        """
        key = alias if alias else portfolio.name
        self.members[key] = portfolio
        pfval = portfolio.value.copy()
        pfval.columns = pd.MultiIndex.from_product([pfval.columns, [key]])
        self.value = pd.concat([self.value, pfval], axis=1)
        if portfolio.performance is not None:
            self.performance = pd.concat([self.performance,
                                          portfolio.performance.rename(columns={portfolio.name: key})], axis=1)

    def set(self, portfolios):
        """Add portfolios.

        Args:
            portfolios: List or dict of `Portfolio` objects. If it is a dict, its keys are used as
                portfolio names.
        """

        if isinstance(portfolios, dict):
            self.members = portfolios
        elif isinstance(portfolios, list):
            self.members = {pf.name: pf for pf in portfolios}
        pfvals = []
        pfperf = []
        for portfolio in self.members.values():
            pfvals.append(portfolio.value)
            pfperf.append(portfolio.performance)
        self.value = pd.concat(pfvals, axis=1, keys=self.members.keys()).swaplevel(0, 1, axis=1)
        if not any(perf is None for perf in pfperf):
            self.performance = pd.concat(pfperf, axis=1)
            self.performance.columns = self.members.keys()

    def eval(self, sdate=None, edate=None, logscale=True, annualize_factor=1, consider_cost=True, percentage=False):
        """Evaluate the portfolios.

        For the arguments, see ``Portfolio.eval()``.

        Returns:
            `performance`, `value`.
        """

        for portfolio in self.members.values():
            portfolio.eval(sdate, edate, logscale, annualize_factor, consider_cost, percentage)
        self.set(self.members)

        return self.performance, self.value[sdate:edate]


# def compare_portfolios(portfolio, benchmark, sdate=None, edate=None, logscale=True, annualized=True,
#                        consider_cost=True):
#     diff = portfolio.diff(benchmark)
#     portfolios = Portfolios([portfolio, benchmark, diff])
#     return portfolios.eval(sdate, edate, logscale, annualized, consider_cost)


