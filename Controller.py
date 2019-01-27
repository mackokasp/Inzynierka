import Finance as ff
import Graph as gg
import Optimizer as opt


def optimize(ratio='omega', method='SLSQP', minW=0.01, maxW=0.8, weights=None, K=3):
    return opt.optimize(ratio, method, minW, maxW, weights, K=3)


def year_returns(tickers, yearfrom, yearto):
    return ff.year_returns(tickers, yearfrom, yearto)


def month_returns(tickers, yearfrom, yearto):
    return ff.month_returns(tickers, yearfrom, yearto)


def draw_portfolios_omega(ddf, ticker, optimal=None):
    gg.draw_portfolios_omega(ddf, ticker, optimal)


def draw_table(df, optimal):
    gg.draw_table(df, optimal)


def set_target(target):
    ff.set_target(target)


def set_avrage(ret):
    ff.set_average(ret)


def set_returns(ret):
    opt.set_returns(ret)


def eval_results(tickers, yearfrom, yearto):
    gg.eval_results4(tickers, yearfrom, yearto)


def eval_portfolio(tickers, yearfrom, yearto, yearsafter, sol=[]):
    gg.eval_results5(tickers, yearfrom, yearto, yearsafter, sol)
