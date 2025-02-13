import os
import re
import json
import threading
import time
import traceback
import importlib.util
from configparser import ConfigParser
from datetime import datetime
from openai import OpenAI
from gm.api import *

# ----------------------
# 配置初始化
# ----------------------
config = ConfigParser()
config.read('config.ini', encoding='utf-8')
llm_extra_body_config = config.get("LLM", "extra_body")
llm_extra_body = json.loads(llm_extra_body_config) if llm_extra_body_config else None


# ----------------------
# 全局状态管理
# ----------------------
class BacktestManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.results = {}
        self.current_iteration = 0
        self.best_metrics = {"value": -float('inf'), "iteration": -1}


manager = BacktestManager()

# ----------------------
# 大模型客户端初始化
# ----------------------
client = OpenAI(
    api_key=config.get("LLM", 'api_key'),
    base_url=config.get("LLM", 'base_url'),
)


# 增强策略生成提示模板
MULTI_STRATEGY_PROMPT = """
请开发一个基于掘金量化的多因子选股策略，要求：

一、策略框架
def init(context):
    # 初始化多因子选股模型
    
def algo(context):
    # 策略算法逻辑详情

def on_bar(context, bars):
    # 当使用subscribe订阅时，使用on_bar获取每个bar详情

二、选股要求
1. 使用至少三种不同类型因子组合：
   - 技术因子（如MACD、RSI、布林带）
   - 基本面因子（如PE、PB、ROE）
   - 量价因子（如成交量变化、资金流）

2. 每周进行一次选股，持仓5-10支股票

3. 下面代码为去掉双创板和ST相关股票之后的代码set,请在此范围进行选股
set(stk_get_sector_constituents(sector_code='001065')['symbol']) & set(stk_get_sector_constituents(sector_code='001048')['symbol'])

三、风控要求
1. 单支股票最大回撤达到{max_drawdown}%时立即平仓
2. 组合最大回撤达到12%时降低仓位
3. 使用波动率调整仓位

四、示例代码结构，不要照抄，仅供参考结构
# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import *
from pandas import DataFrame


'''
本策略每隔1个月定时触发,根据Fama-French三因子模型对每只股票进行回归，得到其alpha值。
假设Fama-French三因子模型可以完全解释市场，则alpha为负表明市场低估该股，因此应该买入。
策略思路：
计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
根据分类得到的组合分别计算其市值加权收益率、SMB和HML.
对各个股票进行回归(假设无风险收益率等于0)得到alpha值.
选取alpha值小于0并为最小的10只股票进入标的池
平掉不在标的池的股票并等权买入在标的池的股票
回测数据:SHSE.000300的成份股
回测时间:2017-07-01 08:00:00到2017-10-01 16:00:00
'''


def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    context.date = 20
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 账面市值比的大/中/小分类
    context.BM_BIG = 3.0
    context.BM_MID = 2.0
    context.BM_SMA = 1.0
    # 市值大/小分类
    context.MV_BIG = 2.0
    context.MV_SMA = 1.0

# 计算市值加权的收益率的函数,MV为市值的分类对应的组别,BM为账目市值比的分类对应的组别
def market_value_weighted(stocks, MV, BM):
    select = stocks[(stocks['NEGOTIABLEMV'] == MV) & (stocks.['BM'] == BM)] # 选出市值为MV，账目市值比为BM的所有股票数据
    market_value = select['mv'].values     # 对应组的全部市值数据
    mv_total = np.sum(market_value)        # 市值求和
    mv_weighted = [mv / mv_total for mv in market_value]   # 市值加权的权重
    stock_return = select['return'].values

    # 返回市值加权的收益率的和
    return_total = []
    for i in range(len(mv_weighted)):
        return_total.append(mv_weighted[i] * stock_return[i])
    return_total = np.sum(return_total)
    return return_total

def algo(context):
    # 获取上一个交易日的日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    # 获取沪深300成份股
    context.stock300 = get_history_constituents(index='SHSE.000300', start_date=last_day,
                                                end_date=last_day)[0]['constituents'].keys()
    # 获取当天有交易的股票
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended,
                           start_date=last_day, end_date=last_day,fields='PB,NEGOTIABLEMV', df=True)  # 获取P/B和市值数据

    # 计算账面市值比,为P/B的倒数
    fin['PB'] = (fin['PB'] ** -1)
    # 计算市值的50%的分位点,用于后面的分类
    size_gate = fin['NEGOTIABLEMV'].quantile(0.50)
    # 计算账面市值比的30%和70%分位点,用于后面的分类
    bm_gate = [fin['PB'].quantile(0.30), fin['PB'].quantile(0.70)]
    fin.index = fin.symbol
    # 设置存放股票收益率的list
    x_return = []

    # 对未停牌的股票进行处理
    for symbol in not_suspended:
        # 计算收益率，存放到x_return里面
        close = history_n(symbol=symbol, frequency='1d', count=context.date + 1, end_time=last_day, fields='close',
                          skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
        stock_return = close[-1] / close[0] - 1
        pb = fin['PB'][symbol]
        market_value = fin['NEGOTIABLEMV'][symbol]
        # 获取[股票代码， 股票收益率, 账面市值比的分类, 市值的分类, 流通市值]
        # 其中账面市值比的分类为：大（3）、中（2）、小（1）
        # 流通市值的分类：大（2）、小（1）
        if pb < bm_gate[0]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_SMA, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_SMA, context.MV_BIG, market_value]
        elif pb < bm_gate[1]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_MID, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_MID, context.MV_BIG, market_value]
        elif market_value < size_gate:
            label = [symbol, stock_return, context.BM_BIG, context.MV_SMA, market_value]
        else:
            label = [symbol, stock_return, context.BM_BIG, context.MV_BIG, market_value]
        if len(x_return) == 0:
            x_return = label
        else:
            x_return = np.vstack([x_return, label])

    # 将股票代码、 股票收益率、 账面市值比的分类、 市值的分类、 流通市值存为数据表
    stocks = DataFrame(data=x_return, columns=['symbol', 'return', 'BM', 'NEGOTIABLEMV', 'mv'])
    stocks.index = stocks.symbol
    columns = ['return', 'BM', 'NEGOTIABLEMV', 'mv']
    for column in columns:
        stocks[column] = stocks[column].astype(np.float64)

    # 计算SMB.HML和市场收益率（市值加权法）
    smb_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_MID) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_BIG)) / 3

    # 获取大市值组合的市值加权组合收益率
    smb_b = (market_value_weighted(stocks, context.MV_BIG, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_MID) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 3
    smb = smb_s - smb_b

    # 获取大账面市值比组合的市值加权组合收益率
    hml_b = (market_value_weighted(stocks, context.MV_SMA, 3) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 2

    # 获取小账面市值比组合的市值加权组合收益率
    hml_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_SMA)) / 2
    hml = hml_b - hml_s

    # 获取市场收益率
    close = history_n(symbol='SHSE.000300', frequency='1d', count=context.date + 1,
                      end_time=last_day, fields='close', skip_suspended=True,
                      fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
    market_return = close[-1] / close[0] - 1
    coff_pool = []

    # 对每只股票进行回归获取其alpha值
    for stock in stocks.index:
        x_value = np.array([[market_return], [smb], [hml], [1.0]])
        y_value = np.array([stocks['return'][stock]])
        # OLS估计系数
        coff = np.linalg.lstsq(x_value.T, y_value)[0][3]
        coff_pool.append(coff)

    # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
    stocks['alpha'] = coff_pool
    stocks = stocks[stocks.alpha < 0].sort_values(by='alpha').head(10)
    symbols_pool = stocks.index.tolist()
    positions = context.account().positions()

    # 平不在标的池的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in symbols_pool:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print('市价单平不在标的池的', symbol)

    # 获取股票的权重
    percent = context.ratio / len(symbols_pool)

    # 买在标的池中的股票
    for symbol in symbols_pool:
        order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(symbol, '以市价单调多仓到仓位', percent)

                
请只输出完整策略代码，不要输出除代码以外的内容，代码中必须包含上述所有要素。
""".format(max_drawdown=config.get("RISK_CONTROL", "stop_loss"))


# ----------------------
# 掘金量化回调函数
# ----------------------
def on_backtest_finished(context, indicators):
    """增强型回测完成回调"""
    with manager.lock:
        iteration = manager.current_iteration
        try:
            key_metrics = {
                "total_return": indicators["pnl_ratio"] * 100,
                "sharpe_ratio": indicators.get("sharpe_ratio", 0),
                "max_drawdown": indicators.get("max_drawdown", 0) * 100,
                "win_rate": indicators.get("win_rate", 0) * 100,
                "stock_count": indicators.get("position_count", 0),
                "turnover": indicators.get("turnover_ratio", 0),
                "timestamp": datetime.now().isoformat(),
                "valid": True
            }

            # 风控合规检查
            max_drawdown_limit = config.getint('RISK_CONTROL', 'stop_loss')
            if key_metrics["max_drawdown"] > max_drawdown_limit:
                key_metrics["valid"] = False
                print(f"迭代{iteration}回撤超标: {key_metrics['max_drawdown']}%")

            # 更新结果集
            manager.results[iteration] = key_metrics

            # 更新最佳记录（仅合规策略）
            if (key_metrics["valid"] and
                    key_metrics["sharpe_ratio"] > manager.best_metrics.get("value", -float('inf'))):
                manager.best_metrics = {
                    "value": key_metrics["sharpe_ratio"],
                    "iteration": iteration,
                    "metrics": key_metrics
                }
                print(f"新最佳策略（迭代{iteration}）夏普比率: {key_metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"回测结果处理异常: {str(e)}")
            traceback.print_exc()


# ----------------------
# 核心功能实现
# ----------------------
def run_backtest_in_thread(strategy_code: str, iteration: int):
    """安全执行回测的线程方法"""
    try:
        # 保存策略版本
        strategy_dir = "strategies"
        os.makedirs(strategy_dir, exist_ok=True)
        strategy_path = os.path.join(strategy_dir, f"strategy_v{iteration}.py")

        with open(strategy_path, "w", encoding="utf-8") as f:
            f.write(strategy_code)

        # 动态加载策略
        spec = importlib.util.spec_from_file_location(
            f"strategy_v{iteration}",
            strategy_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 配置回测参数
        backtest_config = {
            "mode": MODE_BACKTEST,
            "filename": strategy_path,
            "token": config.get("GM", "token"),
            "strategy_id": config.get("GM", "strategy_id"),
            "backtest_start_time": config.get("GM", "backtest_start_time"),
            "backtest_end_time": config.get("GM", "backtest_end_time"),
            "backtest_initial_cash": config.getfloat("GM", "backtest_initial_cash"),
            "backtest_commission_ratio": config.getfloat("GM", "backtest_commission_ratio"),
            "backtest_slippage_ratio": config.getfloat("GM", "backtest_slippage_ratio"),
        }

        # 更新当前迭代号
        with manager.lock:
            manager.current_iteration = iteration

        # 执行回测
        run(**backtest_config)

        # 保存完整日志
        save_iteration_log(iteration, strategy_code)
        return None

    except (TypeError, NameError, Exception) as typee:
        error_info = {
            "iteration": iteration,
            "error_type": type(typee).__name__,
            "error_msg": str(typee),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        error_dir = "errors"
        os.makedirs(error_dir, exist_ok=True)
        with open(os.path.join(error_dir, f"error_v{iteration}.json"), "w") as f:
            json.dump(error_info, f)
        return error_info


# ----------------------
# 验证与优化模块
# ----------------------
def optimize_strategy(current_code: str, backtest_results: dict) -> str:
    """智能策略优化引擎"""
    try:
        prompt = f"""根据以下策略和回测结果进行优化：

【当前策略】
{current_code}

【回测指标】
{json.dumps(backtest_results, indent=2)}

【优化目标】
1. 提升夏普比率（当前：{backtest_results.get('sharpe_ratio', 0):.2f}）
2. 控制最大回撤 <= {config.getint('RISK_CONTROL', 'stop_loss')}%
3. 保持多因子选股框架

【优化方向】
- 改进因子合成方法（动态权重/非线性组合）
- 增强风险控制模块（实时监控+自动止损）
- 优化换仓逻辑（减少交易成本）
- 添加仓位管理策略（波动率调整）

请只输出完整策略代码，不要输出除代码以外的内容，代码中必须包含上述所有要素。
"""

        response = client.chat.completions.create(
            model=config.get("LLM", "mode"),
            messages=[
                {"role": "system", "content": "你是有十年经验的量化投资专家，请给出专业级优化方案"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            extra_body=llm_extra_body,
        )

        return re.sub(r"```python|```", "", response.choices[0].message.content)
    except Exception as e:
        print(f"优化过程异常: {str(e)}")
        return current_code  # 保底返回原代码


# ----------------------
# 辅助函数
# ----------------------
def save_iteration_log(iteration: int, code: str):
    """保存完整迭代记录"""
    log_entry = {
        "iteration": iteration,
        "code": code,
        "metrics": manager.results.get(iteration, {}),
        "timestamp": datetime.now().isoformat()
    }

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    try:
        with open(os.path.join(log_dir, f"iteration_{iteration}.json"), "w") as f:
            json.dump(log_entry, f)

        # 保存最佳版本
        if iteration == manager.best_metrics.get("iteration", -1):
            with open("best_strategy.py", "w") as f:
                f.write(code)
            with open("best_metrics.json", "w") as f:
                json.dump(manager.best_metrics, f)
    except Exception as e:
        print(f"日志保存失败: {str(e)}")


# ----------------------
# 主程序流程
# ----------------------
def optimization_loop():
    """主优化流程"""
    # 初始化环境
    set_token(config.get("GM", "token"))
    for folder in ["strategies", "logs", "errors"]:
        os.makedirs(folder, exist_ok=True)

    current_strategy = generate_strategy_code()
    max_iterations = config.getint("EVO", "max_iterations")

    for iteration in range(max_iterations):
        print(f"\n=== 迭代 {iteration + 1}/{max_iterations} ===")
        print(f"当前最佳夏普比率: {manager.best_metrics.get('value', 0):.2f}")

        # 执行回测
        backtest_result = run_backtest_in_thread(current_strategy, iteration)

        if backtest_result is not None:
            current_strategy = handle_failed_iteration(current_strategy, backtest_result, iteration)
            continue

        # 结果验证
        results = manager.results.get(iteration, {})
        if not results.get("valid", False):
            print(f"迭代{iteration}因回撤超标被拒绝")
            current_strategy = enforce_drawdown_control(current_strategy, results)
            continue

        # 生成优化版本
        current_strategy = optimize_strategy(current_strategy, results)

    print("\n=== 优化流程完成 ===")
    print(f"最终最佳夏普比率: {manager.best_metrics.get('value', 0):.2f}")


def generate_strategy_code() -> str:
    for _ in range(3):  # 最多重试3次
        try:
            response = client.chat.completions.create(
                model=config.get("LLM", "mode"),
                messages=[{
                    "role": "user",
                    "content": MULTI_STRATEGY_PROMPT
                }],
                temperature=0.7,
                extra_body=llm_extra_body,
            )
            return re.sub(r"```python|```", "", response.choices[0].message.content)
        except Exception as e:
            print(f"生成策略异常: {str(e)}")
            time.sleep(1)

    raise Exception("策略生成失败，请检查提示词或模型配置")


def enforce_drawdown_control(code: str, results: dict) -> str:
    """强制风控优化"""
    prompt = f"""【强制风控优化要求】
当前策略最大回撤：{results.get('max_drawdown', 0):.2f}%
允许最大回撤：{config.getint('RISK_CONTROL', 'stop_loss')}%

请改进以下策略代码的风险控制模块：
{code}

具体改进要求：
1. 添加个股实时回撤监控（精确到分钟级）
2. 实现动态止损机制（根据波动率调整）
3. 优化仓位管理算法（不超过5层仓位控制）
4. 保留多因子选股框架（技术+基本面+量价）

输出要求：
- 仅修改风控相关代码
- 保持其他逻辑不变
- 添加必要的日志输出

请只输出完整策略代码，不要输出除代码以外的内容，代码中必须包含上述所有要素。
"""

    try:
        response = client.chat.completions.create(
            model=config.get("LLM", "mode"),
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.4,
            extra_body=llm_extra_body,
        )
        return re.sub(r"```python|```", "", response.choices[0].message.content)
    except Exception as e:
        print(f"强制风控优化失败: {str(e)}")
        return code


def handle_failed_iteration(code: str, backtest_result: dict, iteration: int) -> str:
    """智能处理失败迭代"""
    error_info = backtest_result
    print(f"处理第{iteration}次迭代失败: {error_info.get('error_type')}")
    return handle_generic_error(code, error_info)


def handle_generic_error(code: str, error_info: dict) -> str:
    """通用错误处理：调用大模型自动修复策略代码"""
    # 构建详细错误报告
    error_report = f"""
    【错误类型】{error_info.get('error_type', 'Unknown')}
    【错误信息】{error_info.get('error_msg', '无详细信息')}
    【关键堆栈】{extract_key_traceback(error_info.get('traceback', ''))}
    """

    # 构造修复提示词
    prompt = f"""请修复以下Python量化策略代码中的错误：

=== 当前策略代码 ===
{code}

=== 错误诊断报告 ===
{error_report}

=== 修复要求 ===
1. 保持多因子选股框架（技术+基本面+量价）
2. 确保最大回撤不超过{config.getint('RISK_CONTROL', 'stop_loss')}%
3. 保留原有风险控制模块
4. 仅修改必要部分，保持代码简洁
5. 确保语法完全正确

请只输出完整策略代码，不要输出除代码以外的内容，代码中必须包含上述所有要素。
"""

    try:
        # 调用大模型
        response = client.chat.completions.create(
            model=config.get("LLM", "mode"),
            messages=[
                {
                    "role": "system",
                    "content": "你是有十年经验的量化开发专家，请专业地修复代码错误"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # 低随机性确保稳定
            extra_body = llm_extra_body,
        )

        # 提取并后处理代码
        return re.sub(r"```python|```", "", response.choices[0].message.content)

    except Exception as e:
        print(f"通用错误修复失败: {str(e)}")
        return code  # 返回原代码保底


def extract_key_traceback(traceback_str: str) -> str:
    """提取关键堆栈信息"""
    lines = traceback_str.split('\n')
    # 保留最后3个堆栈帧
    return '\n'.join([line for line in lines if "File" in line][-3:] + [lines[-1]])

if __name__ == "__main__":
    optimization_loop()