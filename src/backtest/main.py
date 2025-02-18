import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os
import json
import logging
import math

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from ta.momentum import RSIIndicator
from dotenv import load_dotenv

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
proxy_http = os.getenv("PROXY_HTTP")
proxy_https = os.getenv("PROXY_HTTPS")

# ----------------------------
# ヘルパー関数定義
# ----------------------------
def timeframe_to_seconds(timeframe: str) -> int:
    """例: '5m' -> 300, '1h' -> 3600, '1d' -> 86400"""
    if timeframe.endswith("m"):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith("h"):
        return int(timeframe[:-1]) * 3600
    elif timeframe.endswith("d"):
        return int(timeframe[:-1]) * 86400
    else:
        return 60
        
# ----------------------------
# バックテスト用クラス定義
# ----------------------------

@dataclass
class BacktestConfig:
    symbol: str = "XRP/USDT"
    timeframe: str = "5m"
    initial_balance: float = 600.0
    leverage: int = 5
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    tp_percent: float = 0.3
    sl_percent: Optional[float] = None  
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class Position:
    direction: str
    entry_price: float
    size: float
    entry_time: datetime
    tp_price: float
    sl_price: Optional[float] = None


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    fee: float
    net_pnl: float
    reason: str


class MarketData:
    """
    ccxt.bybitデータ取得と管理を行うクラス(プロキシ通し)
    progress_callback: 取得フェーズの進捗(0~1)を受け取る
    """
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'proxies': {
                'http': proxy_http,
                'https': proxy_https,
            },
        })
        
    def fetch_data(self, progress_callback: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
        logger.info(f"データ取得開始: {self.config.start_date} から {self.config.end_date}")
        
        start_timestamp = int(self.config.start_date.timestamp() * 1000)
        end_timestamp = int(self.config.end_date.timestamp() * 1000)
        
        # 取得予定件数の目安(適当)
        tf_seconds = timeframe_to_seconds(self.config.timeframe)
        expected_count = math.ceil((self.config.end_date.timestamp() - self.config.start_date.timestamp()) / tf_seconds)
        logger.info(f"想定取得件数: {expected_count}")
        
        all_ohlcv = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.config.symbol,
                    self.config.timeframe,
                    since=current_timestamp,
                    limit=1000
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + 1
                logger.info(f"取得済みデータ数: {len(all_ohlcv)}")
                
                if progress_callback:
                    frac = min(len(all_ohlcv) / expected_count, 1.0)
                    progress_callback(frac)
                    
            except Exception as e:
                logger.error(f"データ取得エラー: {str(e)}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        rsi_indicator = RSIIndicator(close=df['close'], window=self.config.rsi_period)
        df['rsi'] = rsi_indicator.rsi()
        
        return df


class ScalpingStrategy:
    """スキャルピング戦略実装クラス"""
    def __init__(self, config: BacktestConfig):
        self.config = config

    def should_open_long(self, row: pd.Series) -> bool:
        return not np.isnan(row['rsi']) and row['rsi'] <= self.config.rsi_oversold

    def should_open_short(self, row: pd.Series) -> bool:
        return not np.isnan(row['rsi']) and row['rsi'] >= self.config.rsi_overbought

    def should_close_position(self, position: Position, row: pd.Series) -> Tuple[bool, str]:
        if position.direction == 'long':
            if position.sl_price is not None and row['low'] <= position.sl_price:
                return True, 'SL'
            elif row['high'] >= position.tp_price:
                return True, 'TP'
        elif position.direction == 'short':
            if position.sl_price is not None and row['high'] >= position.sl_price:
                return True, 'SL'
            elif row['low'] <= position.tp_price:
                return True, 'TP'
        return False, ''


class PortfolioManager:
    """ポートフォリオ管理クラス"""
    def __init__(self, config: BacktestConfig):
        self.config = config
        self._current_balance = config.initial_balance
        self._position: Optional[Position] = None
        self._trades: List[Trade] = []
        self._balance_history: List[float] = [config.initial_balance]

    @property
    def current_balance(self) -> float:
        return self._current_balance

    @current_balance.setter
    def current_balance(self, value: float):
        self._current_balance = value

    @property
    def position(self) -> Optional[Position]:
        return self._position

    @position.setter
    def position(self, pos: Optional[Position]):
        self._position = pos

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def balance_history(self) -> List[float]:
        return self._balance_history

    def calculate_position_size(self, price: float) -> float:
        return (self._current_balance * self.config.leverage) / price

    def open_position(self, row: pd.Series, direction: str):
        if self._position is not None:
            return

        price = row['close']
        position_size = self.calculate_position_size(price)
        fee = position_size * price * self.config.taker_fee
        
        tp_price = price * (1 + self.config.tp_percent/100) if direction == 'long' \
                   else price * (1 - self.config.tp_percent/100)
        
        if self.config.sl_percent is not None:
            sl_price = price * (1 - self.config.sl_percent/100) if direction == 'long' \
                       else price * (1 + self.config.sl_percent/100)
        else:
            sl_price = None
        
        self._position = Position(
            direction=direction,
            entry_price=price,
            size=position_size,
            entry_time=row.name,
            tp_price=tp_price,
            sl_price=sl_price
        )
        
        self._current_balance -= fee
        logger.info(f"{direction}ポジション発火: 価格 = {price}, サイズ = {position_size}, 手数料 = {fee}")

    def close_position(self, row: pd.Series, reason: str):
        if self._position is None:
            return

        exit_price = row['close']
        if self._position.direction == 'long':
            pnl = (exit_price - self._position.entry_price) * self._position.size
        else:
            pnl = (self._position.entry_price - exit_price) * self._position.size
        
        fee = self._position.size * exit_price * self.config.taker_fee
        
        trade = Trade(
            entry_time=self._position.entry_time,
            exit_time=row.name,
            direction=self._position.direction,
            entry_price=self._position.entry_price,
            exit_price=exit_price,
            size=self._position.size,
            pnl=pnl,
            fee=fee,
            net_pnl=pnl - fee,
            reason=reason
        )
        self._trades.append(trade)
        
        self._current_balance += pnl - fee
        self._balance_history.append(self._current_balance)
        
        logger.info(f"ポジションクローズ: {reason}, PnL = {pnl}, 手数料 = {fee}, 純利益 = {pnl - fee}")
        self._position = None


class BacktestAnalyzer:
    """
    バックテスト結果の分析・可視化および結果保存クラス
    """
    def __init__(self, config: BacktestConfig, portfolio: PortfolioManager, price_data: pd.DataFrame):
        self.config = config
        self.portfolio = portfolio
        self.price_data = price_data
        self.results_dir = Path('export/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plot_style()

    def _setup_plot_style(self):
        plt.style.use('ggplot')
        font_path = Path('assets/fonts/Noto_Sans_JP/static/NotoSansJP-Medium.ttf')
        if font_path.exists():
            from matplotlib.font_manager import FontProperties
            self.custom_font = FontProperties(fname=str(font_path))
            plt.rcParams['font.family'] = self.custom_font.get_name()
        else:
            if os.uname().sysname == 'Darwin':
                plt.rcParams['font.family'] = 'AppleGothic'
            else:
                plt.rcParams['font.family'] = 'MS Gothic'
        plt.rcParams.update({
            'axes.grid': True,
            'font.size': 12,
            'axes.unicode_minus': False
        })

    def plot_all_results(self):
        logger.info("全グラフを1枚の画像にまとめて作成開始")
        fig = plt.figure(figsize=(20, 12), dpi=300)
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], wspace=0.3, hspace=0.4)
        
        # 上段：損益分布、方向別損益、累積損益
        ax_profit = fig.add_subplot(gs[0, 0])
        profits = [trade.net_pnl for trade in self.portfolio.trades]
        ax_profit.hist(profits, bins=30, alpha=0.75, color='coral')
        ax_profit.set_title('損益分布', fontproperties=self.custom_font)
        ax_profit.set_xlabel('損益 (USDT)', fontproperties=self.custom_font)
        ax_profit.set_ylabel('回数', fontproperties=self.custom_font)
        ax_profit.tick_params(labelsize=10)
        
        ax_direction = fig.add_subplot(gs[0, 1])
        long_pnl = [t.net_pnl for t in self.portfolio.trades if t.direction == 'long']
        short_pnl = [t.net_pnl for t in self.portfolio.trades if t.direction == 'short']
        bp = ax_direction.boxplot([long_pnl, short_pnl],
                                  labels=['ロング', 'ショート'],
                                  patch_artist=True)
        for element, color in {'boxes': 'lightblue', 'whiskers': 'black', 'medians': 'red', 'caps': 'black'}.items():
            plt.setp(bp[element], color=color)
        plt.setp(bp['boxes'], facecolor='lightblue', alpha=0.6)
        ax_direction.set_title('方向別の損益分布', fontproperties=self.custom_font)
        ax_direction.set_ylabel('損益 (USDT)', fontproperties=self.custom_font)
        ax_direction.tick_params(labelsize=10)
        
        ax_cumprofit = fig.add_subplot(gs[0, 2])
        cumulative_pnl = np.cumsum([t.net_pnl for t in self.portfolio.trades])
        ax_cumprofit.plot(cumulative_pnl, color='red', linewidth=2)
        ax_cumprofit.set_title('累積損益', fontproperties=self.custom_font)
        ax_cumprofit.set_xlabel('取引回数', fontproperties=self.custom_font)
        ax_cumprofit.set_ylabel('累積損益 (USDT)', fontproperties=self.custom_font)
        ax_cumprofit.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax_cumprofit.tick_params(labelsize=10)
        ax_cumprofit.fill_between(range(len(cumulative_pnl)),
                                  cumulative_pnl,
                                  0,
                                  where=(cumulative_pnl >= 0),
                                  color='lightgreen', alpha=0.3)
        ax_cumprofit.fill_between(range(len(cumulative_pnl)),
                                  cumulative_pnl,
                                  0,
                                  where=(cumulative_pnl < 0),
                                  color='lightcoral', alpha=0.3)
        
        # 下段：価格チャートと残高推移
        gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.3)
        ax_price = fig.add_subplot(gs_bottom[0])
        ax_price.plot(self.price_data.index, self.price_data['close'], label="終値", color='blue')
        ax_price.set_title('XRP/USDT 価格チャート', fontproperties=self.custom_font)
        ax_price.set_xlabel('日時', fontproperties=self.custom_font)
        ax_price.set_ylabel('価格 (USDT)', fontproperties=self.custom_font)
        ax_price.grid(True)
        locator = mdates.DayLocator(interval=3)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax_price.xaxis.set_major_locator(locator)
        ax_price.xaxis.set_major_formatter(formatter)
        plt.setp(ax_price.get_xticklabels(), rotation=45, ha="right", fontsize=10)
        ax_price.legend(prop=self.custom_font)
        
        ax_balance = fig.add_subplot(gs_bottom[1])
        ax_balance.plot(range(len(self.portfolio.balance_history)),
                        self.portfolio.balance_history,
                        marker='o', linestyle='-', color='green')
        ax_balance.set_title('残高推移', fontproperties=self.custom_font)
        ax_balance.set_xlabel('取引回数', fontproperties=self.custom_font)
        ax_balance.set_ylabel('残高 (USDT)', fontproperties=self.custom_font)
        ax_balance.grid(True)
        
        fig.suptitle(f'バックテスト結果：{self.config.symbol} ({self.config.timeframe})', 
                     fontsize=20, fontproperties=self.custom_font, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = self.results_dir / 'all_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"全グラフ画像保存完了: {output_path}")

    def _save_trade_history(self):
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'pnl': t.pnl,
                'fee': t.fee,
                'net_pnl': t.net_pnl,
                'reason': t.reason
            }
            for t in self.portfolio.trades
        ])
        trades_df.to_csv(self.results_dir / 'trades.csv', index=False)

    def _save_statistics(self):
        stats = self.calculate_statistics()
        with open(self.results_dir / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    def calculate_statistics(self) -> Dict:
        if not self.portfolio.trades:
            return {}
        
        profits = [trade.net_pnl for trade in self.portfolio.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        total_return = ((self.portfolio.current_balance - self.config.initial_balance) / 
                        self.config.initial_balance * 100)
        win_rate = (len(winning_trades) / len(profits) * 100) if profits else 0
        
        return {
            '初期残高': round(self.config.initial_balance, 2),
            '最終残高': round(self.portfolio.current_balance, 2),
            '総リターン (%)': round(total_return, 2),
            '総取引回数': len(self.portfolio.trades),
            'ロング取引数': len([t for t in self.portfolio.trades if t.direction == 'long']),
            'ショート取引数': len([t for t in self.portfolio.trades if t.direction == 'short']),
            '勝率 (%)': round(win_rate, 2),
            '平均利益': round(np.mean(winning_trades), 2) if winning_trades else 0,
            '平均損失': round(np.mean(losing_trades), 2) if losing_trades else 0,
            '最大利益': round(max(profits), 2) if profits else 0,
            '最大損失': round(min(profits), 2) if profits else 0,
            '総手数料': round(sum(trade.fee for trade in self.portfolio.trades), 2),
            'シャープレシオ': round(self._calculate_sharpe_ratio(profits), 2),
            '最大ドローダウン (%)': round(self._calculate_max_drawdown(), 2),
            'プロフィットファクター': self._calculate_profit_factor(winning_trades, losing_trades)
        }

    def _calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        if not profits:
            return 0.0
        returns = np.array(profits) / self.config.initial_balance
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        periods_per_year = 365 * 24 * 12
        sharpe = np.sqrt(periods_per_year) * (avg_return - risk_free_rate/periods_per_year) / std_return
        return sharpe

    def _calculate_max_drawdown(self) -> float:
        if not self.portfolio.balance_history:
            return 0.0
        balance_array = np.array(self.portfolio.balance_history)
        peak = np.maximum.accumulate(balance_array)
        drawdown = (peak - balance_array) / peak * 100
        return np.max(drawdown)

    def _calculate_profit_factor(self, winning_trades: List[float], losing_trades: List[float]) -> float:
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0
        return round(total_profit / total_loss, 2)

    def save_results(self):
        logger.info("結果保存開始")
        self._save_trade_history()        
        self._save_statistics()
        logger.info("結果保存完了")


class Backtest:
    """
    バックテスト実行クラス
    フェーズの内訳：
      - データ取得: 70%
      - バックテスト処理: 20%
      - 結果保存: 10%
    """
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data = MarketData(config)
        self.strategy = ScalpingStrategy(config)
        self.portfolio = PortfolioManager(config)
        self.data = None
        self.analyzer = None

    def run(self, progress_callback: Optional[Callable[[float], None]] = None,
                  status_callback: Optional[Callable[[str], None]] = None) -> bool:
        # 各フェーズの重み
        data_fetch_weight = 0.7
        backtest_weight   = 0.2
        # --- データ取得フェーズ ---
        if status_callback:
            status_callback("データを取得しています...")
        # MarketData内の進捗は0～1なので、ここで重み付けして更新
        df = self.market_data.fetch_data(
            progress_callback=lambda frac: progress_callback(data_fetch_weight * frac) if progress_callback else None
        )
        if df.empty:
            logger.error("データを取得できませんでした。")
            return False
        self.data = df

        # --- バックテスト処理フェーズ ---
        if status_callback:
            status_callback("バックテストを開始中...")
        total_steps = len(df)
        for i, (_, row) in enumerate(df.iterrows()):
            if self.portfolio.position is not None:
                should_close, reason = self.strategy.should_close_position(self.portfolio.position, row)
                if should_close:
                    self.portfolio.close_position(row, reason)
                    if progress_callback:
                        overall = data_fetch_weight + ((i + 1) / total_steps * backtest_weight)
                        progress_callback(overall)
                    continue
            if self.strategy.should_open_long(row):
                self.portfolio.open_position(row, 'long')
            elif self.strategy.should_open_short(row):
                self.portfolio.open_position(row, 'short')
            if progress_callback:
                overall = data_fetch_weight + ((i + 1) / total_steps * backtest_weight)
                progress_callback(overall)
        logger.info("バックテスト処理完了")
        return True

    def save_results(self):
        self.analyzer = BacktestAnalyzer(self.config, self.portfolio, self.data)
        self.analyzer.plot_all_results()
        self.analyzer.save_results()
        stats = self.analyzer.calculate_statistics()
        print("\nバックテスト結果一覧:")
        print(f"期間: {self.config.start_date.strftime('%Y-%m-%d')} から {self.config.end_date.strftime('%Y-%m-%d')}")
        print(f"初期残高: {stats.get('初期残高', 'N/A')} USDT")
        print(f"最終残高: {stats.get('最終残高', 'N/A')} USDT")
        print(f"総リターン: {stats.get('総リターン (%)', 'N/A')}%")
        print(f"総取引回数: {stats.get('総取引回数', 'N/A')}")
        print(f"勝率: {stats.get('勝率 (%)', 'N/A')}%")
        print(f"最大DD: {stats.get('最大ドローダウン (%)', 'N/A')}%")
        print(f"\n詳細結果は 'export/results/' に保存")


def run_backtest(
    symbol: str = "XRP/USDT",
    timeframe: str = "5m",
    initial_balance: float = 600.0,
    leverage: int = 5,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    tp_percent: float = 0.3,
    sl_percent: Optional[float] = None,
    days: int = 30,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> None:
    # 各フェーズの重み
    data_fetch_weight = 0.7
    backtest_weight   = 0.2
    result_weight     = 0.1

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    config = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=initial_balance,
        leverage=leverage,
        rsi_period=rsi_period,
        rsi_overbought=rsi_overbought,
        rsi_oversold=rsi_oversold,
        tp_percent=tp_percent,
        sl_percent=sl_percent,
        start_date=start_date,
        end_date=end_date
    )
    
    backtest = Backtest(config)
    if backtest.run(progress_callback=progress_callback, status_callback=status_callback):
        if progress_callback:
            progress_callback(data_fetch_weight + backtest_weight)
        if status_callback:
            status_callback("結果保存中...")
        backtest.save_results()
        if progress_callback:
            progress_callback(1.0)
        if status_callback:
            status_callback("バックテスト完了")
    else:
        if status_callback:
            status_callback("バックテスト実行中にエラーが発生しました。")


# ----------------------------
# Streamlit フロントエンド
# ----------------------------

st.set_page_config(page_title="バックテスト実行アプリ", layout="wide")
st.title("バックテスト実行アプリ")
st.markdown("---")
st.write("以下のパラメータを設定して、バックテストを実行してください。")

# ステータス表示用のコンテナ
status_placeholder = st.empty()
def update_status(message: str):
    status_placeholder.info(message)

# 進捗バー用
progress_placeholder = st.empty()
progress_bar = st.progress(0)
def update_progress(progress: float):
    percent = min(int(progress * 100), 100)
    progress_bar.progress(percent)
    progress_placeholder.text(f"進捗: {percent}%")

with st.sidebar:
    st.header("パラメータ設定")
    symbol = st.text_input("シンボル", "XRP/USDT")
    timeframe = st.selectbox("時間足", ["1m", "5m", "15m", "1h", "4h", "1d"], index=1)
    initial_balance = st.number_input("初期残高 (USDT)", min_value=1.0, value=600.0, step=1.0)
    leverage = st.number_input("レバレッジ", min_value=1, value=5, step=1)
    rsi_period = st.number_input("RSI期間", min_value=1, value=14, step=1)
    rsi_overbought = st.number_input("RSI上限", value=85.0, step=1.0)
    rsi_oversold = st.number_input("RSI下限", value=15.0, step=1.0)
    tp_percent = st.number_input("TP (%)", value=0.3, step=0.1)
    sl_percent = st.number_input("SL (%)", value=3.0, step=0.1)
    days = st.number_input("バックテスト日数", min_value=1, value=60, step=1)

if st.sidebar.button("バックテスト実行"):
    update_status("バックテストを開始します。しばらくお待ちください...")
    try:
        run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            initial_balance=initial_balance,
            leverage=leverage,
            rsi_period=rsi_period,
            rsi_overbought=rsi_overbought,
            rsi_oversold=rsi_oversold,
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            days=days,
            progress_callback=update_progress,
            status_callback=update_status
        )
    except Exception as e:
        update_status(f"バックテスト実行中にエラーが発生しました: {e}")

    # 結果表示（タブ形式）
    results_dir = Path('export/results')
    statistics_file = results_dir / 'statistics.json'
    trades_file = results_dir / 'trades.csv'
    results_image = results_dir / 'all_results.png'
    
    if statistics_file.exists():
        with open(statistics_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    tab1, tab2, tab3 = st.tabs(["統計情報", "トレード履歴", "バックテスト結果グラフ"])
    
    with tab1:
        if stats:
            st.subheader("統計情報")
            st.json(stats)
        else:
            st.error("統計情報が見つかりません。")
    
    with tab2:
        if trades_file.exists():
            trades_df = pd.read_csv(trades_file)
            st.subheader("トレード履歴")
            st.dataframe(trades_df)
        else:
            st.error("トレード履歴が見つかりません。")
    
    with tab3:
        if results_image.exists():
            st.subheader("バックテスト結果グラフ")
            st.image(str(results_image))
        else:
            st.error("バックテスト結果グラフが見つかりません。")