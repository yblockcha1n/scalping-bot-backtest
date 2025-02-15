import pandas as pd
import numpy as np
import ccxt
import matplotlib.pyplot as plt
import talib
import os
import json
import logging

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """市場データの取得と管理を行うクラス"""
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })

    def fetch_data(self) -> pd.DataFrame:
        """指定された期間のOHLCVデータを取得"""
        logger.info(f"データ取得開始: {self.config.start_date} から {self.config.end_date}")
        
        start_timestamp = int(self.config.start_date.timestamp() * 1000)
        end_timestamp = int(self.config.end_date.timestamp() * 1000)
        
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
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.config.rsi_period)
        
        return df

class ScalpingStrategy:
    """スキャトレード戦略の実装クラス"""
    def __init__(self, config: BacktestConfig):
        self.config = config

    def should_open_long(self, row: pd.Series) -> bool:
        """ロングポジションを開くべきかどうかの判断"""
        return not np.isnan(row['rsi']) and row['rsi'] <= self.config.rsi_oversold

    def should_open_short(self, row: pd.Series) -> bool:
        """ショートポジションを開くべきかどうかの判断"""
        return not np.isnan(row['rsi']) and row['rsi'] >= self.config.rsi_overbought

    def should_close_position(self, position: Position, row: pd.Series) -> Tuple[bool, str]:
        """ポジションを閉じるべきかどうかの判断"""
        if position.direction == 'long' and row['high'] >= position.tp_price:
            return True, 'tp'
        elif position.direction == 'short' and row['low'] <= position.tp_price:
            return True, 'tp'
        return False, ''

class PortfolioManager:
    """ポートフォリオ管理の実装クラス"""
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_balance = config.initial_balance
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.balance_history: List[float] = [config.initial_balance]
        self.equity_history: List[float] = [config.initial_balance]

    def calculate_position_size(self, price: float) -> float:
        """SIZE計算"""
        return (self.current_balance * self.config.leverage) / price

    def open_position(self, row: pd.Series, direction: str):
        """ポジション発火"""
        if self.position is not None:
            return

        price = row['close']
        position_size = self.calculate_position_size(price)
        fee = position_size * price * self.config.taker_fee
        
        tp_price = price * (1 + self.config.tp_percent/100) if direction == 'long' \
                  else price * (1 - self.config.tp_percent/100)
        
        self.position = Position(
            direction=direction,
            entry_price=price,
            size=position_size,
            entry_time=row.name,
            tp_price=tp_price
        )
        
        self.current_balance -= fee
        logger.info(f"{direction}ポジション発火: 価格 = {price}, サイズ = {position_size}, 手数料 = {fee}")

    def close_position(self, row: pd.Series, reason: str):
        """ポジションクローズ"""
        if self.position is None:
            return

        exit_price = row['close']
        pnl = (exit_price - self.position.entry_price) * self.position.size if self.position.direction == 'long' \
              else (self.position.entry_price - exit_price) * self.position.size
        
        fee = self.position.size * exit_price * self.config.taker_fee
        
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=row.name,
            direction=self.position.direction,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            size=self.position.size,
            pnl=pnl,
            fee=fee,
            net_pnl=pnl - fee,
            reason=reason
        )
        self.trades.append(trade)
        
        self.current_balance += pnl - fee
        self.balance_history.append(self.current_balance)
        
        logger.info(f"ポジションクローズ: {reason}, PnL = {pnl}, 手数料 = {fee}, 純利益 = {pnl - fee}")
        
        self.position = None

class BacktestAnalyzer:
    """バックテスト結果の分析と可視化を行うクラス"""
    def __init__(self, config: BacktestConfig, portfolio: PortfolioManager):
        self.config = config
        self.portfolio = portfolio
        self.results_dir = Path('export/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_plot_style()

    def _setup_plot_style(self):
        """プロットスタイル設定"""
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
            'figure.figsize': [12, 8],
            'figure.dpi': 100,
            'font.size': 12,
            'axes.unicode_minus': False
        })

    def plot_results(self):
        """バックテスト結果をプロット"""
        logger.info("グラフ作成開始")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)

        title_font = self.custom_font.copy()
        title_font.set_size(16)
        title_font.set_weight('bold')

        label_font = self.custom_font.copy()
        label_font.set_size(12)
        
        fig.suptitle(f'バックテスト結果: {self.config.symbol} ({self.config.timeframe})', 
                    y=0.95, fontproperties=title_font)
        
        self._plot_equity_curve(ax1, label_font)
        self._plot_profit_distribution(ax2, label_font)
        self._plot_direction_analysis(ax3, label_font)
        self._plot_cumulative_profit(ax4, label_font)
        
        plt.savefig(self.results_dir / 'backtest_results.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("グラフ作成完了")

    def _plot_equity_curve(self, ax, font_prop):
        """エクイティカーブのプロット"""
        ax.plot(np.arange(len(self.portfolio.equity_history)), 
                self.portfolio.equity_history, 
                label='エクイティ', 
                color='red')
        ax.plot(np.arange(len(self.portfolio.balance_history)), 
                self.portfolio.balance_history, 
                label='残高', 
                linestyle='--', 
                color='blue')

        ax.set_title('エクイティカーブ', fontproperties=font_prop)
        ax.set_xlabel('取引回数', fontproperties=font_prop)
        ax.set_ylabel('USDT', fontproperties=font_prop)
        ax.legend(loc='upper left', prop=font_prop)
        ax.grid(True, alpha=0.3)

        ax.tick_params(labelsize=10)

    def _plot_profit_distribution(self, ax, font_prop):
        """損益分布のプロット"""
        profits = [trade.net_pnl for trade in self.portfolio.trades]
        ax.hist(profits, bins=30, alpha=0.75, color='coral')
        
        ax.set_title('損益分布', fontproperties=font_prop)
        ax.set_xlabel('損益 (USDT)', fontproperties=font_prop)
        ax.set_ylabel('回数', fontproperties=font_prop)
        ax.grid(True, alpha=0.3)
        
        ax.tick_params(labelsize=10)

    def _plot_direction_analysis(self, ax, font_prop):
        """方向別の損益分布のプロット"""
        long_pnl = [t.net_pnl for t in self.portfolio.trades if t.direction == 'long']
        short_pnl = [t.net_pnl for t in self.portfolio.trades if t.direction == 'short']
        
        colors = {
            'boxes': 'lightblue',
            'whiskers': 'black',
            'medians': 'red',
            'caps': 'black'
        }
        
        bp = ax.boxplot([long_pnl, short_pnl],
                       labels=['ロング', 'ショート'],
                       patch_artist=True)
        
        for element, color in colors.items():
            plt.setp(bp[element], color=color)
        plt.setp(bp['boxes'], facecolor='lightblue', alpha=0.6)
        
        ax.set_title('方向別の損益分布', fontproperties=font_prop)
        ax.set_ylabel('損益 (USDT)', fontproperties=font_prop)
        ax.grid(True, alpha=0.3)
        
        for label in ax.get_xticklabels():
            label.set_fontproperties(font_prop)
        ax.tick_params(labelsize=10)

    def _plot_cumulative_profit(self, ax, font_prop):
        """累積損益のプロット"""
        cumulative_pnl = np.cumsum([t.net_pnl for t in self.portfolio.trades])
        ax.plot(cumulative_pnl, color='red', linewidth=2)
        
        ax.set_title('累積損益', fontproperties=font_prop)
        ax.set_xlabel('取引回数', fontproperties=font_prop)
        ax.set_ylabel('累積損益 (USDT)', fontproperties=font_prop)
        ax.grid(True, alpha=0.3)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax.tick_params(labelsize=10)

        ax.fill_between(range(len(cumulative_pnl)), 
                       cumulative_pnl, 
                       0, 
                       where=(cumulative_pnl >= 0), 
                       color='lightgreen', 
                       alpha=0.3)
        ax.fill_between(range(len(cumulative_pnl)), 
                       cumulative_pnl, 
                       0, 
                       where=(cumulative_pnl < 0), 
                       color='lightcoral', 
                       alpha=0.3)

    def calculate_statistics(self) -> Dict:
        """統計情報を計算"""
        if not self.portfolio.trades:
            return {}
        
        profits = [trade.net_pnl for trade in self.portfolio.trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        total_return = ((self.portfolio.current_balance - self.config.initial_balance) 
                       / self.config.initial_balance * 100)
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
        """シャープレシオの計算"""
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
        """最大ドローダウンの計算"""
        if not self.portfolio.equity_history:
            return 0.0
            
        equity_array = np.array(self.portfolio.equity_history)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak * 100
        max_drawdown = np.max(drawdown)
        
        return max_drawdown

    def _calculate_profit_factor(self, winning_trades: List[float], losing_trades: List[float]) -> float:
        """プロフィットファクターの計算"""
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0
            
        return round(total_profit / total_loss, 2)

    def save_results(self):
        """結果の保存"""
        logger.info("結果の保存開始")
        
        # トレード履歴をCSVに保存
        self._save_trade_history()
        
        # 統計情報の保存
        self._save_statistics()
        
        logger.info("結果の保存完了")

    def _save_trade_history(self):
        """トレード履歴の保存"""
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
        """統計情報の保存"""
        stats = self.calculate_statistics()
        with open(self.results_dir / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

class Backtest:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.market_data = MarketData(config)
        self.strategy = ScalpingStrategy(config)
        self.portfolio = PortfolioManager(config)
        self.analyzer = BacktestAnalyzer(config, self.portfolio)

    def run(self) -> bool:
        """バックテスト実行"""
        df = self.market_data.fetch_data()
        if df.empty:
            logger.error("データを取得できませんでした。")
            return False

        logger.info("バックテスト開始")
        
        for i, row in df.iterrows():
            current_equity = self.portfolio.current_balance
            if self.portfolio.position is not None:
                unrealized_pnl = 0
                if self.portfolio.position.direction == 'long':
                    unrealized_pnl = (row['close'] - self.portfolio.position.entry_price) * \
                                   self.portfolio.position.size
                else:
                    unrealized_pnl = (self.portfolio.position.entry_price - row['close']) * \
                                   self.portfolio.position.size
                current_equity += unrealized_pnl
            self.portfolio.equity_history.append(current_equity)
            
            # ポジション有りの場合の確認
            if self.portfolio.position is not None:
                should_close, reason = self.strategy.should_close_position(
                    self.portfolio.position, row
                )
                if should_close:
                    self.portfolio.close_position(row, reason)
                    continue
            
            # 新規ポジション確認
            if self.strategy.should_open_long(row):
                self.portfolio.open_position(row, 'long')
            elif self.strategy.should_open_short(row):
                self.portfolio.open_position(row, 'short')

        logger.info("バックテスト完了")
        return True

    def save_results(self):
        """結果保存"""
        self.analyzer.plot_results()
        self.analyzer.save_results()
        
        stats = self.analyzer.calculate_statistics()
        print("\nバックテスト結果サマリー:")
        print(f"期間: {self.config.start_date.strftime('%Y-%m-%d')} から "
              f"{self.config.end_date.strftime('%Y-%m-%d')}")
        print(f"初期残高: {stats['初期残高']} USDT")
        print(f"最終残高: {stats['最終残高']} USDT")
        print(f"総リターン: {stats['総リターン (%)']}%")
        print(f"総取引回数: {stats['総取引回数']}")
        print(f"勝率: {stats['勝率 (%)']}%")
        print(f"最大ドローダウン: {stats['最大ドローダウン (%)']}%")
        print(f"\n詳細な結果は 'export/results/' ディレクトリに保存されました。")

def run_backtest(
    symbol: str = "XRP/USDT",
    timeframe: str = "5m",
    initial_balance: float = 600.0,
    leverage: int = 5,
    rsi_period: int = 14,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
    tp_percent: float = 0.3,
    days: int = 30
) -> None:
    """バックテストを実行するためのヘルパー関数"""
    try:
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
            start_date=start_date,
            end_date=end_date
        )
        
        backtest = Backtest(config)
        if backtest.run():
            backtest.save_results()
            
    except Exception as e:
        logger.error(f"バックテスト実行中にエラーが発生しました: {str(e)}")
        raise

def main():
    params = {
        'symbol': "XRP/USDT",
        'timeframe': "5m",
        'initial_balance': 600.0,
        'leverage': 5,
        'rsi_period': 14,
        'rsi_overbought': 70.0,
        'rsi_oversold': 30.0,
        'tp_percent': 0.3,
        'days': 60
    }
    
    run_backtest(**params)

if __name__ == "__main__":
    main()