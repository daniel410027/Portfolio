import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re
from collections import Counter 

warnings.filterwarnings('ignore')

class StockDataProcessor:
    """處理單一 CSV 檔案的類別"""
    def __init__(self, df=None):
        self.df = df
        
        # 定義保留欄位
        self.key_columns = ['證券代碼', '年月日']
        self.target_columns = ['return', 'return_tick', 'return_tick_0']
    
    def normalize_column_names(self):
        """消除欄位名稱中的特殊符號，並處理重複名稱"""
        print("   正規化欄位名稱...")
        
        old_columns = self.df.columns.tolist()
        new_columns = []
        
        for col in old_columns:
            new_col = re.sub(r'[^\w\u4e00-\u9fff]', '', col)
            new_columns.append(new_col)
        
        col_counts = Counter(new_columns)
        duplicates = {col: count for col, count in col_counts.items() if count > 1}
        
        if duplicates:
            final_columns = []
            col_counter = {}
            
            for old_col, new_col in zip(old_columns, new_columns):
                if new_col in duplicates:
                    if new_col not in col_counter:
                        col_counter[new_col] = 0
                    else:
                        col_counter[new_col] += 1
                    
                    if col_counter[new_col] == 0:
                        final_col = new_col
                    else:
                        final_col = f"{new_col}_{col_counter[new_col]}"
                    
                    final_columns.append(final_col)
                else:
                    final_columns.append(new_col)
            
            new_columns = final_columns
        
        self.df.columns = new_columns
        self.key_columns = [re.sub(r'[^\w\u4e00-\u9fff]', '', col) for col in self.key_columns]

    def clean_columns(self):
        """清理欄位：移除0值>=99%或缺值>=99%的欄位"""
        print("   清理欄位...")
        
        if '證券代碼' not in self.df.columns or '年月日' not in self.df.columns:
            print("      ❌ 缺少關鍵欄位：證券代碼或年月日")
            return False
        
        self.df['證券代碼'] = self.df['證券代碼'].astype(str).str.extract(r'(\d{4})', expand=False)
        self.df['年月日'] = pd.to_datetime(self.df['年月日'], format='%Y%m%d', errors='coerce')
        
        before_len = len(self.df)
        self.df = self.df.dropna(subset=['年月日'])
        after_len = len(self.df)
        if before_len != after_len:
            print(f"      移除年月日缺失的資料: {before_len - after_len} 筆")
        
        columns_to_keep = self.key_columns.copy()
        columns_to_drop = []
        total_rows = len(self.df)
        
        for col in self.df.columns:
            if col in self.key_columns:
                continue
            
            null_ratio = self.df[col].isnull().sum() / total_rows
            
            zero_ratio = 0
            if pd.api.types.is_numeric_dtype(self.df[col]):
                zero_ratio = (self.df[col] == 0).sum() / total_rows
            
            if null_ratio >= 0.99:
                columns_to_drop.append((col, f'缺失值 {null_ratio:.1%}'))
            elif zero_ratio >= 0.99:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    columns_to_drop.append((col, f'0值 {zero_ratio:.1%}'))
                else:
                    columns_to_keep.append(col)
            else:
                columns_to_keep.append(col)
        
        print(f"      保留欄位: {len(columns_to_keep)}, 移除欄位: {len(columns_to_drop)}")
        
        self.df = self.df[columns_to_keep]
        self.df = self.df.sort_values(by=['證券代碼', '年月日']).reset_index(drop=True)
        
        return True

    def remove_categorical_features(self):
        """移除所有非數值型或非日期型的欄位（類別型特徵）"""
        print("   移除類別型特徵...")
        
        current_cols = set(self.df.columns)
        cols_to_keep = set(self.key_columns)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep.update(numeric_cols)
        
        cols_to_remove = [col for col in current_cols if col not in cols_to_keep]
        
        if '證券代碼' in cols_to_remove:
             cols_to_remove.remove('證券代碼')
        
        self.df = self.df.drop(columns=cols_to_remove, errors='ignore')
        self.df = self.df.sort_values(by=['證券代碼', '年月日']).reset_index(drop=True)
        
        print(f"      移除 {len(cols_to_remove)} 個類別型特徵")
        
        return True
    
    def fix_discontinuous_data(self):
        """填補不連續的交易日（避免 data leakage）"""
        print("   填補不連續資料...")
        
        actual_trading_days = sorted(self.df['年月日'].unique())
        all_stocks = self.df['證券代碼'].unique()
        
        price_col = '收盤價元' if '收盤價元' in self.df.columns else None
        if not price_col:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        fixed_data_list = []
        stocks_with_early_nan = 0
        
        for stock in all_stocks:
            stock_data = self.df[self.df['證券代碼'] == stock].copy()
            
            stock_complete_dates = pd.DataFrame({
                '年月日': actual_trading_days,
                '證券代碼': stock
            })
            
            stock_complete = stock_complete_dates.merge(
                stock_data,
                on=['年月日', '證券代碼'],
                how='left'
            )
            
            stock_complete[price_col] = stock_complete[price_col].fillna(method='ffill')
            
            if stock_complete[price_col].isnull().any():
                stocks_with_early_nan += 1
                first_valid_idx = stock_complete[price_col].first_valid_index()
                if first_valid_idx is not None:
                    stock_complete = stock_complete.loc[first_valid_idx:].copy()
            
            numeric_cols = stock_complete.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != price_col:
                    stock_complete[col] = stock_complete[col].fillna(method='ffill')
                    stock_complete[col] = stock_complete[col].fillna(0)
            
            fixed_data_list.append(stock_complete)
        
        self.df = pd.concat(fixed_data_list, ignore_index=True)
        self.df = self.df.sort_values(['證券代碼', '年月日']).reset_index(drop=True)
        
        print(f"      填補完成，{stocks_with_early_nan} 檔股票被截斷")
        
        return True
    
    # ==================== 技術指標計算方法 ====================
    
    def compute_rsi(self, series, window):
        """計算RSI指標"""
        series = pd.to_numeric(series, errors='coerce')
        delta = series.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        alpha = 1.0 / window
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi.iloc[:window] = np.nan
        
        return rsi
    
    def compute_sma(self, series, window):
        """計算簡單移動平均線"""
        return series.rolling(window=window, min_periods=window).mean()
    
    def compute_ema(self, series, window):
        """計算指數移動平均線"""
        return series.ewm(span=window, adjust=False).mean()
    
    def compute_macd(self, series, fast=12, slow=26, signal=9):
        """計算MACD指標"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def compute_kd(self, high, low, close, k_window=9, d_window=3):
        """計算KD隨機指標"""
        lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
        highest_high = high.rolling(window=k_window, min_periods=k_window).max()
        
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
        
        k_values = []
        k_prev = 50
        for rsv_val in rsv:
            if pd.isna(rsv_val):
                k_values.append(np.nan)
            else:
                k_curr = (2/3) * k_prev + (1/3) * rsv_val
                k_values.append(k_curr)
                k_prev = k_curr
        
        k_series = pd.Series(k_values, index=close.index)
        
        d_values = []
        d_prev = 50
        for k_val in k_series:
            if pd.isna(k_val):
                d_values.append(np.nan)
            else:
                d_curr = (2/3) * d_prev + (1/3) * k_val
                d_values.append(d_curr)
                d_prev = d_curr
        
        d_series = pd.Series(d_values, index=close.index)
        
        return k_series, d_series
    
    def compute_williams_r(self, high, low, close, window=14):
        """計算威廉指標"""
        highest_high = high.rolling(window=window, min_periods=window).max()
        lowest_low = low.rolling(window=window, min_periods=window).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
        return wr
    
    def compute_roc(self, series, window=10):
        """計算變動率"""
        return 100 * (series - series.shift(window)) / (series.shift(window) + 1e-9)
    
    def compute_atr(self, high, low, close, window=14):
        """計算平均真實區間"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=window, adjust=False).mean()
        
        return atr
    
    def compute_bollinger_bands(self, series, window=20, num_std=2):
        """計算布林通道"""
        middle = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std()
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        
        percent_b = (series - lower) / (upper - lower + 1e-9)
        
        return upper, middle, lower, percent_b
    
    def compute_obv(self, close, volume):
        """計算能量潮"""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=close.index)
    
    def compute_vwap(self, high, low, close, volume):
        """計算成交量加權均價"""
        typical_price = (high + low + close) / 3
        cumulative_tpv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_tpv / (cumulative_volume + 1e-9)
        return vwap
    
    def compute_momentum(self, series, window=10):
        """計算動量"""
        return series - series.shift(window)
    
    def compute_cci(self, high, low, close, window=20):
        """計算商品通道指標"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window, min_periods=window).mean()
        mean_deviation = typical_price.rolling(window=window, min_periods=window).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-9)
        return cci
    
    # ==================== 特徵新增方法 ====================
    
    def add_rsi_features(self):
        """新增RSI特徵"""
        print("   計算RSI特徵...")
        
        price_col = '收盤價元'
        if price_col not in self.df.columns:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        RSI_WINDOWS = [5, 10, 20, 60, 120, 240]
        
        for window in RSI_WINDOWS:
            rsi_col = f'RSI_{window}'
            self.df[rsi_col] = self.df.groupby('證券代碼')[price_col].transform(
                lambda x: self.compute_rsi(x, window)
            )
            self.df[rsi_col] = self.df[rsi_col].fillna(-1)
        
        return True
    
    def add_moving_average_features(self):
        """新增移動平均線特徵"""
        print("   計算移動平均線特徵...")
        
        price_col = '收盤價元'
        if price_col not in self.df.columns:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        MA_WINDOWS = [5, 10, 20, 60, 120, 240]
        
        for window in MA_WINDOWS:
            sma_col = f'SMA_{window}'
            self.df[sma_col] = self.df.groupby('證券代碼')[price_col].transform(
                lambda x: self.compute_sma(x, window)
            )
            
            ema_col = f'EMA_{window}'
            self.df[ema_col] = self.df.groupby('證券代碼')[price_col].transform(
                lambda x: self.compute_ema(x, window)
            )
            
            self.df[f'Price_SMA{window}_Ratio'] = (
                self.df[price_col] / (self.df[sma_col] + 1e-9) - 1
            ) * 100
        
        ma_cols = [col for col in self.df.columns if 'SMA' in col or 'EMA' in col or 'Ratio' in col]
        for col in ma_cols:
            self.df[col] = self.df[col].fillna(0)
        
        return True
    
    def add_macd_features(self):
        """新增MACD特徵"""
        print("   計算MACD特徵...")
        
        price_col = '收盤價元'
        if price_col not in self.df.columns:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        def calc_macd_group(group):
            macd, signal, hist = self.compute_macd(group)
            return pd.DataFrame({
                'MACD': macd,
                'MACD_Signal': signal,
                'MACD_Hist': hist
            }, index=group.index)
        
        macd_df = self.df.groupby('證券代碼')[price_col].apply(calc_macd_group)
        macd_df = macd_df.reset_index(level=0, drop=True)
        
        self.df['MACD'] = macd_df['MACD']
        self.df['MACD_Signal'] = macd_df['MACD_Signal']
        self.df['MACD_Hist'] = macd_df['MACD_Hist']
        
        for col in ['MACD', 'MACD_Signal', 'MACD_Hist']:
            self.df[col] = self.df[col].fillna(0)
        
        return True
    
    def add_kd_features(self):
        """新增KD隨機指標特徵"""
        print("   計算KD隨機指標特徵...")
        
        high_col = '最高價元'
        low_col = '最低價元'
        close_col = '收盤價元'
        
        required_cols = [high_col, low_col, close_col]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"      ❌ 找不到 {col} 欄位")
                return False
        
        def calc_kd_group(group):
            k, d = self.compute_kd(
                group[high_col], 
                group[low_col], 
                group[close_col]
            )
            return pd.DataFrame({'K': k, 'D': d}, index=group.index)
        
        kd_df = self.df.groupby('證券代碼').apply(calc_kd_group)
        kd_df = kd_df.reset_index(level=0, drop=True)
        
        self.df['K'] = kd_df['K']
        self.df['D'] = kd_df['D']
        self.df['KD_Diff'] = self.df['K'] - self.df['D']
        
        for col in ['K', 'D', 'KD_Diff']:
            self.df[col] = self.df[col].fillna(50)
        
        return True
    
    def add_williams_r_features(self):
        """新增威廉指標特徵"""
        print("   計算威廉指標特徵...")
        
        high_col = '最高價元'
        low_col = '最低價元'
        close_col = '收盤價元'
        
        required_cols = [high_col, low_col, close_col]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"      ❌ 找不到 {col} 欄位")
                return False
        
        WR_WINDOWS = [14, 28]
        
        for window in WR_WINDOWS:
            wr_col = f'Williams_R_{window}'
            
            def calc_wr(group):
                return self.compute_williams_r(
                    group[high_col], group[low_col], group[close_col], window
                )
            
            self.df[wr_col] = self.df.groupby('證券代碼').apply(calc_wr).reset_index(level=0, drop=True)
            self.df[wr_col] = self.df[wr_col].fillna(-50)
        
        return True
    
    def add_roc_momentum_features(self):
        """新增ROC和動量特徵"""
        print("   計算ROC和動量特徵...")
        
        price_col = '收盤價元'
        if price_col not in self.df.columns:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        WINDOWS = [5, 10, 20]
        
        for window in WINDOWS:
            roc_col = f'ROC_{window}'
            self.df[roc_col] = self.df.groupby('證券代碼')[price_col].transform(
                lambda x: self.compute_roc(x, window)
            )
            
            mom_col = f'Momentum_{window}'
            self.df[mom_col] = self.df.groupby('證券代碼')[price_col].transform(
                lambda x: self.compute_momentum(x, window)
            )
        
        roc_mom_cols = [col for col in self.df.columns if 'ROC' in col or 'Momentum' in col]
        for col in roc_mom_cols:
            self.df[col] = self.df[col].fillna(0)
        
        return True
    
    def add_volatility_features(self):
        """新增波動性特徵 (ATR, Bollinger Bands)"""
        print("   計算波動性特徵...")
        
        high_col = '最高價元'
        low_col = '最低價元'
        close_col = '收盤價元'
        
        required_cols = [high_col, low_col, close_col]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"      ❌ 找不到 {col} 欄位")
                return False
        
        ATR_WINDOWS = [14, 28]
        for window in ATR_WINDOWS:
            atr_col = f'ATR_{window}'
            
            def calc_atr(group):
                return self.compute_atr(
                    group[high_col], group[low_col], group[close_col], window
                )
            
            self.df[atr_col] = self.df.groupby('證券代碼').apply(calc_atr).reset_index(level=0, drop=True)
            self.df[f'ATR_{window}_Pct'] = self.df[atr_col] / (self.df[close_col] + 1e-9) * 100
        
        BB_WINDOWS = [20]
        for window in BB_WINDOWS:
            def calc_bb(group):
                upper, middle, lower, pct_b = self.compute_bollinger_bands(group, window)
                return pd.DataFrame({
                    f'BB_Upper_{window}': upper,
                    f'BB_Middle_{window}': middle,
                    f'BB_Lower_{window}': lower,
                    f'BB_PctB_{window}': pct_b
                }, index=group.index)
            
            bb_df = self.df.groupby('證券代碼')[close_col].apply(calc_bb)
            bb_df = bb_df.reset_index(level=0, drop=True)
            
            for bb_col in bb_df.columns:
                self.df[bb_col] = bb_df[bb_col]
        
        vol_cols = [col for col in self.df.columns if 'ATR' in col or 'BB_' in col]
        for col in vol_cols:
            self.df[col] = self.df[col].fillna(0)
        
        return True
    
    def add_volume_features(self):
        """新增成交量相關特徵 (OBV, VWAP)"""
        print("   計算成交量特徵...")
        
        high_col = '最高價元'
        low_col = '最低價元'
        close_col = '收盤價元'
        volume_col = '成交量千股'
        
        if volume_col not in self.df.columns:
            possible_vol_cols = [col for col in self.df.columns if '成交量' in col or 'volume' in col.lower()]
            if possible_vol_cols:
                volume_col = possible_vol_cols[0]
            else:
                print("      ⚠️ 找不到成交量欄位，跳過")
                return True
        
        required_cols = [high_col, low_col, close_col, volume_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"      ⚠️ 缺少欄位: {missing_cols}，跳過")
            return True
        
        def calc_obv(group):
            return self.compute_obv(group[close_col], group[volume_col])
        
        self.df['OBV'] = self.df.groupby('證券代碼').apply(calc_obv).reset_index(level=0, drop=True)
        
        self.df['OBV_ROC'] = self.df.groupby('證券代碼')['OBV'].transform(
            lambda x: x.pct_change(periods=5) * 100
        )
        
        def calc_vwap(group):
            return self.compute_vwap(
                group[high_col], group[low_col], group[close_col], group[volume_col]
            )
        
        self.df['VWAP'] = self.df.groupby('證券代碼').apply(calc_vwap).reset_index(level=0, drop=True)
        
        self.df['Price_VWAP_Ratio'] = (self.df[close_col] / (self.df['VWAP'] + 1e-9) - 1) * 100
        
        VOL_WINDOWS = [5, 20]
        for window in VOL_WINDOWS:
            self.df[f'Volume_SMA_{window}'] = self.df.groupby('證券代碼')[volume_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            self.df[f'Volume_Ratio_{window}'] = self.df[volume_col] / (self.df[f'Volume_SMA_{window}'] + 1e-9)
        
        vol_cols = [col for col in self.df.columns if 'OBV' in col or 'VWAP' in col or 'Volume' in col]
        for col in vol_cols:
            self.df[col] = self.df[col].fillna(0)
        
        return True
    
    def add_cci_features(self):
        """新增CCI商品通道指標"""
        print("   計算CCI特徵...")
        
        high_col = '最高價元'
        low_col = '最低價元'
        close_col = '收盤價元'
        
        required_cols = [high_col, low_col, close_col]
        for col in required_cols:
            if col not in self.df.columns:
                print(f"      ❌ 找不到 {col} 欄位")
                return False
        
        CCI_WINDOWS = [14, 20]
        
        for window in CCI_WINDOWS:
            cci_col = f'CCI_{window}'
            
            def calc_cci(group):
                return self.compute_cci(
                    group[high_col], group[low_col], group[close_col], window
                )
            
            self.df[cci_col] = self.df.groupby('證券代碼').apply(calc_cci).reset_index(level=0, drop=True)
            self.df[cci_col] = self.df[cci_col].fillna(0)
        
        return True
    
    def add_return_features(self):
        """新增return相關特徵（作為預測標籤）
        Return = (後天開盤價 - 明天開盤價) / 明天開盤價
        return_tick: return > 1% 設為 1，否則為 0
        """
        print("   計算return特徵...")
        
        price_col = '收盤價元'
        open_col = '開盤價元'
        
        if price_col not in self.df.columns:
            print("      ❌ 找不到收盤價欄位")
            return False
        
        if open_col not in self.df.columns:
            print("      ❌ 找不到開盤價欄位")
            return False
        
        def calc_intraday_return(group):
            tomorrow_open = group[open_col].shift(-1)
            day_after_tomorrow_open = group[open_col].shift(-2)
            return (day_after_tomorrow_open - tomorrow_open) / tomorrow_open
        
        self.df['return'] = self.df.groupby('證券代碼').apply(
            calc_intraday_return
        ).reset_index(level=0, drop=True)
        
        CLIP_VALUE = 0.15
        self.df['return'] = self.df['return'].clip(lower=-CLIP_VALUE, upper=CLIP_VALUE)
        
        # return_tick_0: 簡單正負分類
        self.df['return_tick_0'] = (self.df['return'] > 0).astype(int)
        self.df.loc[self.df['return'].isna(), 'return_tick_0'] = np.nan
        
        # return_tick: 使用絕對閾值 1%
        self.df['return_tick'] = (self.df['return'] > 0.01).astype(int)
        self.df.loc[self.df['return'].isna(), 'return_tick'] = np.nan
        
        return True
    
    def process(self):
        """執行完整處理流程"""
        # 正規化欄位名稱
        self.normalize_column_names()
        
        # 清理欄位
        if not self.clean_columns():
            return None
        
        # 移除類別型特徵
        if not self.remove_categorical_features():
            return None
        
        # 填補不連續資料
        if not self.fix_discontinuous_data():
            return None
        
        # RSI特徵
        if not self.add_rsi_features():
            return None
        
        # 移動平均線特徵
        if not self.add_moving_average_features():
            return None
        
        # MACD特徵
        if not self.add_macd_features():
            return None
        
        # KD特徵
        if not self.add_kd_features():
            return None
        
        # 威廉指標
        if not self.add_williams_r_features():
            return None
        
        # ROC和動量
        if not self.add_roc_momentum_features():
            return None
        
        # 波動性特徵
        if not self.add_volatility_features():
            return None
        
        # 成交量特徵
        if not self.add_volume_features():
            return None
        
        # CCI特徵
        if not self.add_cci_features():
            return None
        
        # return特徵
        if not self.add_return_features():
            return None
        
        return self.df


def process_merge_csv():
    """處理 database/merge.csv 並輸出到 database/merge_processed.csv"""
    
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / 'database' / 'merge.csv'
    output_file = base_dir / 'database' / 'merge_processed.csv'
    
    print("=" * 60)
    print("處理 merge.csv")
    print("=" * 60)
    
    # ====== 步驟 1: 讀取資料 ======
    try:
        print("\n📁 載入 merge.csv...")
        # 自動嘗試多種編碼
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-sig', 'big5', 'gbk', 'latin-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding, sep='\t', low_memory=False)
                print(f"   ✓ 使用編碼: {encoding}")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if df is None:
            print(f"❌ 無法讀取檔案，嘗試了所有編碼")
            return
        
        print(f"✅ 形狀: {df.shape}")
        print(f"   欄位數: {len(df.columns)}")
        print(f"   資料筆數: {len(df)}")
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {input_file}")
        return
    except Exception as e:
        print(f"❌ 讀取失敗: {e}")
        return
    
    # ====== 步驟 2: 處理資料 ======
    print(f"\n{'='*60}")
    print("開始處理資料...")
    print(f"{'='*60}")
    
    processor = StockDataProcessor(df=df)
    processed_df = processor.process()
    
    if processed_df is None:
        print("\n❌ 處理失敗")
        return
    
    # ====== 步驟 3: 儲存結果 ======
    print(f"\n{'='*60}")
    print("儲存處理後的資料...")
    print(f"{'='*60}")
    
    processed_df.to_csv(output_file, index=False, encoding='utf-16', sep='\t')
    
    print(f"\n🎉 處理完成！")
    print(f"   輸入: {input_file}")
    print(f"   輸出: {output_file}")
    print(f"   最終形狀: {processed_df.shape}")
    
    if '證券代碼' in processed_df.columns:
        print(f"   股票數量: {processed_df['證券代碼'].nunique()}")
    if '年月日' in processed_df.columns:
        print(f"   日期範圍: {processed_df['年月日'].min()} ~ {processed_df['年月日'].max()}")
    
    print(f"\n新增的技術指標包含:")
    indicator_cols = [col for col in processed_df.columns if any(
        ind in col for ind in ['RSI', 'SMA', 'EMA', 'MACD', 'KD', 'Williams', 
                                'ROC', 'Momentum', 'ATR', 'BB_', 'OBV', 'VWAP', 'CCI']
    )]
    print(f"   技術指標欄位數: {len(indicator_cols)}")


if __name__ == "__main__":
    process_merge_csv()