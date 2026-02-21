# requirements.txt
# streamlit==1.28.0
# pandas==2.0.3
# numpy==1.24.3
# tushare==1.2.89
# plotly==5.17.0
# matplotlib==3.7.2
# scikit-learn==1.3.0
# xgboost==1.7.6
# lightgbm==4.1.0
# tensorflow==2.15.0

# main.py - 量化分析系统主程序
import streamlit as st
import pandas as pd
import numpy as np
import tushare as ts
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import time
import warnings
import pickle
from pathlib import Path
import requests
import hashlib
from io import BytesIO
import traceback
from datetime import datetime, timedelta  # 修复这里的导入

# 机器学习相关库
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# 交易策略和回测相关库
import backtrader as bt
from backtrader.feeds import PandasData
import talib
from typing import Dict, List, Tuple, Optional


# 动态导入TensorFlow
def try_import_tensorflow():
    """动态导入TensorFlow"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, Model, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        return tf, True
    except ImportError:
        return None, False


# 检查TensorFlow是否可用
def is_tensorflow_available():
    _, available = try_import_tensorflow()
    return available


warnings.filterwarnings('ignore')


# ==================== 初始化配置 ====================

# 设置中文字体
def setup_chinese_font():
    try:
        if os.name == 'nt':
            plt.rcParams['font.sans-serif'] = ['SimHei']
        elif os.name == 'posix':
            plt.rcParams['font.sans-serif'] = ['PingFang HK']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


setup_chinese_font()

# 配置常量
TUSHARE_TOKEN = "dcb6a1ee13f3cb60cc84968e8cc7d4444890b5800a54d383ea3c2db3"
DATA_DIR = Path("quant_data")
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
DOWNLOAD_DIR = DATA_DIR / "downloads"
FEATURE_CONFIG_FILE = DATA_DIR / "feature_config.json"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DOWNLOAD_DIR.mkdir(exist_ok=True)


# ==================== 数据获取模块（增强版）====================
class DataFetcher:
    def __init__(self, token=TUSHARE_TOKEN):
        self.token = token
        self.pro = self._init_tushare()
        self.request_count = 0
        self.last_request_time = time.time()
        self.min_interval = 0.3

    def _init_tushare(self):
        try:
            pro = ts.pro_api(self.token)
            # 快速测试
            test = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240105',
                             fields='ts_code')
            if not test.empty:
                print("✅ Tushare连接成功")
                return pro
        except Exception as e:
            print(f"❌ Tushare连接失败: {e}")
            return None

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def fetch_stock_data(self, symbol, start_date, end_date, adj=None):
        """获取股票数据，支持复权选项"""
        if not self.pro:
            return None

        # 构建缓存键，包含复权信息
        cache_key = f"{symbol}_{start_date}_{end_date}_{adj if adj else 'none'}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        self._rate_limit()
        try:
            # 根据复权方式选择接口
            if adj is None or adj == 'none':
                # 不复权
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                )
            elif adj == 'qfq':
                # 前复权
                df = ts.pro_bar(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adj='qfq',
                    factors=['tor', 'vr']
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        'trade_date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'vol'
                    })
            elif adj == 'hfq':
                # 后复权
                df = ts.pro_bar(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adj='hfq',
                    factors=['tor', 'vr']
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        'trade_date': 'trade_date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'vol': 'vol'
                    })
            else:
                # 默认不复权
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                )

            if df is None or df.empty:
                return None

            df = self._process_data(df)
            self._save_cache(df, cache_key)
            return df
        except Exception as e:
            print(f"获取{symbol}失败: {e}")
            # 尝试备用方法
            try:
                df = self.pro.daily(
                    ts_code=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount'
                )
                if df is None or df.empty:
                    return None
                df = self._process_data(df)
                self._save_cache(df, cache_key)
                return df
            except Exception as e2:
                print(f"备用获取{symbol}也失败: {e2}")
                return None

    def _process_data(self, df):
        df = df.sort_values('trade_date')
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 确保列存在
        if 'close' in df.columns and 'pre_close' in df.columns:
            df['returns'] = df['close'].pct_change()
        elif 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
        else:
            df['returns'] = 0

        # 计算技术指标
        if 'close' in df.columns:
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA10'] = df['close'].rolling(10).mean()
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA60'] = df['close'].rolling(60).mean()
            df['RSI'] = self._calculate_rsi(df['close'])
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = self._calculate_macd_full(df['close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = self._calculate_bollinger_bands(df['close'])
            df['ATR'] = self._calculate_atr(df)
            df['OBV'] = self._calculate_obv(df)

        # 成交量相关指标
        if 'vol' in df.columns:
            df['VOL_MA5'] = df['vol'].rolling(5).mean()
            df['VOL_MA10'] = df['vol'].rolling(10).mean()

        # 价格动量
        if 'close' in df.columns:
            df['MOM'] = df['close'] - df['close'].shift(5)
            df['ROC'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

        return df.reset_index(drop=True)

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def _calculate_macd_full(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
        return macd, signal_line, macd_hist

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    def _calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        return atr

    def _calculate_obv(self, df):
        obv = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
        return obv

    def _save_cache(self, df, key):
        try:
            cache_file = CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        except:
            pass

    def _load_cache(self, key):
        try:
            cache_file = CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        return None

    def save_to_file(self, df, symbol, format='csv'):
        if df is None or df.empty:
            return False
        try:
            symbol_clean = symbol.replace('.', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format.lower() == 'csv':
                filename = f"{symbol_clean}_{timestamp}.csv"
                filepath = DATA_DIR / filename
                df.to_csv(filepath, index=False, encoding='utf-8-sig')
                return filepath
            elif format.lower() == 'xlsx':
                filename = f"{symbol_clean}_{timestamp}.xlsx"
                filepath = DATA_DIR / filename
                df.to_excel(filepath, index=False)
                return filepath
            else:
                return None
        except Exception as e:
            print(f"保存失败: {e}")
            return None


# ==================== 预测模型模块（增强版）====================
class StockPredictor:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.best_model = None
        self.best_model_type = None
        self.feature_columns = []
        self.loaded_model_info = None

    def prepare_features(self, df, lookback=30, forecast_days=5, selected_features=None):
        """准备特征数据，支持特征选择"""
        if df is None or len(df) < lookback + forecast_days:
            return None, None, None, None

        # 可用的特征列
        all_feature_cols = ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20', 'MA60',
                            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                            'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'OBV',
                            'VOL_MA5', 'VOL_MA10', 'MOM', 'ROC']

        # 如果未指定特征，使用所有可用特征
        if selected_features is None:
            available_cols = [col for col in all_feature_cols if col in df.columns]
        else:
            # 确保只选择存在的特征
            available_cols = [col for col in selected_features if col in df.columns]

        # 如果可用特征太少，使用默认特征
        if len(available_cols) < 3:
            default_cols = ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20']
            available_cols = [col for col in default_cols if col in df.columns]

        self.feature_columns = available_cols

        # 准备数据
        data = df[available_cols].fillna(method='ffill').fillna(0).values

        # 创建特征和标签
        X, y = [], []
        for i in range(len(data) - lookback - forecast_days + 1):
            X.append(data[i:i + lookback])
            # 预测未来几天的收盘价
            if 'close' in df.columns:
                close_idx = available_cols.index('close') if 'close' in available_cols else 0
                y.append(data[i + lookback:i + lookback + forecast_days, close_idx])
            else:
                y.append(data[i + lookback:i + lookback + forecast_days, 0])

        X = np.array(X)
        y = np.array(y)

        # 分割数据
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 标准化
        self.scalers['X'] = StandardScaler()
        self.scalers['y'] = StandardScaler()

        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = self.scalers['X'].fit_transform(X_train_reshaped)
        X_test_scaled = self.scalers['X'].transform(X_test_reshaped)

        X_train = X_train_scaled.reshape(X_train.shape)
        X_test = X_test_scaled.reshape(X_test.shape)

        y_train_scaled = self.scalers['y'].fit_transform(y_train)
        y_test_scaled = self.scalers['y'].transform(y_test)

        return X_train, y_train_scaled, X_test, y_test_scaled

    def train_random_forest(self, X_train, y_train, **params):
        # 将3D数据转换为2D
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42
        )
        model.fit(X_train_2d, y_train)
        return model

    def train_xgboost(self, X_train, y_train, **params):
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        model = xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            random_state=42
        )
        model.fit(X_train_2d, y_train)
        return model

    def train_lstm(self, X_train, y_train, **params):
        """训练LSTM模型 - 使用动态导入"""
        tf_module, tf_available = try_import_tensorflow()
        if not tf_available:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow==2.15.0")

        # 现在可以使用tf_module
        model = tf_module.keras.Sequential([
            tf_module.keras.layers.LSTM(
                params.get('lstm_units', 50),
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ),
            tf_module.keras.layers.Dropout(params.get('dropout_rate', 0.2)),
            tf_module.keras.layers.LSTM(params.get('lstm_units', 50)),
            tf_module.keras.layers.Dropout(params.get('dropout_rate', 0.2)),
            tf_module.keras.layers.Dense(y_train.shape[1])
        ])

        model.compile(
            optimizer=tf_module.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )

        early_stopping = tf_module.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('patience', 10),
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        return model, history

    def train_lstm_attention(self, X_train, y_train, **params):
        """训练带Attention的LSTM模型 - 使用动态导入"""
        tf_module, tf_available = try_import_tensorflow()
        if not tf_available:
            raise ImportError("TensorFlow未安装，请运行: pip install tensorflow==2.15.0")

        inputs = tf_module.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

        # LSTM层
        lstm_out = tf_module.keras.layers.LSTM(
            params.get('lstm_units', 50),
            return_sequences=True
        )(inputs)
        lstm_out = tf_module.keras.layers.Dropout(params.get('dropout_rate', 0.2))(lstm_out)

        # Attention机制
        attention = tf_module.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf_module.keras.layers.Flatten()(attention)
        attention = tf_module.keras.layers.Activation('softmax')(attention)
        attention = tf_module.keras.layers.RepeatVector(params.get('lstm_units', 50))(attention)
        attention = tf_module.keras.layers.Permute([2, 1])(attention)

        # 应用Attention
        merged = tf_module.keras.layers.multiply([lstm_out, attention])
        merged = tf_module.keras.layers.Flatten()(merged)

        # 输出层
        outputs = tf_module.keras.layers.Dense(y_train.shape[1])(merged)

        model = tf_module.keras.models.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf_module.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )

        early_stopping = tf_module.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('patience', 10),
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        return model, history

    def continue_training(self, model, X_train, y_train, **params):
        """继续训练现有模型"""
        tf_module, tf_available = try_import_tensorflow()
        if not tf_available:
            raise ImportError("TensorFlow未安装")

        # 设置优化器
        if params.get('learning_rate'):
            model.compile(
                optimizer=tf_module.keras.optimizers.Adam(learning_rate=params.get('learning_rate')),
                loss='mse',
                metrics=['mae']
            )

        early_stopping = tf_module.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=params.get('patience', 10),
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 20),
            batch_size=params.get('batch_size', 32),
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        return model, history

    def evaluate_model(self, model, X_test, y_test, model_type, y_original=None):
        """评估模型性能，包括方向准确率"""
        if model_type in ['random_forest', 'xgboost']:
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
            y_pred = model.predict(X_test_2d)
        else:
            y_pred = model.predict(X_test)

        # 反标准化
        if hasattr(self, 'scalers') and 'y' in self.scalers:
            y_pred_original = self.scalers['y'].inverse_transform(y_pred)
            y_test_original = self.scalers['y'].inverse_transform(y_test)
        else:
            y_pred_original = y_pred
            y_test_original = y_test

        # 如果提供了原始y值，使用它（用于方向准确率计算）
        if y_original is not None:
            y_test_original = y_original

        # 计算标准指标
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        # 计算方向准确率
        direction_accuracy = 0
        if len(y_test_original) > 1:
            # 获取价格变化方向
            actual_direction = np.sign(np.diff(y_test_original.flatten()))
            pred_direction = np.sign(np.diff(y_pred_original.flatten()))

            # 计算准确率（忽略变化为0的情况）
            valid_indices = (actual_direction != 0)
            if np.any(valid_indices):
                direction_accuracy = np.mean(
                    actual_direction[valid_indices] == pred_direction[valid_indices]
                )

        return {
            'predictions': y_pred_original,
            'actual': y_test_original,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

    def predict_future(self, model, latest_data, forecast_days=5, model_type='random_forest'):
        """预测未来价格"""
        if model_type in ['random_forest', 'xgboost']:
            latest_2d = latest_data.reshape(1, -1)
            predictions = model.predict(latest_2d)
        else:
            predictions = model.predict(latest_data.reshape(1, latest_data.shape[0], latest_data.shape[1]))

        if hasattr(self, 'scalers') and 'y' in self.scalers:
            predictions = self.scalers['y'].inverse_transform(predictions)

        return predictions[0]

    def save_model(self, model, model_type, symbol, feature_columns, params=None, path=None):
        """保存模型和相关配置"""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{symbol}_{model_type}_{timestamp}"
            path = MODEL_DIR / f"{model_name}.pkl"

        # 准备保存的数据
        save_data = {
            'model': model,
            'model_type': model_type,
            'symbol': symbol,
            'feature_columns': feature_columns,
            'params': params if params else {},
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'scaler_X': self.scalers.get('X'),
            'scaler_y': self.scalers.get('y')
        }

        try:
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
            return path
        except Exception as e:
            print(f"保存模型失败: {e}")
            return None

    def load_model(self, model_path):
        """加载保存的模型"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.loaded_model_info = model_data

            # 恢复scaler
            if 'scaler_X' in model_data:
                self.scalers['X'] = model_data['scaler_X']
            if 'scaler_y' in model_data:
                self.scalers['y'] = model_data['scaler_y']

            return model_data['model'], model_data['model_type'], model_data
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None, None, None

    def get_saved_models(self):
        """获取所有保存的模型"""
        model_files = list(MODEL_DIR.glob("*.pkl"))
        models_info = []

        for model_file in model_files:
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                models_info.append({
                    'path': model_file,
                    'name': model_file.stem,
                    'symbol': model_data.get('symbol', '未知'),
                    'model_type': model_data.get('model_type', '未知'),
                    'save_time': model_data.get('save_time', '未知'),
                    'feature_count': len(model_data.get('feature_columns', [])),
                    'params': model_data.get('params', {})
                })
            except:
                continue

        # 按保存时间排序
        models_info.sort(key=lambda x: x['save_time'], reverse=True)
        return models_info


# ==================== 可视化模块（增强版）====================
class DataVisualizer:
    def __init__(self):
        self.up_color = '#26a69a'
        self.down_color = '#ef5350'
        self.color_palette = px.colors.qualitative.Set3

    def plot_candlestick(self, df, symbol):
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3]
        )
        # K线
        fig.add_trace(
            go.Candlestick(
                x=df['trade_date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='价格',
                increasing_line_color=self.up_color,
                decreasing_line_color=self.down_color
            ),
            row=1, col=1
        )
        # 移动平均线
        for ma in ['MA5', 'MA10', 'MA20']:
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['trade_date'],
                        y=df[ma],
                        mode='lines',
                        name=ma,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        # 成交量
        colors = [self.up_color if c >= o else self.down_color for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(
                x=df['trade_date'],
                y=df['vol'],
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        fig.update_layout(
            title=f'{symbol} 股票走势',
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white'
        )
        return fig

    def plot_predictions(self, actual, predictions, dates, forecast_days=5, model_name=""):
        """绘制预测结果，包括历史预测和未来预测"""
        fig = go.Figure()

        # 实际值
        fig.add_trace(go.Scatter(
            x=dates[:len(actual)],
            y=actual.flatten() if len(actual.shape) > 1 else actual,
            mode='lines',
            name='实际值',
            line=dict(color='blue', width=2)
        ))

        # 预测值
        pred_start_idx = len(actual) - len(predictions)
        if pred_start_idx < 0:
            pred_start_idx = 0

        fig.add_trace(go.Scatter(
            x=dates[pred_start_idx:pred_start_idx + len(predictions)],
            y=predictions.flatten() if len(predictions.shape) > 1 else predictions,
            mode='lines+markers',
            name='预测值',
            line=dict(color='red', width=2, dash='dash')
        ))

        # 未来预测
        if forecast_days > 0:
            last_date = dates[-1]
            if isinstance(last_date, np.datetime64):
                last_date = pd.to_datetime(last_date)

            future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
            future_predictions = predictions[-forecast_days:] if len(predictions) >= forecast_days else predictions

            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions.flatten() if len(future_predictions.shape) > 1 else future_predictions,
                mode='lines+markers',
                name='未来预测',
                line=dict(color='green', width=2, dash='dot')
            ))

        title = '股价预测结果'
        if model_name:
            title = f'{model_name} - {title}'

        fig.update_layout(
            title=title,
            height=500,
            template='plotly_white',
            xaxis_title='日期',
            yaxis_title='价格',
            hovermode='x unified'
        )

        return fig

    def plot_model_comparison(self, results_dict):
        models = list(results_dict.keys())
        metrics = ['rmse', 'mae', 'r2', 'direction_accuracy']
        metric_names = ['RMSE (越低越好)', 'MAE (越低越好)', 'R² (越高越好)', '方向准确率 (越高越好)']

        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metric_names
        )

        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    text=[f'{v:.4f}' for v in values],
                    textposition='auto',
                    marker_color=self.color_palette[i % len(self.color_palette)]
                ),
                row=1, col=i + 1
            )

            # 设置y轴范围
            if metric == 'r2':
                fig.update_yaxes(range=[-0.5, 1], row=1, col=i + 1)
            elif metric == 'direction_accuracy':
                fig.update_yaxes(range=[0, 1], row=1, col=i + 1)

        fig.update_layout(
            height=500,
            showlegend=False,
            template='plotly_white',
            title_text="模型性能对比"
        )
        return fig

    def plot_validation_results(self, actual, predictions, dates, metrics, model_name=""):
        """绘制验证结果，包括详细指标"""
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=['预测结果对比', '预测误差']
        )

        # 第一个子图：预测结果对比
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual.flatten() if len(actual.shape) > 1 else actual,
                mode='lines',
                name='实际值',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions.flatten() if len(predictions.shape) > 1 else predictions,
                mode='lines',
                name='预测值',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )

        # 第二个子图：预测误差
        errors = predictions.flatten() - actual.flatten()
        fig.add_trace(
            go.Bar(
                x=dates,
                y=errors,
                name='预测误差',
                marker_color=['green' if e >= 0 else 'red' for e in errors]
            ),
            row=2, col=1
        )

        # 添加指标注释
        metrics_text = f"""
        RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}<br>
        R²: {metrics['r2']:.4f} | 方向准确率: {metrics['direction_accuracy']:.2%}
        """

        title = '模型验证结果'
        if model_name:
            title = f'{model_name} - {title}<br><span style="font-size:12px">{metrics_text}</span>'
        else:
            title = f'{title}<br><span style="font-size:12px">{metrics_text}</span>'

        fig.update_layout(
            title_text=title,
            height=600,
            template='plotly_white',
            showlegend=True
        )

        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="误差", row=2, col=1)

        return fig

    def plot_training_history(self, history, model_name=""):
        """绘制训练历史"""
        if not history:
            return None

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['损失函数', 'MAE']
        )

        # 损失函数
        fig.add_trace(
            go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='训练损失',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        if 'val_loss' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['val_loss'],
                    mode='lines',
                    name='验证损失',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )

        # MAE
        if 'mae' in history.history:
            fig.add_trace(
                go.Scatter(
                    y=history.history['mae'],
                    mode='lines',
                    name='训练MAE',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )

            if 'val_mae' in history.history:
                fig.add_trace(
                    go.Scatter(
                        y=history.history['val_mae'],
                        mode='lines',
                        name='验证MAE',
                        line=dict(color='red', width=2, dash='dash')
                    ),
                    row=1, col=2
                )

        title = '训练历史'
        if model_name:
            title = f'{model_name} - {title}'

        fig.update_layout(
            title_text=title,
            height=400,
            template='plotly_white',
            showlegend=True
        )

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)

        return fig


# ==================== 特征选择工具 ====================
class FeatureSelector:
    def __init__(self):
        self.available_features = [
            ('close', '收盘价'),
            ('vol', '成交量'),
            ('returns', '日收益率'),
            ('MA5', '5日均线'),
            ('MA10', '10日均线'),
            ('MA20', '20日均线'),
            ('MA60', '60日均线'),
            ('RSI', '相对强弱指标'),
            ('MACD', 'MACD'),
            ('MACD_signal', 'MACD信号线'),
            ('MACD_hist', 'MACD柱状图'),
            ('BB_upper', '布林带上轨'),
            ('BB_middle', '布林带中轨'),
            ('BB_lower', '布林带下轨'),
            ('ATR', '平均真实波幅'),
            ('OBV', '能量潮'),
            ('VOL_MA5', '成交量5日均线'),
            ('VOL_MA10', '成交量10日均线'),
            ('MOM', '动量指标'),
            ('ROC', '价格变动率')
        ]

    def get_feature_groups(self):
        """获取特征分组"""
        groups = {
            '价格特征': ['close', 'returns', 'MOM', 'ROC'],
            '均线特征': ['MA5', 'MA10', 'MA20', 'MA60'],
            '技术指标': ['RSI', 'MACD', 'MACD_signal', 'MACD_hist'],
            '波动特征': ['BB_upper', 'BB_middle', 'BB_lower', 'ATR'],
            '成交量特征': ['vol', 'OBV', 'VOL_MA5', 'VOL_MA10']
        }
        return groups

    def get_recommended_features(self, level='basic'):
        """获取推荐的特征组合"""
        recommendations = {
            'basic': ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20'],
            'advanced': ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20', 'RSI', 'MACD', 'BB_middle'],
            'full': [feature[0] for feature in self.available_features]
        }
        return recommendations.get(level, recommendations['basic'])

    def save_feature_config(self, symbol, features, config_name):
        """保存特征配置"""
        config = {
            'symbol': symbol,
            'features': features,
            'config_name': config_name,
            'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        config_file = FEATURE_CONFIG_FILE
        all_configs = self.load_all_configs()
        all_configs.append(config)

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(all_configs, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存特征配置失败: {e}")
            return False

    def load_all_configs(self):
        """加载所有特征配置"""
        if FEATURE_CONFIG_FILE.exists():
            try:
                with open(FEATURE_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []


class TradingStrategy:
    """交易策略基类和具体策略实现"""

    def __init__(self):
        self.strategies = {
            'ma_crossover': '移动平均线交叉',
            'bollinger_band': '布林带策略',
            'rsi_strategy': 'RSI策略',
            'macd_strategy': 'MACD策略',
            'model_based': '模型预测策略'
        }

    def get_strategy_params(self, strategy_name):
        """获取策略的默认参数"""
        params = {
            'ma_crossover': {
                'fast_period': 5,
                'slow_period': 20,
                'stop_loss': 0.05,  # 5%止损
                'take_profit': 0.10  # 10%止盈
            },
            'bollinger_band': {
                'period': 20,
                'devfactor': 2.0,
                'stop_loss': 0.04,
                'take_profit': 0.08
            },
            'rsi_strategy': {
                'period': 14,
                'oversold': 30,
                'overbought': 70,
                'stop_loss': 0.05
            },
            'macd_strategy': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'stop_loss': 0.05
            },
            'model_based': {
                'lookback': 30,
                'forecast_days': 5,
                'confidence_threshold': 0.6,
                'stop_loss': 0.07,
                'take_profit': 0.12
            }
        }
        return params.get(strategy_name, {})

    def calculate_signals(self, df, strategy_name, params=None, model=None):
        """计算交易信号"""
        if params is None:
            params = self.get_strategy_params(strategy_name)

        df = df.copy()
        signals = pd.DataFrame(index=df.index)
        signals['price'] = df['close']
        signals['returns'] = df['returns']

        if strategy_name == 'ma_crossover':
            signals = self._ma_crossover_strategy(df, signals, params)
        elif strategy_name == 'bollinger_band':
            signals = self._bollinger_band_strategy(df, signals, params)
        elif strategy_name == 'rsi_strategy':
            signals = self._rsi_strategy(df, signals, params)
        elif strategy_name == 'macd_strategy':
            signals = self._macd_strategy(df, signals, params)
        elif strategy_name == 'model_based':
            signals = self._model_based_strategy(df, signals, params, model)

        return signals

    def _ma_crossover_strategy(self, df, signals, params):
        """移动平均线交叉策略"""
        fast_period = params.get('fast_period', 5)
        slow_period = params.get('slow_period', 20)

        df[f'MA{fast_period}'] = df['close'].rolling(fast_period).mean()
        df[f'MA{slow_period}'] = df['close'].rolling(slow_period).mean()

        # 金叉买入，死叉卖出
        signals['signal'] = 0
        signals['signal'] = np.where(
            df[f'MA{fast_period}'] > df[f'MA{slow_period}'], 1, 0
        )

        # 生成交易信号（1: 买入, -1: 卖出, 0: 持有）
        signals['position'] = signals['signal'].diff()
        signals.loc[signals['position'] == 0, 'position'] = np.nan

        return signals

    def _bollinger_band_strategy(self, df, signals, params):
        """布林带策略"""
        period = params.get('period', 20)
        devfactor = params.get('devfactor', 2.0)

        # 计算布林带
        df['BB_middle'] = df['close'].rolling(period).mean()
        df['BB_std'] = df['close'].rolling(period).std()
        df['BB_upper'] = df['BB_middle'] + devfactor * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - devfactor * df['BB_std']

        # 价格触及下轨买入，触及上轨卖出
        signals['signal'] = 0
        signals['signal'] = np.where(
            df['close'] < df['BB_lower'], 1,  # 超卖买入
            np.where(df['close'] > df['BB_upper'], -1, 0)  # 超买卖出
        )

        signals['position'] = signals['signal'].diff()
        signals.loc[signals['position'] == 0, 'position'] = np.nan

        return signals

    def _rsi_strategy(self, df, signals, params):
        """RSI策略"""
        period = params.get('period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)

        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # RSI超卖买入，超买卖出
        signals['signal'] = 0
        signals['signal'] = np.where(
            df['RSI'] < oversold, 1,  # 超卖买入
            np.where(df['RSI'] > overbought, -1, 0)  # 超买卖出
        )

        signals['position'] = signals['signal'].diff()
        signals.loc[signals['position'] == 0, 'position'] = np.nan

        return signals

    def _macd_strategy(self, df, signals, params):
        """MACD策略"""
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)

        # 计算MACD
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # MACD金叉买入，死叉卖出
        signals['signal'] = 0
        signals['signal'] = np.where(
            df['MACD'] > df['MACD_signal'], 1, -1
        )

        signals['position'] = signals['signal'].diff()
        signals.loc[signals['position'] == 0, 'position'] = np.nan

        return signals

    def _model_based_strategy(self, df, signals, params, model):
        """基于模型预测的策略"""
        # 如果没有模型，使用简单策略替代
        if model is None:
            # 使用简单的移动平均策略作为替代
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()

            signals['signal'] = 0
            signals['signal'] = np.where(df['MA5'] > df['MA20'], 1, 0)

            signals['position'] = signals['signal'].diff()
            signals.loc[signals['position'] == 0, 'position'] = np.nan

            return signals

        # 如果有模型，尝试使用模型
        try:
            lookback = params.get('lookback', 30)
            forecast_days = params.get('forecast_days', 5)
            confidence_threshold = params.get('confidence_threshold', 0.6)

            # 使用预测模型生成信号
            signals['signal'] = 0
            signals['predicted_returns'] = 0

            # 这里简化处理，实际需要调用预测模型
            # 假设我们有预测的收益率
            if 'returns' in df.columns:
                # 使用简单的移动平均作为预测（实际应使用模型）
                signals['predicted_returns'] = df['returns'].rolling(lookback).mean().shift(-1)

                # 基于预测收益率生成信号
                signals['signal'] = np.where(
                    signals['predicted_returns'] > confidence_threshold, 1,
                    np.where(signals['predicted_returns'] < -confidence_threshold, -1, 0)
                )

            signals['position'] = signals['signal'].diff()
            signals.loc[signals['position'] == 0, 'position'] = np.nan

            return signals

        except Exception as e:
            # 如果模型预测失败，使用简单策略
            print(f"模型预测策略出错，使用备用策略: {e}")

            # 使用简单的移动平均策略
            df['MA5'] = df['close'].rolling(5).mean()
            df['MA20'] = df['close'].rolling(20).mean()

            signals['signal'] = 0
            signals['signal'] = np.where(df['MA5'] > df['MA20'], 1, 0)

            signals['position'] = signals['signal'].diff()
            signals.loc[signals['position'] == 0, 'position'] = np.nan

            return signals
# ==================== 交易策略模块 ====================
# class TradingStrategy:
#     """交易策略基类和具体策略实现"""
#
#     def __init__(self):
#         self.strategies = {
#             'ma_crossover': '移动平均线交叉',
#             'bollinger_band': '布林带策略',
#             'rsi_strategy': 'RSI策略',
#             'macd_strategy': 'MACD策略',
#             'model_based': '模型预测策略'
#         }
#
#     def get_strategy_params(self, strategy_name):
#         """获取策略的默认参数"""
#         params = {
#             'ma_crossover': {
#                 'fast_period': 5,
#                 'slow_period': 20,
#                 'stop_loss': 0.05,  # 5%止损
#                 'take_profit': 0.10  # 10%止盈
#             },
#             'bollinger_band': {
#                 'period': 20,
#                 'devfactor': 2.0,
#                 'stop_loss': 0.04,
#                 'take_profit': 0.08
#             },
#             'rsi_strategy': {
#                 'period': 14,
#                 'oversold': 30,
#                 'overbought': 70,
#                 'stop_loss': 0.05
#             },
#             'macd_strategy': {
#                 'fast_period': 12,
#                 'slow_period': 26,
#                 'signal_period': 9,
#                 'stop_loss': 0.05
#             },
#             'model_based': {
#                 'lookback': 30,
#                 'forecast_days': 5,
#                 'confidence_threshold': 0.6,
#                 'stop_loss': 0.07,
#                 'take_profit': 0.12
#             }
#         }
#         return params.get(strategy_name, {})
#
#     def calculate_signals(self, df, strategy_name, params=None, model=None):
#         """计算交易信号"""
#         if params is None:
#             params = self.get_strategy_params(strategy_name)
#
#         df = df.copy()
#         signals = pd.DataFrame(index=df.index)
#         signals['price'] = df['close']
#         signals['returns'] = df['returns']
#
#         if strategy_name == 'ma_crossover':
#             signals = self._ma_crossover_strategy(df, signals, params)
#         elif strategy_name == 'bollinger_band':
#             signals = self._bollinger_band_strategy(df, signals, params)
#         elif strategy_name == 'rsi_strategy':
#             signals = self._rsi_strategy(df, signals, params)
#         elif strategy_name == 'macd_strategy':
#             signals = self._macd_strategy(df, signals, params)
#         elif strategy_name == 'model_based':
#             signals = self._model_based_strategy(df, signals, params, model)
#
#         return signals
#
#     def _ma_crossover_strategy(self, df, signals, params):
#         """移动平均线交叉策略"""
#         fast_period = params.get('fast_period', 5)
#         slow_period = params.get('slow_period', 20)
#
#         df[f'MA{fast_period}'] = df['close'].rolling(fast_period).mean()
#         df[f'MA{slow_period}'] = df['close'].rolling(slow_period).mean()
#
#         # 金叉买入，死叉卖出
#         signals['signal'] = 0
#         signals['signal'] = np.where(
#             df[f'MA{fast_period}'] > df[f'MA{slow_period}'], 1, 0
#         )
#
#         # 生成交易信号（1: 买入, -1: 卖出, 0: 持有）
#         signals['position'] = signals['signal'].diff()
#         signals.loc[signals['position'] == 0, 'position'] = np.nan
#
#         return signals
#
#     def _bollinger_band_strategy(self, df, signals, params):
#         """布林带策略"""
#         period = params.get('period', 20)
#         devfactor = params.get('devfactor', 2.0)
#
#         # 计算布林带
#         df['BB_middle'] = df['close'].rolling(period).mean()
#         df['BB_std'] = df['close'].rolling(period).std()
#         df['BB_upper'] = df['BB_middle'] + devfactor * df['BB_std']
#         df['BB_lower'] = df['BB_middle'] - devfactor * df['BB_std']
#
#         # 价格触及下轨买入，触及上轨卖出
#         signals['signal'] = 0
#         signals['signal'] = np.where(
#             df['close'] < df['BB_lower'], 1,  # 超卖买入
#             np.where(df['close'] > df['BB_upper'], -1, 0)  # 超买卖出
#         )
#
#         signals['position'] = signals['signal'].diff()
#         signals.loc[signals['position'] == 0, 'position'] = np.nan
#
#         return signals
#
#     def _rsi_strategy(self, df, signals, params):
#         """RSI策略"""
#         period = params.get('period', 14)
#         oversold = params.get('oversold', 30)
#         overbought = params.get('overbought', 70)
#
#         # 计算RSI
#         delta = df['close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(period).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
#         rs = gain / loss
#         df['RSI'] = 100 - (100 / (1 + rs))
#
#         # RSI超卖买入，超买卖出
#         signals['signal'] = 0
#         signals['signal'] = np.where(
#             df['RSI'] < oversold, 1,  # 超卖买入
#             np.where(df['RSI'] > overbought, -1, 0)  # 超买卖出
#         )
#
#         signals['position'] = signals['signal'].diff()
#         signals.loc[signals['position'] == 0, 'position'] = np.nan
#
#         return signals
    #




    # def _macd_strategy(self, df, signals, params):
    #     """MACD策略"""
    #     fast_period = params.get('fast_period', 12)
    #     slow_period = params.get('slow_period', 26)
    #     signal_period = params.get('signal_period', 9)
    #
    #     # 计算MACD
    #     exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
    #     exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
    #     df['MACD'] = exp1 - exp2
    #     df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    #     df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    #
    #     # MACD金叉买入，死叉卖出
    #     signals['signal'] = 0
    #     signals['signal'] = np.where(
    #         df['MACD'] > df['MACD_signal'], 1, -1
    #     )
    #
    #     signals['position'] = signals['signal'].diff()
    #     signals.loc[signals['position'] == 0, 'position'] = np.nan
    #
    #     return signals

    def _macd_strategy(self, df, signals, params):
        """MACD策略 - 修复版"""
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)

        # 计算MACD
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # 改进的信号生成逻辑
        # 1. MACD金叉（上穿信号线）买入
        # 2. MACD死叉（下穿信号线）卖出
        # 3. 增加零轴过滤

        # 生成信号：1=买入，-1=卖出，0=持有
        signals['signal'] = 0

        # 金叉条件：MACD从下向上穿越信号线，且MACD>0
        golden_cross = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift() <= df['MACD_signal'].shift())
        # 死叉条件：MACD从上向下穿越信号线，且MACD<0
        death_cross = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift() >= df['MACD_signal'].shift())

        signals.loc[golden_cross, 'signal'] = 1
        signals.loc[death_cross, 'signal'] = -1

        # 生成交易信号（位置变化）
        signals['position'] = 0
        # 当信号从0变为1时，买入
        signals.loc[(signals['signal'] == 1) & (signals['signal'].shift() != 1), 'position'] = 1
        # 当信号从1变为-1时，卖出
        signals.loc[(signals['signal'] == -1) & (signals['signal'].shift() == 1), 'position'] = -1

        # 确保第一行没有NaN
        signals = signals.fillna(0)

        return signals


    # def _model_based_strategy(self, df, signals, params, model):
    #     """基于模型预测的策略"""
    #     if model is None:
    #         return signals
    #
    #     lookback = params.get('lookback', 30)
    #     forecast_days = params.get('forecast_days', 5)
    #     confidence_threshold = params.get('confidence_threshold', 0.6)
    #
    #     # 使用预测模型生成信号
    #     signals['signal'] = 0
    #     signals['predicted_returns'] = 0
    #
    #     # 这里简化处理，实际需要调用预测模型
    #     # 假设我们有预测的收益率
    #     if 'returns' in df.columns:
    #         # 使用简单的移动平均作为预测（实际应使用模型）
    #         signals['predicted_returns'] = df['returns'].rolling(lookback).mean().shift(-1)
    #
    #         # 基于预测收益率生成信号
    #         signals['signal'] = np.where(
    #             signals['predicted_returns'] > confidence_threshold, 1,
    #             np.where(signals['predicted_returns'] < -confidence_threshold, -1, 0)
    #         )
    #
    #     signals['position'] = signals['signal'].diff()
    #     signals.loc[signals['position'] == 0, 'position'] = np.nan
    #
    #     return signals


    # def _model_based_strategy(self, df, signals, params, model):
    #     """基于模型预测的策略 - 修复版"""
    #     if model is None:
    #         st.warning("⚠️ 没有可用的模型，将使用简单策略替代")
    #         return self._simple_model_strategy(df, signals, params)
    #
    #     lookback = params.get('lookback', 30)
    #     forecast_days = params.get('forecast_days', 5)
    #     confidence_threshold = params.get('confidence_threshold', 0.4)  # 降低阈值
    #     stop_loss = params.get('stop_loss', 0.07)
    #
    #     # 使用预测模型生成信号
    #     signals['signal'] = 0
    #     signals['predicted_returns'] = 0
    #
    #     try:
    #         # 检查是否有训练好的特征列
    #         if hasattr(model, 'feature_columns'):
    #             feature_columns = model.feature_columns
    #         elif hasattr(self, 'feature_columns'):
    #             feature_columns = self.feature_columns
    #         else:
    #             # 使用默认特征
    #             feature_columns = ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20']
    #
    #         # 为每行数据生成预测
    #         for i in range(lookback, len(df)):
    #             # 获取最近lookback天的数据
    #             recent_data = df[feature_columns].iloc[i - lookback:i].values
    #
    #             # 标准化数据（如果有scaler）
    #             if hasattr(self, 'scalers') and 'X' in self.scalers:
    #                 recent_data_scaled = self.scalers['X'].transform(
    #                     recent_data.reshape(-1, len(feature_columns))
    #                 ).reshape(1, lookback, len(feature_columns))
    #             else:
    #                 recent_data_scaled = recent_data.reshape(1, lookback, len(feature_columns))
    #
    #             # 预测未来收益率
    #             if hasattr(model, 'predict'):
    #                 prediction = model.predict(recent_data_scaled)
    #                 predicted_return = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
    #             else:
    #                 # 备用预测方法
    #                 predicted_return = df['returns'].iloc[i - lookback:i].mean()
    #
    #             signals.at[signals.index[i], 'predicted_returns'] = predicted_return
    #
    #             # 基于预测收益率生成信号
    #             if predicted_return > confidence_threshold:
    #                 signals.at[signals.index[i], 'signal'] = 1
    #             elif predicted_return < -confidence_threshold:
    #                 signals.at[signals.index[i], 'signal'] = -1
    #             else:
    #                 signals.at[signals.index[i], 'signal'] = 0
    #
    #     except Exception as e:
    #         st.warning(f"模型预测出错，使用备用策略: {e}")
    #         return self._simple_model_strategy(df, signals, params)
    #
    #     # 生成交易信号
    #     signals['position'] = 0
    #     signals.loc[(signals['signal'] == 1) & (signals['signal'].shift() != 1), 'position'] = 1
    #     signals.loc[(signals['signal'] == -1) & (signals['signal'].shift() == 1), 'position'] = -1
    #
    #     return signals
    #
    #
    # def _simple_model_strategy(self, df, signals, params):
    #     """简化模型策略（备用）"""
    #     # 使用简单的技术指标组合
    #     lookback = params.get('lookback', 20)
    #
    #     # 计算多个指标
    #     df['MA5'] = df['close'].rolling(5).mean()
    #     df['MA20'] = df['close'].rolling(20).mean()
    #     df['RSI'] = 50  # 简化，实际应计算RSI
    #
    #     # 复合信号
    #     ma_signal = (df['MA5'] > df['MA20']).astype(int)
    #     volume_signal = (df['vol'] > df['vol'].rolling(20).mean()).astype(int)
    #
    #     signals['signal'] = 0
    #     signals.loc[(ma_signal == 1) & (volume_signal == 1), 'signal'] = 1
    #     signals.loc[(ma_signal == 0) & (volume_signal == 0), 'signal'] = -1
    #
    #     signals['position'] = signals['signal'].diff()
    #     signals.loc[signals['position'] == 0, 'position'] = np.nan
    #
    #     return signals




# ==================== 回测引擎模块 ====================
class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital=100000.0, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # 交易佣金

    def run_backtest(self, df, signals, strategy_name, params=None):
        """运行回测"""
        if params is None:
            params = {}

        # 确保数据正确排序
        df = df.sort_values('trade_date').copy()
        signals = signals.loc[df.index].copy()

        # 初始化回测变量
        capital = self.initial_capital
        position = 0  # 持仓数量
        trades = []  # 交易记录
        portfolio_values = [capital]  # 投资组合价值历史
        dates = [df['trade_date'].iloc[0]]  # 日期历史

        stop_loss = params.get('stop_loss', 0.05)
        take_profit = params.get('take_profit', 0.10)

        buy_price = None  # 买入价格（用于计算止损止盈）

        for i in range(1, len(df)):
            current_date = df['trade_date'].iloc[i]
            current_price = df['close'].iloc[i]
            signal = signals['position'].iloc[i] if not pd.isna(signals['position'].iloc[i]) else 0

            # 检查止损止盈
            if position > 0 and buy_price is not None:
                price_change = (current_price - buy_price) / buy_price

                # 止损
                if price_change < -stop_loss:
                    signal = -1  # 强制卖出
                # 止盈
                elif price_change > take_profit:
                    signal = -1  # 止盈卖出

            # 执行交易信号
            if signal == 1 and position == 0:  # 买入信号，空仓时买入
                # 计算可买数量（考虑佣金）
                available_shares = (capital * (1 - self.commission)) / current_price
                position = available_shares
                capital = 0
                buy_price = current_price

                trades.append({
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'shares': position,
                    'value': position * current_price
                })

            elif signal == -1 and position > 0:  # 卖出信号，有持仓时卖出
                sell_value = position * current_price * (1 - self.commission)
                capital = sell_value

                trades.append({
                    'date': current_date,
                    'type': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': sell_value,
                    'profit': sell_value - (position * buy_price) if buy_price else 0
                })

                position = 0
                buy_price = None

            # 计算当前投资组合价值
            current_value = capital + (position * current_price)
            portfolio_values.append(current_value)
            dates.append(current_date)

        # 计算回测结果
        results = self._calculate_backtest_results(
            portfolio_values, dates, trades, df, signals
        )

        return results, trades, portfolio_values, dates

    def _calculate_backtest_results(self, portfolio_values, dates, trades, df, signals):
        """计算回测结果指标"""
        results = {}

        # 基本指标
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value

        # 计算每日收益率
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()

        # 年化收益率
        if len(daily_returns) > 0:
            years = len(df) / 252  # 假设一年252个交易日
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            # 年化波动率
            annual_volatility = daily_returns.std() * np.sqrt(252)

            # 夏普比率（假设无风险利率为3%）
            risk_free_rate = 0.03
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

            # 最大回撤
            cumulative = pd.Series(portfolio_values)
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # 交易统计
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']

            # 胜率（盈利交易比例）
            profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
            win_rate = len(profitable_trades) / len(sell_trades) if len(sell_trades) > 0 else 0

            # 平均盈亏比
            if len(profitable_trades) > 0 and len(sell_trades) > len(profitable_trades):
                avg_profit = np.mean([t.get('profit', 0) for t in profitable_trades])
                avg_loss = np.mean([t.get('profit', 0) for t in sell_trades if t.get('profit', 0) <= 0])
                profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
            else:
                profit_loss_ratio = 0

            results = {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'win_rate': win_rate,
                'avg_profit_loss_ratio': profit_loss_ratio,
                'avg_daily_return': daily_returns.mean(),
                'std_daily_return': daily_returns.std()
            }

        return results


# ==================== Streamlit应用（增强版）====================
def main():
    # 页面配置
    st.set_page_config(
        page_title="量化分析系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 量化分析系统")
    st.markdown("股票数据获取、分析与预测一体化平台")

    # 初始化
    if 'fetcher' not in st.session_state:
        st.session_state.fetcher = DataFetcher()
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StockPredictor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = DataVisualizer()
    if 'feature_selector' not in st.session_state:
        st.session_state.feature_selector = FeatureSelector()

    model_to_use = None  # 确保这个变量在任何地方都被定义

    # 添加交易策略和回测引擎
    if 'trading_strategy' not in st.session_state:
        st.session_state.trading_strategy = TradingStrategy()
    if 'backtest_engine' not in st.session_state:
        st.session_state.backtest_engine = BacktestEngine()

    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = {}
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_model_type' not in st.session_state:
        st.session_state.current_model_type = None
    if 'current_model_info' not in st.session_state:
        st.session_state.current_model_info = None
    if 'trained_models_history' not in st.session_state:
        st.session_state.trained_models_history = []
    if 'last_trained_model' not in st.session_state:
        st.session_state.last_trained_model = None

    fetcher = st.session_state.fetcher
    predictor = st.session_state.predictor
    visualizer = st.session_state.visualizer
    feature_selector = st.session_state.feature_selector
    trading_strategy = st.session_state.trading_strategy
    backtest_engine = st.session_state.backtest_engine

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")
        # 数据源状态
        st.markdown(f"**数据源**: {'✅ Tushare已连接' if fetcher.pro else '❌ Tushare未连接'}")

        # 股票选择
        st.subheader("股票选择")
        symbols_input = st.text_input(
            "股票代码 (逗号分隔)",
            value="000001.SZ,600000.SH",
            help="示例: 000001.SZ, 600000.SH"
        )
        symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]

        # 日期选择
        st.subheader("日期范围")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=365)
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # 复权方式
        st.subheader("复权方式")
        adj_method = st.selectbox(
            "选择复权方式",
            ["none", "qfq", "hfq"],
            format_func=lambda x: {
                "none": "不复权",
                "qfq": "前复权",
                "hfq": "后复权"
            }[x],
            help="选择股票数据的复权方式"
        )

        # 文件格式
        st.subheader("导出设置")
        file_format = st.radio("格式", ["CSV", "Excel"], horizontal=True)

        # 当前模型状态
        st.markdown("---")
        st.subheader("当前模型")
        if st.session_state.current_model:
            model_type_display = {
                'random_forest': '随机森林',
                'xgboost': 'XGBoost',
                'lstm': 'LSTM',
                'lstm_attention': 'LSTM with Attention'
            }
            display_type = model_type_display.get(st.session_state.current_model_type,
                                                  st.session_state.current_model_type)
            st.success(f"✅ 已加载: {display_type}")
            if st.session_state.current_model_info:
                st.caption(f"股票: {st.session_state.current_model_info.get('symbol', '未知')}")
                st.caption(f"特征数: {len(st.session_state.current_model_info.get('feature_columns', []))}")
                st.caption(f"保存时间: {st.session_state.current_model_info.get('save_time', '未知')}")
        else:
            st.info("未加载模型")

        # 快速加载最近训练的模型
        if st.session_state.trained_models_history:
            st.subheader("最近训练模型")
            for i, model_info in enumerate(st.session_state.trained_models_history[-3:]):
                if st.button(f"📥 加载 {model_info.get('name', '模型')}", key=f"load_recent_{i}"):
                    model_path = model_info.get('path')
                    if model_path and model_path.exists():
                        model, model_type, model_info = predictor.load_model(model_path)
                        if model:
                            st.session_state.current_model = model
                            st.session_state.current_model_type = model_type
                            st.session_state.current_model_info = model_info
                            st.success("✅ 模型加载成功！")
                            st.rerun()

        st.markdown("---")

    # 主界面 - 标签页
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📥 数据抓取", "📊 数据显示", "💾 数据存储", "🔄 数据更新",
        "🤖 模型训练", "✅ 模型验证", "📈 交易策略与回测"
    ])

    # 标签页1: 数据抓取
    with tab1:
        st.header("股票数据抓取")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🚀 抓取数据", type="primary", use_container_width=True):
                if not symbols:
                    st.warning("请输入股票代码")
                else:
                    progress_bar = st.progress(0)
                    results = {}
                    for i, symbol in enumerate(symbols):
                        progress = (i + 1) / len(symbols)
                        progress_bar.progress(progress)

                        # 使用选择的复权方式
                        df = fetcher.fetch_stock_data(symbol, start_str, end_str, adj=adj_method)
                        if df is not None and not df.empty:
                            # 存储到session_state
                            st.session_state.data_cache[symbol] = df
                            results[symbol] = df
                            st.success(f"✅ {symbol} 数据抓取成功 ({len(df)} 条)")
                        else:
                            st.error(f"❌ {symbol} 数据抓取失败")

                    progress_bar.empty()
                    if results:
                        st.success(f"✅ 完成! 成功抓取 {len(results)}/{len(symbols)} 只股票数据")

        with col2:
            if st.button("🗑️ 清空缓存", use_container_width=True):
                st.session_state.data_cache = {}
                st.success("缓存已清空")

        # 显示已抓取的数据概览
        if st.session_state.data_cache:
            st.subheader("已抓取数据概览")
            cache_df = pd.DataFrame([{
                '股票代码': symbol,
                '数据条数': len(df),
                '开始日期': df['trade_date'].min().strftime('%Y-%m-%d'),
                '结束日期': df['trade_date'].max().strftime('%Y-%m-%d'),
                '最新收盘价': df['close'].iloc[-1] if 'close' in df.columns else None,
                '复权方式': adj_method
            } for symbol, df in st.session_state.data_cache.items()])
            st.dataframe(cache_df, use_container_width=True)

    # 标签页2: 数据显示
    with tab2:
        st.header("股票数据显示")
        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取数据")
        else:
            selected_symbol = st.selectbox(
                "选择要显示的股票",
                list(st.session_state.data_cache.keys()),
                key="display_select"
            )
            if selected_symbol:
                df = st.session_state.data_cache[selected_symbol]
                # 基本信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("数据条数", len(df))
                with col2:
                    st.metric("开始日期", df['trade_date'].min().strftime('%Y-%m-%d'))
                with col3:
                    st.metric("结束日期", df['trade_date'].max().strftime('%Y-%m-%d'))
                with col4:
                    if 'close' in df.columns:
                        latest_close = df['close'].iloc[-1]
                        prev_close = df['close'].iloc[-2] if len(df) > 1 else latest_close
                        change_pct = ((latest_close - prev_close) / prev_close * 100) if len(df) > 1 else 0
                        st.metric("最新收盘价", f"{latest_close:.2f}", f"{change_pct:.2f}%")

                # 显示模式
                display_mode = st.radio(
                    "显示模式",
                    ["数据表格", "可视化图表", "技术指标"],
                    horizontal=True
                )

                if display_mode == "数据表格":
                    # 列选择
                    available_columns = df.columns.tolist()
                    selected_columns = st.multiselect(
                        "选择显示的列",
                        available_columns,
                        default=['trade_date', 'open', 'high', 'low', 'close', 'vol', 'pct_chg']
                    )
                    if selected_columns:
                        display_df = df[selected_columns].sort_values('trade_date', ascending=False)
                        st.dataframe(display_df, use_container_width=True)

                elif display_mode == "可视化图表":
                    fig = visualizer.plot_candlestick(df, selected_symbol)
                    st.plotly_chart(fig, use_container_width=True)

                elif display_mode == "技术指标":
                    tab_rsi, tab_macd, tab_bb = st.tabs(["RSI", "MACD", "布林带"])

                    with tab_rsi:
                        if 'RSI' in df.columns:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=df['trade_date'], y=df['RSI'], mode='lines', name='RSI'))
                            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
                            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
                            fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="中线")
                            fig_rsi.update_layout(title="RSI相对强弱指标", height=400)
                            st.plotly_chart(fig_rsi, use_container_width=True)

                    with tab_macd:
                        if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                            fig_macd = go.Figure()
                            fig_macd.add_trace(go.Scatter(x=df['trade_date'], y=df['MACD'], mode='lines', name='MACD'))
                            fig_macd.add_trace(
                                go.Scatter(x=df['trade_date'], y=df['MACD_signal'], mode='lines', name='信号线'))

                            # MACD柱状图
                            if 'MACD_hist' in df.columns:
                                colors = ['green' if x >= 0 else 'red' for x in df['MACD_hist']]
                                fig_macd.add_trace(go.Bar(
                                    x=df['trade_date'],
                                    y=df['MACD_hist'],
                                    name='MACD柱',
                                    marker_color=colors,
                                    opacity=0.5
                                ))

                            fig_macd.update_layout(title="MACD指标", height=400)
                            st.plotly_chart(fig_macd, use_container_width=True)

                    with tab_bb:
                        if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
                            fig_bb = go.Figure()

                            # 布林带区域
                            fig_bb.add_trace(go.Scatter(
                                x=df['trade_date'].tolist() + df['trade_date'].tolist()[::-1],
                                y=df['BB_upper'].tolist() + df['BB_lower'].tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(0,100,80,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='布林带',
                                showlegend=True
                            ))

                            # 三条线
                            fig_bb.add_trace(go.Scatter(x=df['trade_date'], y=df['BB_upper'],
                                                        mode='lines', name='上轨', line=dict(color='red', width=1)))
                            fig_bb.add_trace(go.Scatter(x=df['trade_date'], y=df['BB_middle'],
                                                        mode='lines', name='中轨', line=dict(color='blue', width=1)))
                            fig_bb.add_trace(go.Scatter(x=df['trade_date'], y=df['BB_lower'],
                                                        mode='lines', name='下轨', line=dict(color='green', width=1)))

                            # 收盘价
                            fig_bb.add_trace(go.Scatter(x=df['trade_date'], y=df['close'],
                                                        mode='lines', name='收盘价', line=dict(color='black', width=2)))

                            fig_bb.update_layout(title="布林带指标", height=400)
                            st.plotly_chart(fig_bb, use_container_width=True)

    # 标签页3: 数据存储
    with tab3:
        st.header("股票数据存储")
        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取数据")
        else:
            st.subheader("选择要存储的数据")
            selected_symbols = st.multiselect(
                "选择股票",
                list(st.session_state.data_cache.keys()),
                default=list(st.session_state.data_cache.keys())[:1]
            )
            if selected_symbols:
                # 存储选项
                col1, col2 = st.columns(2)
                with col1:
                    storage_format = st.radio("存储格式", ["CSV", "Excel"], horizontal=True)
                with col2:
                    combine_files = st.checkbox("合并为一个文件", value=False)
                if st.button("💾 存储数据", type="primary"):
                    saved_files = []
                    if combine_files:
                        # 合并所有选中的股票数据
                        combined_dfs = []
                        for symbol in selected_symbols:
                            df = st.session_state.data_cache[symbol].copy()
                            df['股票代码'] = symbol
                            combined_dfs.append(df)
                        if combined_dfs:
                            combined_df = pd.concat(combined_dfs, ignore_index=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            if storage_format == "CSV":
                                filename = f"stocks_combined_{timestamp}.csv"
                                filepath = DOWNLOAD_DIR / filename
                                combined_df.to_csv(filepath, index=False, encoding='utf-8-sig')
                            else:
                                filename = f"stocks_combined_{timestamp}.xlsx"
                                filepath = DOWNLOAD_DIR / filename
                                combined_df.to_excel(filepath, index=False)
                            saved_files.append((filename, filepath))
                            st.success(f"✅ 数据已合并保存到: {filename}")
                    else:
                        # 分别保存每个股票数据
                        for symbol in selected_symbols:
                            df = st.session_state.data_cache[symbol]
                            filepath = fetcher.save_to_file(df, symbol, storage_format.lower())
                            if filepath:
                                saved_files.append((filepath.name, filepath))
                                st.success(f"✅ {symbol} 数据已保存到: {filepath.name}")
                    # 提供下载按钮
                    if saved_files:
                        st.subheader("下载文件")
                        for filename, filepath in saved_files:
                            with open(filepath, 'rb') as f:
                                file_data = f.read()
                            mime_type = "text/csv" if storage_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            st.download_button(
                                label=f"📥 下载 {filename}",
                                data=file_data,
                                file_name=filename,
                                mime=mime_type,
                                key=f"download_{filename}"
                            )
                # 显示已存储的文件
                st.subheader("已存储文件")
                stored_files = list(DOWNLOAD_DIR.glob("*.*"))
                stored_files = [f for f in stored_files if f.suffix in ['.csv', '.xlsx']]
                if stored_files:
                    for file in stored_files[-10:]:  # 显示最近10个文件
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"📄 {file.name}")
                        with col2:
                            with open(file, 'rb') as f:
                                st.download_button(
                                    label="下载",
                                    data=f,
                                    file_name=file.name,
                                    mime="text/csv" if file.suffix == '.csv' else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"dl_{file.name}"
                                )
                        with col3:
                            if st.button("删除", key=f"del_{file.name}"):
                                file.unlink()
                                st.success("已删除")
                                st.rerun()
                else:
                    st.info("暂无已存储文件")

    # 标签页4: 数据更新
    with tab4:
        st.header("股票数据更新")
        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取数据")
        else:
            selected_symbol = st.selectbox(
                "选择要更新的股票",
                list(st.session_state.data_cache.keys()),
                key="update_select"
            )
            if selected_symbol:
                current_df = st.session_state.data_cache[selected_symbol]
                last_date = current_df['trade_date'].max()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("当前数据截止", last_date.strftime('%Y-%m-%d'))
                with col2:
                    st.metric("当前数据条数", len(current_df))
                with col3:
                    days_since_last = (datetime.now().date() - last_date.date()).days
                    st.metric("距今天数", days_since_last)
                # 更新选项
                st.subheader("更新设置")
                update_mode = st.radio(
                    "更新模式",
                    ["增量更新（从最后日期开始）", "重新抓取（完整更新）"],
                    horizontal=True
                )
                if update_mode == "增量更新（从最后日期开始）":
                    update_start_date = last_date + timedelta(days=1)
                    update_start_str = update_start_date.strftime("%Y%m%d")
                    st.info(f"将从 {update_start_date.strftime('%Y-%m-%d')} 开始更新数据")
                else:
                    update_start_str = start_str
                    st.warning("⚠️ 将重新抓取完整数据，原有数据将被覆盖")
                # 确认更新
                if st.button("🔄 开始更新", type="primary"):
                    with st.spinner("正在更新数据..."):
                        # 获取更新数据
                        new_df = fetcher.fetch_stock_data(
                            selected_symbol,
                            update_start_str,
                            end_str
                        )
                        if new_df is not None and not new_df.empty:
                            if update_mode == "增量更新（从最后日期开始）":
                                # 合并新旧数据，去重
                                combined_df = pd.concat([current_df, new_df], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=['trade_date'], keep='last')
                                combined_df = combined_df.sort_values('trade_date').reset_index(drop=True)
                                st.session_state.data_cache[selected_symbol] = combined_df
                                st.success(
                                    f"✅ 数据已更新！新增 {len(new_df)} 条记录，总记录数：{len(combined_df)}")
                            else:
                                # 替换为完整数据
                                st.session_state.data_cache[selected_symbol] = new_df
                                st.success(f"✅ 数据已重新抓取！总记录数：{len(new_df)}")
                            # 显示更新前后的对比
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("更新前")
                                st.dataframe(current_df[['trade_date', 'close']].tail(5))
                            with col2:
                                st.subheader("更新后")
                                updated_df = st.session_state.data_cache[selected_symbol]
                                st.dataframe(updated_df[['trade_date', 'close']].tail(5))
                            # 显示更新统计
                            if update_mode == "增量更新（从最后日期开始）":
                                added_dates = new_df['trade_date'].dt.strftime('%Y-%m-%d').tolist()
                                if added_dates:
                                    st.info(f"📅 新增日期: {', '.join(added_dates)}")
                                else:
                                    st.warning(f"⚠️ 没有获取到新的数据，可能已是最新")
                # 批量更新选项
                st.subheader("批量更新")
                symbols_to_update = st.multiselect(
                    "选择要批量更新的股票",
                    list(st.session_state.data_cache.keys()),
                    default=[selected_symbol]
                )
                if symbols_to_update and st.button("🔄 批量更新选中股票"):
                    progress_bar = st.progress(0)
                    update_results = []
                    for i, symbol in enumerate(symbols_to_update):
                        progress = (i + 1) / len(symbols_to_update)
                        progress_bar.progress(progress)
                        current_data = st.session_state.data_cache[symbol]
                        last_date = current_data['trade_date'].max()
                        update_start = last_date + timedelta(days=1)
                        update_start_str = update_start.strftime("%Y%m%d")
                        new_data = fetcher.fetch_stock_data(symbol, update_start_str, end_str)
                        if new_data is not None and not new_data.empty:
                            # 合并数据
                            combined_df = pd.concat([current_data, new_data], ignore_index=True)
                            combined_df = combined_df.drop_duplicates(subset=['trade_date'], keep='last')
                            combined_df = combined_df.sort_values('trade_date').reset_index(drop=True)
                            st.session_state.data_cache[symbol] = combined_df
                            update_results.append(f"✅ {symbol}: 新增 {len(new_data)} 条")
                        else:
                            update_results.append(f"⚠️ {symbol}: 无新数据")
                    progress_bar.empty()
                    if update_results:
                        st.subheader("更新结果")
                        for result in update_results:
                            st.write(result)

    # 标签页5: 模型训练
    with tab5:
        st.header("模型训练")

        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取数据")
        else:
            # 选择股票
            train_symbol = st.selectbox(
                "选择训练股票",
                list(st.session_state.data_cache.keys()),
                key="train_select"
            )

            if train_symbol:
                df = st.session_state.data_cache[train_symbol]

                if len(df) >= 100:
                    # 训练配置
                    st.subheader("训练配置")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        lookback = st.slider("回看天数", 20, 100, 30)
                    with col2:
                        forecast_days = st.slider("预测天数", 1, 10, 5)
                    with col3:
                        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, step=0.05)

                    # 特征选择
                    st.subheader("特征选择")

                    # 获取数据中的可用特征
                    available_features_in_data = []
                    for feature_tuple in feature_selector.available_features:
                        feature_name = feature_tuple[0]
                        if feature_name in df.columns:
                            available_features_in_data.append(feature_tuple)

                    if not available_features_in_data:
                        st.warning("数据中没有可用的技术指标特征，请检查数据获取")
                        default_features = ['close', 'vol', 'returns']
                    else:
                        # 快速选择按钮
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("基础特征", use_container_width=True):
                                st.session_state.selected_features = feature_selector.get_recommended_features('basic')
                                st.rerun()
                        with col2:
                            if st.button("高级特征", use_container_width=True):
                                st.session_state.selected_features = feature_selector.get_recommended_features(
                                    'advanced')
                                st.rerun()
                        with col3:
                            if st.button("全选特征", use_container_width=True):
                                st.session_state.selected_features = [f[0] for f in available_features_in_data]
                                st.rerun()

                        # 初始化选中的特征
                        if 'selected_features' not in st.session_state:
                            st.session_state.selected_features = feature_selector.get_recommended_features('basic')

                        # 多选框选择特征
                        selected_features = st.multiselect(
                            "选择特征（可多选）",
                            options=[f"{f[1]} ({f[0]})" for f in available_features_in_data],
                            default=[f"{f[1]} ({f[0]})" for f in available_features_in_data
                                     if f[0] in st.session_state.selected_features],
                            help="选择用于模型训练的特征"
                        )

                        # 提取特征名
                        selected_feature_names = []
                        for feature_display in selected_features:
                            # 从显示文本中提取特征名（括号内的部分）
                            import re
                            match = re.search(r'\((.*?)\)', feature_display)
                            if match:
                                selected_feature_names.append(match.group(1))

                        # 如果没有选择任何特征，使用默认特征
                        if not selected_feature_names:
                            selected_feature_names = ['close', 'vol', 'returns']

                    # 模型选择
                    st.subheader("模型选择")
                    model_type_display = st.selectbox(
                        "选择模型类型",
                        ["随机森林", "XGBoost", "LSTM", "LSTM with Attention"],
                        help="选择要使用的预测模型"
                    )

                    # 模型类型映射
                    model_type_map = {
                        "随机森林": "random_forest",
                        "XGBoost": "xgboost",
                        "LSTM": "lstm",
                        "LSTM with Attention": "lstm_attention"
                    }
                    model_type = model_type_map[model_type_display]

                    # 加载现有模型
                    st.subheader("加载现有模型")
                    load_existing = st.checkbox("加载现有模型继续训练")

                    loaded_model = None
                    loaded_model_info = None

                    if load_existing:
                        saved_models = predictor.get_saved_models()
                        if saved_models:
                            model_options = [f"{m['name']} ({m['symbol']}, {m['model_type']})" for m in saved_models]
                            selected_model_display = st.selectbox("选择模型", model_options)

                            if selected_model_display and st.button("📥 加载模型"):
                                selected_index = model_options.index(selected_model_display)
                                model_path = saved_models[selected_index]['path']

                                with st.spinner("正在加载模型..."):
                                    model, loaded_model_type, model_info = predictor.load_model(model_path)
                                    if model:
                                        loaded_model = model
                                        loaded_model_info = model_info
                                        st.success(f"✅ 模型加载成功: {model_info.get('symbol', '未知')}")
                                        # 自动设置特征选择
                                        if 'feature_columns' in model_info:
                                            st.session_state.selected_features = model_info['feature_columns']
                                            st.rerun()
                                    else:
                                        st.error("❌ 模型加载失败")
                        else:
                            st.info("没有找到保存的模型")

                    # 训练参数
                    with st.expander("高级参数配置", expanded=False):
                        params = {}

                        if model_type in ["random_forest", "xgboost"]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                params['n_estimators'] = st.number_input("树数量", 50, 500, 100, step=10)
                            with col2:
                                params['max_depth'] = st.number_input("最大深度", 3, 20, 10)
                            with col3:
                                if model_type == "random_forest":
                                    params['min_samples_split'] = st.number_input("最小分割样本数", 2, 20, 2)
                                elif model_type == "xgboost":
                                    params['learning_rate'] = st.number_input("学习率", 0.01, 0.5, 0.1, step=0.01)

                        elif model_type in ["lstm", "lstm_attention"]:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                params['lstm_units'] = st.number_input("LSTM单元数", 16, 256, 50)
                            with col2:
                                params['dropout_rate'] = st.number_input("Dropout率", 0.1, 0.5, 0.2, step=0.05)
                            with col3:
                                params['epochs'] = st.number_input("训练轮数", 10, 200, 50)
                            with col4:
                                params['batch_size'] = st.selectbox("批大小", [16, 32, 64, 128], index=1)

                            if load_existing and loaded_model:
                                params['learning_rate'] = st.number_input("学习率", 0.0001, 0.1, 0.001, step=0.0001,
                                                                          format="%.4f")
                                params['patience'] = st.number_input("早停耐心值", 5, 30, 10)

                    # 开始训练按钮
                    if st.button("🚀 开始训练", type="primary", use_container_width=True):
                        with st.spinner("正在准备数据和训练模型..."):
                            try:
                                # 准备数据
                                X_train, y_train, X_test, y_test = predictor.prepare_features(
                                    df, lookback, forecast_days, selected_feature_names
                                )

                                if X_train is None:
                                    st.error("数据不足，请增加数据量或减少回看天数")
                                else:
                                    st.info(f"✅ 数据准备完成: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
                                    st.info(f"✅ 使用特征: {', '.join(predictor.feature_columns)}")

                                    model = None
                                    history = None

                                    # 训练模型
                                    if load_existing and loaded_model:
                                        st.info("正在继续训练现有模型...")
                                        if model_type in ["lstm", "lstm_attention"]:
                                            model, history = predictor.continue_training(loaded_model, X_train, y_train,
                                                                                         **params)
                                            st.success("✅ 模型继续训练完成!")
                                        else:
                                            st.warning("该模型类型不支持继续训练，将重新训练")
                                            load_existing = False

                                    if not load_existing or not loaded_model:
                                        st.info(f"正在训练{model_type_display}模型...")

                                        if model_type == "random_forest":
                                            model = predictor.train_random_forest(X_train, y_train, **params)
                                            history = None
                                        elif model_type == "xgboost":
                                            model = predictor.train_xgboost(X_train, y_train, **params)
                                            history = None
                                        elif model_type == "lstm":
                                            model, history = predictor.train_lstm(X_train, y_train, **params)
                                        elif model_type == "lstm_attention":
                                            model, history = predictor.train_lstm_attention(X_train, y_train, **params)

                                        st.success("✅ 模型训练完成!")

                                    if model:
                                        # 评估模型
                                        st.info("正在评估模型性能...")
                                        results = predictor.evaluate_model(model, X_test, y_test, model_type)

                                        # 显示性能指标
                                        st.success("✅ 模型评估完成!")

                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("RMSE", f"{results['rmse']:.4f}")
                                        with col2:
                                            st.metric("MAE", f"{results['mae']:.4f}")
                                        with col3:
                                            st.metric("R²", f"{results['r2']:.4f}")
                                        with col4:
                                            st.metric("方向准确率", f"{results['direction_accuracy']:.2%}")

                                        # 训练历史图（如果可用）
                                        if history:
                                            fig_history = visualizer.plot_training_history(history, model_type_display)
                                            if fig_history:
                                                st.plotly_chart(fig_history, use_container_width=True)

                                        # 保存模型到session state
                                        model_info = {
                                            'model': model,
                                            'model_type': model_type,
                                            'symbol': train_symbol,
                                            'feature_columns': predictor.feature_columns,
                                            'params': params,
                                            'lookback': lookback,
                                            'forecast_days': forecast_days,
                                            'results': results
                                        }

                                        st.session_state.current_model = model
                                        st.session_state.current_model_type = model_type
                                        st.session_state.current_model_info = model_info
                                        st.session_state.last_trained_model = model_info

                                        st.success("✅ 模型已保存到当前会话!")

                                        # 提供保存到文件的选项
                                        st.subheader("保存模型到文件")
                                        save_to_file = st.checkbox("保存模型到文件", value=True)

                                        if save_to_file:
                                            model_name = st.text_input(
                                                "模型名称",
                                                value=f"{train_symbol}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                            )

                                            if st.button("💾 保存到文件"):
                                                model_path = predictor.save_model(
                                                    model, model_type, train_symbol,
                                                    predictor.feature_columns, params
                                                )
                                                if model_path:
                                                    # 添加到训练历史
                                                    model_file_info = {
                                                        'path': model_path,
                                                        'name': model_path.stem,
                                                        'symbol': train_symbol,
                                                        'model_type': model_type,
                                                        'save_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        'results': results
                                                    }
                                                    st.session_state.trained_models_history.append(model_file_info)
                                                    st.success(f"✅ 模型已保存到文件: {model_path.name}")

                                        # 显示预测结果
                                        st.subheader("测试集预测结果")

                                        # 获取对应的日期
                                        test_start_idx = int(len(df) * 0.8) - lookback
                                        if test_start_idx < 0:
                                            test_start_idx = 0

                                        # 确保有足够的数据
                                        num_predictions = len(results['actual'])
                                        if num_predictions > 0:
                                            # 获取日期
                                            test_dates = []
                                            for i in range(num_predictions):
                                                date_idx = test_start_idx + lookback + i * forecast_days
                                                if date_idx < len(df):
                                                    test_dates.append(df['trade_date'].iloc[date_idx])
                                                else:
                                                    # 如果没有足够的日期，使用最后一个日期加上偏移
                                                    last_date = df['trade_date'].iloc[-1]
                                                    offset = date_idx - len(df) + 1
                                                    if isinstance(last_date, pd.Timestamp):
                                                        new_date = last_date + timedelta(days=offset)
                                                    else:
                                                        new_date = pd.to_datetime(last_date) + timedelta(days=offset)
                                                    test_dates.append(new_date)

                                            # 限制显示数量
                                            display_limit = min(20, num_predictions)

                                            fig_results = visualizer.plot_validation_results(
                                                results['actual'][:display_limit],
                                                results['predictions'][:display_limit],
                                                test_dates[:display_limit],
                                                results,
                                                model_type_display
                                            )
                                            st.plotly_chart(fig_results, use_container_width=True)

                                        # 预测未来价格
                                        st.subheader("未来价格预测")

                                        if 'close' in df.columns:
                                            # 获取最新数据
                                            latest_features = df[predictor.feature_columns].iloc[-lookback:].values

                                            # 标准化
                                            if hasattr(predictor, 'scalers') and 'X' in predictor.scalers:
                                                latest_features_scaled = predictor.scalers['X'].transform(
                                                    latest_features.reshape(-1, len(predictor.feature_columns))
                                                ).reshape(1, lookback, len(predictor.feature_columns))
                                            else:
                                                latest_features_scaled = latest_features.reshape(1, lookback,
                                                                                                 len(predictor.feature_columns))

                                            # 预测未来
                                            future_predictions = predictor.predict_future(
                                                model, latest_features_scaled,
                                                forecast_days, model_type
                                            )

                                            # 显示预测结果
                                            last_price = df['close'].iloc[-1]
                                            last_date = df['trade_date'].iloc[-1]

                                            if isinstance(last_date, str):
                                                last_date = pd.to_datetime(last_date)

                                            pred_dates = [last_date + timedelta(days=i + 1) for i in
                                                          range(forecast_days)]

                                            pred_df = pd.DataFrame({
                                                '日期': [d.strftime('%Y-%m-%d') for d in pred_dates],
                                                '预测价格': np.round(future_predictions, 4),
                                                '变化量': np.round(future_predictions - last_price, 4),
                                                '变化率%': np.round(
                                                    (future_predictions - last_price) / last_price * 100, 2)
                                            })

                                            # 格式化显示
                                            st.dataframe(pred_df.style.format({
                                                '预测价格': '{:.4f}',
                                                '变化量': '{:.4f}',
                                                '变化率%': '{:.2f}%'
                                            }).apply(
                                                lambda x: [
                                                    'background-color: lightgreen' if v > 0 else 'background-color: lightcoral'
                                                    for v in x],
                                                subset=['变化率%']
                                            ))

                                            # 绘制未来预测图
                                            fig_future = go.Figure()

                                            # 历史价格（最近60天）
                                            history_days = min(60, len(df))
                                            history_prices = df['close'].values[-history_days:]
                                            history_dates = df['trade_date'].values[-history_days:]

                                            fig_future.add_trace(go.Scatter(
                                                x=history_dates,
                                                y=history_prices,
                                                mode='lines',
                                                name='历史价格',
                                                line=dict(color='blue', width=2)
                                            ))

                                            # 未来预测
                                            fig_future.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=future_predictions,
                                                mode='lines+markers',
                                                name='未来预测',
                                                line=dict(color='green', width=3, dash='dash'),
                                                marker=dict(size=8)
                                            ))

                                            # 连接线
                                            fig_future.add_trace(go.Scatter(
                                                x=[history_dates[-1], pred_dates[0]],
                                                y=[history_prices[-1], future_predictions[0]],
                                                mode='lines',
                                                line=dict(color='gray', width=1, dash='dot'),
                                                showlegend=False
                                            ))

                                            fig_future.update_layout(
                                                title=f'{train_symbol} 未来价格预测 ({model_type_display})',
                                                xaxis_title='日期',
                                                yaxis_title='价格',
                                                height=500,
                                                template='plotly_white',
                                                hovermode='x unified'
                                            )

                                            st.plotly_chart(fig_future, use_container_width=True)

                            except ImportError as e:
                                st.error(str(e))
                                st.info("请安装TensorFlow: pip install tensorflow")
                            except Exception as e:
                                st.error(f"❌ 训练失败: {e}")
                                st.code(traceback.format_exc())
                else:
                    st.warning(f"⚠️ {train_symbol}数据不足，至少需要100条数据，当前只有{len(df)}条")

    # 标签页6: 模型验证
    with tab6:
        st.header("模型验证")

        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取数据")
        else:
            # 选择验证股票
            validate_symbol = st.selectbox(
                "选择验证股票",
                list(st.session_state.data_cache.keys()),
                key="validate_select"
            )

            if validate_symbol:
                df = st.session_state.data_cache[validate_symbol]

                if len(df) >= 100:
                    # 验证配置
                    st.subheader("验证配置")

                    col1, col2 = st.columns(2)
                    with col1:
                        lookback = st.slider("回看天数", 20, 100, 30, key="validate_lookback")
                    with col2:
                        forecast_days = st.slider("预测天数", 1, 10, 5, key="validate_forecast")

                    # 模型选择
                    st.subheader("选择验证模型")

                    validation_mode = st.radio(
                        "验证模式",
                        ["使用当前会话模型", "加载历史模型文件", "使用最近训练模型"],
                        horizontal=True
                    )

                    model_to_validate = None
                    model_type_to_validate = None
                    model_info = None
                    feature_columns = None

                    if validation_mode == "使用当前会话模型":
                        if st.session_state.current_model:
                            model_to_validate = st.session_state.current_model
                            model_type_to_validate = st.session_state.current_model_type
                            model_info = st.session_state.current_model_info
                            if model_info and 'feature_columns' in model_info:
                                feature_columns = model_info['feature_columns']

                            model_type_display = {
                                'random_forest': '随机森林',
                                'xgboost': 'XGBoost',
                                'lstm': 'LSTM',
                                'lstm_attention': 'LSTM with Attention'
                            }
                            display_type = model_type_display.get(model_type_to_validate, model_type_to_validate)
                            st.success(f"✅ 将使用当前会话模型: {display_type}")
                        else:
                            st.warning("⚠️ 当前会话没有可用的模型，请先训练模型或加载模型文件")

                    elif validation_mode == "使用最近训练模型":
                        if st.session_state.last_trained_model:
                            model_to_validate = st.session_state.last_trained_model['model']
                            model_type_to_validate = st.session_state.last_trained_model['model_type']
                            model_info = st.session_state.last_trained_model
                            if 'feature_columns' in model_info:
                                feature_columns = model_info['feature_columns']

                            model_type_display = {
                                'random_forest': '随机森林',
                                'xgboost': 'XGBoost',
                                'lstm': 'LSTM',
                                'lstm_attention': 'LSTM with Attention'
                            }
                            display_type = model_type_display.get(model_type_to_validate, model_type_to_validate)
                            st.success(f"✅ 将使用最近训练模型: {display_type} ({model_info.get('symbol', '未知')})")
                        else:
                            st.warning("⚠️ 没有最近训练的模型，请先训练模型")

                    else:  # 加载历史模型文件
                        saved_models = predictor.get_saved_models()
                        if saved_models:
                            model_options = [f"{m['name']} ({m['symbol']}, {m['model_type']}, {m['save_time']})"
                                             for m in saved_models]
                            selected_model_display = st.selectbox("选择模型文件", model_options)

                            if selected_model_display and st.button("📥 加载模型文件"):
                                selected_index = model_options.index(selected_model_display)
                                model_path = saved_models[selected_index]['path']

                                with st.spinner("正在加载模型文件..."):
                                    model, loaded_model_type, loaded_model_info = predictor.load_model(model_path)

                                    if model:
                                        model_to_validate = model
                                        model_type_to_validate = loaded_model_type
                                        model_info = loaded_model_info
                                        if 'feature_columns' in loaded_model_info:
                                            feature_columns = loaded_model_info['feature_columns']

                                        # 也更新到当前会话
                                        st.session_state.current_model = model
                                        st.session_state.current_model_type = loaded_model_type
                                        st.session_state.current_model_info = loaded_model_info

                                        model_type_display = {
                                            'random_forest': '随机森林',
                                            'xgboost': 'XGBoost',
                                            'lstm': 'LSTM',
                                            'lstm_attention': 'LSTM with Attention'
                                        }
                                        display_type = model_type_display.get(loaded_model_type, loaded_model_type)
                                        st.success(
                                            f"✅ 模型加载成功: {display_type} ({loaded_model_info.get('symbol', '未知')})")
                                        st.rerun()
                                    else:
                                        st.error("❌ 模型加载失败")
                        else:
                            st.info("没有找到保存的模型文件")

                    # 开始验证
                    if model_to_validate and st.button("✅ 开始验证", type="primary", use_container_width=True):
                        with st.spinner("正在验证模型..."):
                            try:
                                # 使用模型训练时使用的特征，如果未指定则使用默认特征
                                if not feature_columns:
                                    feature_columns = ['close', 'vol', 'returns', 'MA5', 'MA10', 'MA20']

                                # 准备数据
                                X_train, y_train, X_test, y_test = predictor.prepare_features(
                                    df, lookback, forecast_days, feature_columns
                                )

                                if X_train is None:
                                    st.error("数据不足，请增加数据量或减少回看天数")
                                else:
                                    # 评估模型
                                    results = predictor.evaluate_model(
                                        model_to_validate, X_test, y_test, model_type_to_validate
                                    )

                                    # 显示验证结果
                                    st.success("✅ 模型验证完成!")

                                    # 性能指标
                                    st.subheader("验证指标")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("RMSE", f"{results['rmse']:.4f}",
                                                  delta_color="inverse" if results['rmse'] < 1 else "off")
                                    with col2:
                                        st.metric("MAE", f"{results['mae']:.4f}",
                                                  delta_color="inverse" if results['mae'] < 0.5 else "off")
                                    with col3:
                                        st.metric("R²", f"{results['r2']:.4f}",
                                                  delta_color="normal" if results['r2'] > 0.5 else "off")
                                    with col4:
                                        st.metric("方向准确率", f"{results['direction_accuracy']:.2%}",
                                                  delta_color="normal" if results[
                                                                              'direction_accuracy'] > 0.5 else "off")

                                    # 详细指标表格
                                    st.subheader("详细指标")
                                    metrics_df = pd.DataFrame({
                                        '指标': ['RMSE', 'MAE', 'R²', '方向准确率'],
                                        '值': [f"{results['rmse']:.4f}",
                                               f"{results['mae']:.4f}",
                                               f"{results['r2']:.4f}",
                                               f"{results['direction_accuracy']:.2%}"],
                                        '解释': [
                                            '均方根误差，越小越好',
                                            '平均绝对误差，越小越好',
                                            '决定系数，越接近1越好',
                                            '价格方向预测准确率，越高越好'
                                        ]
                                    })
                                    st.dataframe(metrics_df, use_container_width=True)

                                    # 可视化验证结果
                                    st.subheader("验证结果可视化")

                                    # 获取对应的日期
                                    test_start_idx = int(len(df) * 0.8) - lookback
                                    if test_start_idx < 0:
                                        test_start_idx = 0

                                    num_predictions = len(results['actual'])
                                    if num_predictions > 0:
                                        # 获取日期
                                        test_dates = []
                                        for i in range(num_predictions):
                                            date_idx = test_start_idx + lookback + i * forecast_days
                                            if date_idx < len(df):
                                                test_dates.append(df['trade_date'].iloc[date_idx])
                                            else:
                                                # 如果没有足够的日期，使用最后一个日期加上偏移
                                                last_date = df['trade_date'].iloc[-1]
                                                offset = date_idx - len(df) + 1
                                                if isinstance(last_date, pd.Timestamp):
                                                    new_date = last_date + timedelta(days=offset)
                                                else:
                                                    new_date = pd.to_datetime(last_date) + timedelta(days=offset)
                                                test_dates.append(new_date)

                                        # 限制显示数量
                                        display_limit = min(30, num_predictions)

                                        # 模型名称显示
                                        model_name = f"{model_type_to_validate}"
                                        if model_info and 'symbol' in model_info:
                                            model_name = f"{model_info['symbol']}_{model_name}"

                                        fig_validation = visualizer.plot_validation_results(
                                            results['actual'][:display_limit],
                                            results['predictions'][:display_limit],
                                            test_dates[:display_limit],
                                            results,
                                            model_name
                                        )
                                        st.plotly_chart(fig_validation, use_container_width=True)

                                        # 预测未来价格
                                        st.subheader("未来价格预测")

                                        if 'close' in df.columns:
                                            # 获取最新数据
                                            latest_features = df[feature_columns].iloc[-lookback:].values

                                            # 标准化（如果模型有scaler）
                                            if hasattr(predictor, 'scalers') and 'X' in predictor.scalers:
                                                latest_features_scaled = predictor.scalers['X'].transform(
                                                    latest_features.reshape(-1, len(feature_columns))
                                                ).reshape(1, lookback, len(feature_columns))
                                            else:
                                                latest_features_scaled = latest_features.reshape(1, lookback,
                                                                                                 len(feature_columns))

                                            # 预测未来
                                            future_predictions = predictor.predict_future(
                                                model_to_validate, latest_features_scaled,
                                                forecast_days, model_type_to_validate
                                            )

                                            # 显示预测结果
                                            last_price = df['close'].iloc[-1]
                                            last_date = df['trade_date'].iloc[-1]

                                            if isinstance(last_date, str):
                                                last_date = pd.to_datetime(last_date)

                                            pred_dates = [last_date + timedelta(days=i + 1) for i in
                                                          range(forecast_days)]

                                            pred_df = pd.DataFrame({
                                                '日期': [d.strftime('%Y-%m-%d') for d in pred_dates],
                                                '预测价格': np.round(future_predictions, 4),
                                                '变化量': np.round(future_predictions - last_price, 4),
                                                '变化率%': np.round(
                                                    (future_predictions - last_price) / last_price * 100, 2)
                                            })

                                            # 格式化显示
                                            st.dataframe(pred_df.style.format({
                                                '预测价格': '{:.4f}',
                                                '变化量': '{:.4f}',
                                                '变化率%': '{:.2f}%'
                                            }).apply(
                                                lambda x: [
                                                    'background-color: lightgreen' if v > 0 else 'background-color: lightcoral'
                                                    for v in x],
                                                subset=['变化率%']
                                            ))

                                            # 绘制未来预测图
                                            fig_future = go.Figure()

                                            # 历史价格（最近60天）
                                            history_days = min(60, len(df))
                                            history_prices = df['close'].values[-history_days:]
                                            history_dates = df['trade_date'].values[-history_days:]

                                            fig_future.add_trace(go.Scatter(
                                                x=history_dates,
                                                y=history_prices,
                                                mode='lines',
                                                name='历史价格',
                                                line=dict(color='blue', width=2)
                                            ))

                                            # 未来预测
                                            fig_future.add_trace(go.Scatter(
                                                x=pred_dates,
                                                y=future_predictions,
                                                mode='lines+markers',
                                                name='未来预测',
                                                line=dict(color='green', width=3, dash='dash'),
                                                marker=dict(size=8)
                                            ))

                                            # 连接线
                                            fig_future.add_trace(go.Scatter(
                                                x=[history_dates[-1], pred_dates[0]],
                                                y=[history_prices[-1], future_predictions[0]],
                                                mode='lines',
                                                line=dict(color='gray', width=1, dash='dot'),
                                                showlegend=False
                                            ))

                                            # 模型名称显示
                                            model_type_display = {
                                                'random_forest': '随机森林',
                                                'xgboost': 'XGBoost',
                                                'lstm': 'LSTM',
                                                'lstm_attention': 'LSTM with Attention'
                                            }
                                            display_type = model_type_display.get(model_type_to_validate,
                                                                                  model_type_to_validate)

                                            fig_future.update_layout(
                                                title=f'{validate_symbol} 未来价格预测 ({display_type})',
                                                xaxis_title='日期',
                                                yaxis_title='价格',
                                                height=500,
                                                template='plotly_white',
                                                hovermode='x unified'
                                            )

                                            st.plotly_chart(fig_future, use_container_width=True)

                            except Exception as e:
                                st.error(f"❌ 验证失败: {e}")
                                st.code(traceback.format_exc())
                else:
                    st.warning(f"⚠️ {validate_symbol}数据不足，至少需要100条数据，当前只有{len(df)}条")

    # 标签页7: 交易策略与回测
    with tab7:
        st.header("📈 交易策略与回测")

        if not st.session_state.data_cache:
            st.info("请先在数据抓取页面获取股票数据")
        else:
            # 选择股票数据
            strategy_symbol = st.selectbox(
                "选择股票数据",
                list(st.session_state.data_cache.keys()),
                key="strategy_select"
            )

            if strategy_symbol:
                df = st.session_state.data_cache[strategy_symbol].copy()

                if len(df) >= 50:
                    # 策略选择部分
                    st.subheader("策略选择")

                    col1, col2 = st.columns(2)
                    with col1:
                        strategy_options = list(trading_strategy.strategies.items())
                        selected_strategy_display = st.selectbox(
                            "选择交易策略",
                            options=[f"{name} ({desc})" for name, desc in strategy_options],
                            help="选择要测试的交易策略"
                        )

                        # 提取策略名称
                        for name, desc in strategy_options:
                            if f"{name} ({desc})" == selected_strategy_display:
                                selected_strategy = name
                                break

                    with col2:
                        initial_capital = st.number_input(
                            "初始资金（元）",
                            min_value=10000,
                            max_value=1000000,
                            value=100000,
                            step=10000
                        )

                    # 策略参数配置 - 使用更积极的参数
                    st.subheader("策略参数")
                    default_params = trading_strategy.get_strategy_params(selected_strategy)

                    # 设置更积极的参数以生成更多交易信号
                    if selected_strategy == 'ma_crossover':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            fast_period = st.number_input("快速均线周期", 3, 20, 5)
                        with col2:
                            slow_period = st.number_input("慢速均线周期", 10, 60, 20)
                        with col3:
                            stop_loss = st.number_input("止损比例", 0.01, 0.20, 0.05, step=0.01, format="%.2f")
                        with col4:
                            take_profit = st.number_input("止盈比例", 0.01, 0.30, 0.10, step=0.01, format="%.2f")

                        strategy_params = {
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }

                    elif selected_strategy == 'bollinger_band':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            period = st.number_input("布林带周期", 10, 30, 20)
                        with col2:
                            devfactor = st.number_input("标准差倍数", 1.5, 3.0, 2.0, step=0.1)
                        with col3:
                            stop_loss = st.number_input("止损比例", 0.01, 0.20, 0.04, step=0.01, format="%.2f")
                        with col4:
                            take_profit = st.number_input("止盈比例", 0.01, 0.30, 0.08, step=0.01, format="%.2f")

                        strategy_params = {
                            'period': period,
                            'devfactor': devfactor,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }

                    elif selected_strategy == 'rsi_strategy':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            period = st.number_input("RSI周期", 5, 20, 14)
                        with col2:
                            oversold = st.number_input("超卖阈值", 20, 40, 30)
                        with col3:
                            overbought = st.number_input("超买阈值", 60, 80, 70)

                        col4, col5 = st.columns(2)
                        with col4:
                            stop_loss = st.number_input("止损比例", 0.01, 0.20, 0.05, step=0.01, format="%.2f")

                        strategy_params = {
                            'period': period,
                            'oversold': oversold,
                            'overbought': overbought,
                            'stop_loss': stop_loss
                        }

                    # elif selected_strategy == 'macd_strategy':
                    #     col1, col2, col3, col4 = st.columns(4)
                    #     with col1:
                    #         fast_period = st.number_input("快速周期", 8, 15, 12)
                    #     with col2:
                    #         slow_period = st.number_input("慢速周期", 20, 30, 26)
                    #     with col3:
                    #         signal_period = st.number_input("信号周期", 5, 15, 9)
                    #     with col4:
                    #         stop_loss = st.number_input("止损比例", 0.01, 0.20, 0.05, step=0.01, format="%.2f")
                    #
                    #     strategy_params = {
                    #         'fast_period': fast_period,
                    #         'slow_period': slow_period,
                    #         'signal_period': signal_period,
                    #         'stop_loss': stop_loss
                    #     }
                    #
                    # elif selected_strategy == 'model_based':
                    #     model_to_use = None
                    #     if st.session_state.current_model:
                    #         use_model = st.checkbox("使用当前训练模型", value=True)
                    #         if use_model:
                    #             model_to_use = st.session_state.current_model
                    #             st.success(f"✅ 将使用 {st.session_state.current_model_type} 模型")
                    #
                    #     col1, col2, col3, col4 = st.columns(4)
                    #     with col1:
                    #         lookback = st.number_input("回看天数", 10, 60, 30)
                    #     with col2:
                    #         forecast_days = st.number_input("预测天数", 1, 10, 5)
                    #     with col3:
                    #         confidence = st.number_input("置信阈值", 0.1, 0.9, 0.6, step=0.1)
                    #     with col4:
                    #         stop_loss = st.number_input("止损比例", 0.01, 0.20, 0.07, step=0.01, format="%.2f")
                    #
                    #     strategy_params = {
                    #         'lookback': lookback,
                    #         'forecast_days': forecast_days,
                    #         'confidence_threshold': confidence,
                    #         'stop_loss': stop_loss,
                    #         'take_profit': 0.12
                    #     }
                    #
                    # # 在tab7的策略参数部分，为MACD和模型策略设置更积极的参数：

                    elif selected_strategy == 'macd_strategy':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            fast_period = st.number_input("快速周期", 8, 15, 12)
                        with col2:
                            slow_period = st.number_input("慢速周期", 20, 30, 26)
                        with col3:
                            signal_period = st.number_input("信号周期", 5, 15, 9)
                        with col4:
                            stop_loss = st.number_input("止损比例", 0.02, 0.10, 0.05, step=0.01, format="%.2f")

                        # 添加零轴过滤选项
                        zero_line_filter = st.checkbox("启用零轴过滤", value=False)
                        # if zero_line_filter:
                        #     strategy_params['zero_line_filter'] = True

                        strategy_params = {
                            'fast_period': fast_period,
                            'slow_period': slow_period,
                            'signal_period': signal_period,
                            'stop_loss': stop_loss
                        }

                    elif selected_strategy == 'model_based':
                        # 模型策略参数
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            confidence = st.slider("置信阈值", 0.1, 0.9, 0.4, step=0.05)
                        with col2:
                            min_holding_days = st.number_input("最小持有天数", 1, 10, 3)
                        with col3:
                            stop_loss = st.number_input("止损比例", 0.03, 0.15, 0.07, step=0.01, format="%.2f")
                        with col4:
                            take_profit = st.number_input("止盈比例", 0.05, 0.20, 0.12, step=0.01, format="%.2f")

                        strategy_params = {
                            'confidence_threshold': confidence,
                            'min_holding_days': min_holding_days,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit
                        }



                    # 回测时间范围
                    st.subheader("回测设置")

                    col1, col2 = st.columns(2)
                    with col1:
                        backtest_start_date = st.date_input(
                            "回测开始日期",
                            value=df['trade_date'].min().date(),
                            min_value=df['trade_date'].min().date(),
                            max_value=df['trade_date'].max().date()
                        )
                    with col2:
                        backtest_end_date = st.date_input(
                            "回测结束日期",
                            value=df['trade_date'].max().date(),
                            min_value=df['trade_date'].min().date(),
                            max_value=df['trade_date'].max().date()
                        )

                    # 筛选回测数据
                    backtest_df = df[
                        (df['trade_date'] >= pd.to_datetime(backtest_start_date)) &
                        (df['trade_date'] <= pd.to_datetime(backtest_end_date))
                        ].copy()

                    # 调试选项
                    st.subheader("调试选项")
                    show_debug = st.checkbox("显示调试信息", value=False)

                    # 开始回测
                    if st.button("🚀 开始回测", type="primary", use_container_width=True):
                        if len(backtest_df) < 20:
                            st.error("回测数据不足，请选择更长的时间范围")
                        else:
                            with st.spinner("正在运行回测..."):
                                try:
                                    # 更新回测引擎的初始资金
                                    backtest_engine.initial_capital = initial_capital

                                    # 计算交易信号 - 增加调试信息
                                    signals = trading_strategy.calculate_signals(
                                        backtest_df, selected_strategy, strategy_params, model_to_use
                                    )

                                    # 显示调试信息
                                    if show_debug:
                                        st.subheader("信号调试信息")
                                        st.write(
                                            f"数据日期范围: {backtest_df['trade_date'].min()} 到 {backtest_df['trade_date'].max()}")
                                        st.write(f"数据条数: {len(backtest_df)}")
                                        st.write(f"信号数据形状: {signals.shape}")

                                        if 'signal' in signals.columns:
                                            signal_counts = signals['signal'].value_counts()
                                            st.write(f"信号分布: {dict(signal_counts)}")

                                        if 'position' in signals.columns:
                                            position_counts = signals['position'].dropna().value_counts()
                                            st.write(f"交易信号分布: {dict(position_counts)}")

                                        # 显示信号预览
                                        st.write("信号数据预览:")
                                        preview_cols = ['signal'] if 'signal' in signals.columns else []
                                        if 'position' in signals.columns:
                                            preview_cols.append('position')
                                        if preview_cols:
                                            st.dataframe(signals[preview_cols].head(20))

                                    # 检查是否有交易信号
                                    if 'position' in signals.columns and signals['position'].notna().sum() == 0:
                                        st.warning("⚠️ 没有生成交易信号，可能参数过于保守或数据不满足策略条件")
                                        st.info("建议：尝试调整策略参数或选择更长的时间范围")

                                    # 运行回测
                                    results, trades, portfolio_values, dates = backtest_engine.run_backtest(
                                        backtest_df, signals, selected_strategy, strategy_params
                                    )

                                    # 显示回测结果
                                    st.success("✅ 回测完成!")

                                    # 显示关键指标
                                    st.subheader("回测结果")

                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("总收益率", f"{results.get('total_return', 0):.2%}")
                                    with col2:
                                        st.metric("年化收益率", f"{results.get('annual_return', 0):.2%}")
                                    with col3:
                                        st.metric("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}")
                                    with col4:
                                        st.metric("最大回撤", f"{results.get('max_drawdown', 0):.2%}")

                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("总交易次数", results.get('total_trades', 0))
                                    with col2:
                                        st.metric("胜率", f"{results.get('win_rate', 0):.2%}")
                                    with col3:
                                        st.metric("初始资金", f"¥{results.get('initial_capital', 0):,.0f}")
                                    with col4:
                                        st.metric("最终价值", f"¥{results.get('final_value', 0):,.0f}")

                                    # 绘制回测图表
                                    st.subheader("回测图表")

                                    if results.get('total_trades', 0) > 0:
                                        # 创建图表
                                        fig = make_subplots(
                                            rows=2, cols=1,
                                            shared_xaxes=True,
                                            vertical_spacing=0.05,
                                            row_heights=[0.7, 0.3],
                                            subplot_titles=['投资组合价值曲线', '每日收益率']
                                        )

                                        # 投资组合价值曲线
                                        fig.add_trace(
                                            go.Scatter(
                                                x=dates,
                                                y=portfolio_values,
                                                mode='lines',
                                                name='投资组合价值',
                                                line=dict(color='green', width=2)
                                            ),
                                            row=1, col=1
                                        )

                                        # 标记买入卖出点
                                        if trades:
                                            buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
                                            buy_values = [portfolio_values[dates.index(date)] if date in dates else 0
                                                          for date in buy_dates]

                                            sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
                                            sell_values = [portfolio_values[dates.index(date)] if date in dates else 0
                                                           for date in sell_dates]

                                            fig.add_trace(
                                                go.Scatter(
                                                    x=buy_dates,
                                                    y=buy_values,
                                                    mode='markers',
                                                    name='买入点',
                                                    marker=dict(color='blue', size=10, symbol='triangle-up')
                                                ),
                                                row=1, col=1
                                            )

                                            fig.add_trace(
                                                go.Scatter(
                                                    x=sell_dates,
                                                    y=sell_values,
                                                    mode='markers',
                                                    name='卖出点',
                                                    marker=dict(color='red', size=10, symbol='triangle-down')
                                                ),
                                                row=1, col=1
                                            )

                                        # 每日收益率
                                        if len(portfolio_values) > 1:
                                            daily_returns = pd.Series(portfolio_values).pct_change().dropna()
                                            colors = ['green' if r >= 0 else 'red' for r in daily_returns]

                                            fig.add_trace(
                                                go.Bar(
                                                    x=dates[1:],
                                                    y=daily_returns,
                                                    name='日收益率',
                                                    marker_color=colors,
                                                    opacity=0.7
                                                ),
                                                row=2, col=1
                                            )

                                        fig.update_layout(
                                            height=700,
                                            showlegend=True,
                                            template='plotly_white',
                                            title_text=f"{strategy_symbol} - {trading_strategy.strategies[selected_strategy]} 回测结果"
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                        # 显示交易记录
                                        st.subheader("交易记录")

                                        trades_df = pd.DataFrame(trades)
                                        if not trades_df.empty:
                                            # 计算盈亏百分比
                                            trades_df['profit_pct'] = trades_df.apply(
                                                lambda row: row['profit'] / (row['shares'] * row['price']) * 100
                                                if row['type'] == 'SELL' and row['shares'] > 0 else 0,
                                                axis=1
                                            )

                                            # 格式化显示
                                            display_cols = ['date', 'type', 'price', 'shares', 'value', 'profit',
                                                            'profit_pct']
                                            display_trades = trades_df[display_cols].copy()
                                            display_trades['date'] = display_trades['date'].dt.strftime('%Y-%m-%d')

                                            st.dataframe(
                                                display_trades.style.format({
                                                    'price': '{:.4f}',
                                                    'value': '{:.2f}',
                                                    'profit': '{:.2f}',
                                                    'profit_pct': '{:.2f}%'
                                                }).apply(
                                                    lambda x: [
                                                        'background-color: lightgreen' if v > 0 else 'background-color: lightcoral'
                                                        for v in x] if x.name in ['profit',
                                                                                  'profit_pct'] else [''] * len(x),
                                                    axis=0
                                                ),
                                                use_container_width=True
                                            )
                                        else:
                                            st.info("没有交易记录")

                                    else:
                                        st.info("本次回测没有产生交易，图表无法显示")
                                        st.warning("""
                                        可能原因：
                                        1. 策略参数过于保守，没有触发交易信号
                                        2. 回测时间范围太短
                                        3. 策略条件不满足

                                        建议：
                                        1. 调整策略参数（如降低阈值）
                                        2. 选择更长的时间范围
                                        3. 尝试不同的策略
                                        """)

                                except Exception as e:
                                    st.error(f"❌ 回测失败: {e}")
                                    st.code(traceback.format_exc())



    # # 标签页7: 交易策略与回测
    # with tab7:
    #     st.header("📈 交易策略与回测")
    #
    #     if not st.session_state.data_cache:
    #         st.info("请先在数据抓取页面获取股票数据")
    #     else:
    #         # 选择股票数据
    #         strategy_symbol = st.selectbox(
    #             "选择股票数据",
    #             list(st.session_state.data_cache.keys()),
    #             key="strategy_select"
    #         )
    #
    #         if strategy_symbol:
    #             df = st.session_state.data_cache[strategy_symbol].copy()
    #
    #             if len(df) >= 50:  # 至少需要50条数据
    #                 # 策略选择部分
    #                 st.subheader("策略选择")
    #
    #                 col1, col2 = st.columns(2)
    #                 with col1:
    #                     # 选择策略类型
    #                     strategy_options = list(trading_strategy.strategies.items())
    #                     selected_strategy_display = st.selectbox(
    #                         "选择交易策略",
    #                         options=[f"{name} ({desc})" for name, desc in strategy_options],
    #                         help="选择要测试的交易策略"
    #                     )
    #
    #                     # 提取策略名称
    #                     for name, desc in strategy_options:
    #                         if f"{name} ({desc})" == selected_strategy_display:
    #                             selected_strategy = name
    #                             break
    #
    #                 with col2:
    #                     # 初始资金设置
    #                     initial_capital = st.number_input(
    #                         "初始资金（元）",
    #                         min_value=10000,
    #                         max_value=1000000,
    #                         value=100000,
    #                         step=10000
    #                     )
    #
    #                 # 策略参数配置
    #                 st.subheader("策略参数")
    #
    #                 # 获取策略默认参数
    #                 default_params = trading_strategy.get_strategy_params(selected_strategy)
    #
    #                 # 根据策略类型显示不同的参数
    #                 if selected_strategy == 'ma_crossover':
    #                     col1, col2, col3, col4 = st.columns(4)
    #                     with col1:
    #                         fast_period = st.number_input("快速均线周期", 3, 50, default_params.get('fast_period', 5))
    #                     with col2:
    #                         slow_period = st.number_input("慢速均线周期", 10, 200,
    #                                                       default_params.get('slow_period', 20))
    #                     with col3:
    #                         stop_loss = st.number_input("止损比例", 0.01, 0.20, default_params.get('stop_loss', 0.05),
    #                                                     step=0.01, format="%.2f")
    #                     with col4:
    #                         take_profit = st.number_input("止盈比例", 0.01, 0.30,
    #                                                       default_params.get('take_profit', 0.10), step=0.01,
    #                                                       format="%.2f")
    #
    #                     strategy_params = {
    #                         'fast_period': fast_period,
    #                         'slow_period': slow_period,
    #                         'stop_loss': stop_loss,
    #                         'take_profit': take_profit
    #                     }
    #
    #                 elif selected_strategy == 'bollinger_band':
    #                     col1, col2, col3, col4 = st.columns(4)
    #                     with col1:
    #                         period = st.number_input("布林带周期", 10, 50, default_params.get('period', 20))
    #                     with col2:
    #                         devfactor = st.number_input("标准差倍数", 1.0, 3.0, default_params.get('devfactor', 2.0),
    #                                                     step=0.1)
    #                     with col3:
    #                         stop_loss = st.number_input("止损比例", 0.01, 0.20, default_params.get('stop_loss', 0.04),
    #                                                     step=0.01, format="%.2f")
    #                     with col4:
    #                         take_profit = st.number_input("止盈比例", 0.01, 0.30,
    #                                                       default_params.get('take_profit', 0.08), step=0.01,
    #                                                       format="%.2f")
    #
    #                     strategy_params = {
    #                         'period': period,
    #                         'devfactor': devfactor,
    #                         'stop_loss': stop_loss,
    #                         'take_profit': take_profit
    #                     }
    #
    #                 elif selected_strategy == 'rsi_strategy':
    #                     col1, col2, col3 = st.columns(3)
    #                     with col1:
    #                         period = st.number_input("RSI周期", 5, 30, default_params.get('period', 14))
    #                     with col2:
    #                         oversold = st.number_input("超卖阈值", 10, 40, default_params.get('oversold', 30))
    #                     with col3:
    #                         overbought = st.number_input("超买阈值", 60, 90, default_params.get('overbought', 70))
    #
    #                     col4, col5 = st.columns(2)
    #                     with col4:
    #                         stop_loss = st.number_input("止损比例", 0.01, 0.20, default_params.get('stop_loss', 0.05),
    #                                                     step=0.01, format="%.2f")
    #
    #                     strategy_params = {
    #                         'period': period,
    #                         'oversold': oversold,
    #                         'overbought': overbought,
    #                         'stop_loss': stop_loss
    #                     }
    #
    #                 elif selected_strategy == 'macd_strategy':
    #                     col1, col2, col3, col4 = st.columns(4)
    #                     with col1:
    #                         fast_period = st.number_input("快速周期", 5, 20, default_params.get('fast_period', 12))
    #                     with col2:
    #                         slow_period = st.number_input("慢速周期", 15, 40, default_params.get('slow_period', 26))
    #                     with col3:
    #                         signal_period = st.number_input("信号周期", 5, 20, default_params.get('signal_period', 9))
    #                     with col4:
    #                         stop_loss = st.number_input("止损比例", 0.01, 0.20, default_params.get('stop_loss', 0.05),
    #                                                     step=0.01, format="%.2f")
    #
    #                     # strategy_params = {
    #                     #     'fast_period': fast_period,
    #                     #     'slow_period': slow_period,
    #                     #     'signal_period': signal_period,
    #                     #     'stop_loss': stop_loss
    #                     # }
    #                     strategy_params = {
    #                         'fast_period': 8,  # 缩短快速周期
    #                         'slow_period': 21,  # 缩短慢速周期
    #                         'signal_period': 9,
    #                         'stop_loss': 0.03  # 减小止损比例
    #                     }
    #
    #
    #                 elif selected_strategy == 'model_based':
    #                     # 检查是否有训练好的模型
    #                     model_to_use = None
    #                     if st.session_state.current_model:
    #                         use_model = st.checkbox("使用当前训练模型", value=True)
    #                         if use_model:
    #                             model_to_use = st.session_state.current_model
    #                             st.success(f"✅ 将使用 {st.session_state.current_model_type} 模型")
    #                     else:
    #                         st.warning("⚠️ 没有可用的训练模型，将使用简化模型")
    #
    #                     col1, col2, col3, col4 = st.columns(4)
    #                     with col1:
    #                         lookback = st.number_input("回看天数", 10, 100, default_params.get('lookback', 30))
    #                     with col2:
    #                         forecast_days = st.number_input("预测天数", 1, 10, default_params.get('forecast_days', 5))
    #                     with col3:
    #                         confidence = st.number_input("置信阈值", 0.1, 0.9,
    #                                                      default_params.get('confidence_threshold', 0.6), step=0.1)
    #                     with col4:
    #                         stop_loss = st.number_input("止损比例", 0.01, 0.20, default_params.get('stop_loss', 0.07),
    #                                                     step=0.01, format="%.2f")
    #
    #                     # strategy_params = {
    #                     #     'lookback': lookback,
    #                     #     'forecast_days': forecast_days,
    #                     #     'confidence_threshold': confidence,
    #                     #     'stop_loss': stop_loss,
    #                     #     'take_profit': default_params.get('take_profit', 0.12)
    #                     # }
    #                     strategy_params = {
    #                         'lookback': 20,  # 缩短回看天数
    #                         'forecast_days': 3,  # 缩短预测天数
    #                         'confidence_threshold': 0.4,  # 降低置信阈值
    #                         'stop_loss': 0.05,
    #                         'take_profit': 0.08
    #                     }
    #
    #                 # 回测时间范围
    #                 st.subheader("回测设置")
    #
    #                 col1, col2 = st.columns(2)
    #                 with col1:
    #                     # 选择回测时间范围
    #                     backtest_start_date = st.date_input(
    #                         "回测开始日期",
    #                         value=df['trade_date'].min().date(),
    #                         min_value=df['trade_date'].min().date(),
    #                         max_value=df['trade_date'].max().date()
    #                     )
    #                 with col2:
    #                     backtest_end_date = st.date_input(
    #                         "回测结束日期",
    #                         value=df['trade_date'].max().date(),
    #                         min_value=df['trade_date'].min().date(),
    #                         max_value=df['trade_date'].max().date()
    #                     )
    #
    #                 # 筛选回测数据
    #                 backtest_df = df[
    #                     (df['trade_date'] >= pd.to_datetime(backtest_start_date)) &
    #                     (df['trade_date'] <= pd.to_datetime(backtest_end_date))
    #                     ].copy()
    #
    #                 # # 开始回测按钮
    #                 # if st.button("🚀 开始回测", type="primary", use_container_width=True):
    #                 #     if len(backtest_df) < 20:
    #                 #         st.error("回测数据不足，请选择更长的时间范围")
    #                 #     else:
    #                 #         with st.spinner("正在运行回测..."):
    #                 #             try:
    #                 #                 # 更新回测引擎的初始资金
    #                 #                 backtest_engine.initial_capital = initial_capital
    #                 #
    #                 #                 # 计算交易信号
    #                 #                 signals = trading_strategy.calculate_signals(
    #                 #                     backtest_df, selected_strategy, strategy_params, model_to_use
    #                 #                 )
    #                 #
    #                 #                 # 运行回测
    #                 #                 results, trades, portfolio_values, dates = backtest_engine.run_backtest(
    #                 #                     backtest_df, signals, selected_strategy, strategy_params
    #                 #                 )
    #                 #
    #                 #                 # 显示回测结果
    #                 #                 st.success("✅ 回测完成!")
    #
    #                 # 在回测部分添加调试信息
    #                 if st.button("🚀 开始回测", type="primary", use_container_width=True):
    #                     if len(backtest_df) < 20:
    #                         st.error("回测数据不足，请选择更长的时间范围")
    #                     else:
    #                         with st.spinner("正在运行回测..."):
    #                             try:
    #                                 # 更新回测引擎的初始资金
    #                                 backtest_engine.initial_capital = initial_capital
    #
    #                                 # 调试：显示数据信息
    #                                 st.write(
    #                                     f"回测数据时间段: {backtest_df['trade_date'].min()} 到 {backtest_df['trade_date'].max()}")
    #                                 st.write(f"回测数据条数: {len(backtest_df)}")
    #
    #                                 # 计算交易信号
    #                                 signals = trading_strategy.calculate_signals(
    #                                     backtest_df, selected_strategy, strategy_params, model_to_use
    #                                 )
    #
    #                                 # 调试：显示信号统计
    #                                 st.write("信号统计:")
    #                                 st.write(f"- 总信号数: {len(signals)}")
    #                                 if 'position' in signals.columns:
    #                                     buy_signals = sum(signals['position'] == 1)
    #                                     sell_signals = sum(signals['position'] == -1)
    #                                     st.write(f"- 买入信号: {buy_signals}")
    #                                     st.write(f"- 卖出信号: {sell_signals}")
    #
    #                                 # 运行回测
    #                                 results, trades, portfolio_values, dates = backtest_engine.run_backtest(
    #                                     backtest_df, signals, selected_strategy, strategy_params
    #                                 )
    #
    #                                 # 调试：显示回测结果
    #                                 st.write("回测结果统计:")
    #                                 st.write(f"- 交易次数: {len(trades)}")
    #                                 st.write(f"- 投资组合价值数组长度: {len(portfolio_values)}")
    #                                 st.write(
    #                                     f"- 投资组合最终价值: {portfolio_values[-1] if portfolio_values else 'N/A'}")
    #
    #                                 # ... 其余代码不变 ...
    #
    #
    #
    #
    #
    #                                 # 1. 显示关键指标
    #                                 st.subheader("回测结果")
    #
    #                                 # 创建指标卡片
    #                                 col1, col2, col3, col4 = st.columns(4)
    #                                 with col1:
    #                                     st.metric("总收益率", f"{results.get('total_return', 0):.2%}")
    #                                 with col2:
    #                                     st.metric("年化收益率", f"{results.get('annual_return', 0):.2%}")
    #                                 with col3:
    #                                     st.metric("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}")
    #                                 with col4:
    #                                     st.metric("最大回撤", f"{results.get('max_drawdown', 0):.2%}")
    #
    #                                 col1, col2, col3, col4 = st.columns(4)
    #                                 with col1:
    #                                     st.metric("总交易次数", results.get('total_trades', 0))
    #                                 with col2:
    #                                     st.metric("胜率", f"{results.get('win_rate', 0):.2%}")
    #                                 with col3:
    #                                     st.metric("初始资金", f"¥{results.get('initial_capital', 0):,.0f}")
    #                                 with col4:
    #                                     st.metric("最终价值", f"¥{results.get('final_value', 0):,.0f}")
    #
    #                                 # 2. 绘制回测图表
    #                                 st.subheader("回测图表")
    #
    #                                 # 创建图表
    #                                 fig = make_subplots(
    #                                     rows=2, cols=1,
    #                                     shared_xaxes=True,
    #                                     vertical_spacing=0.05,
    #                                     row_heights=[0.7, 0.3],
    #                                     subplot_titles=['投资组合价值曲线', '每日收益率']
    #                                 )
    #
    #                                 # 投资组合价值曲线
    #                                 fig.add_trace(
    #                                     go.Scatter(
    #                                         x=dates,
    #                                         y=portfolio_values,
    #                                         mode='lines',
    #                                         name='投资组合价值',
    #                                         line=dict(color='green', width=2)
    #                                     ),
    #                                     row=1, col=1
    #                                 )
    #
    #                                 # 标记买入卖出点
    #                                 if trades:
    #                                     buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    #                                     buy_values = [portfolio_values[dates.index(date)] if date in dates else 0 for
    #                                                   date in buy_dates]
    #
    #                                     sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    #                                     sell_values = [portfolio_values[dates.index(date)] if date in dates else 0 for
    #                                                    date in sell_dates]
    #
    #                                     fig.add_trace(
    #                                         go.Scatter(
    #                                             x=buy_dates,
    #                                             y=buy_values,
    #                                             mode='markers',
    #                                             name='买入点',
    #                                             marker=dict(color='blue', size=10, symbol='triangle-up')
    #                                         ),
    #                                         row=1, col=1
    #                                     )
    #
    #                                     fig.add_trace(
    #                                         go.Scatter(
    #                                             x=sell_dates,
    #                                             y=sell_values,
    #                                             mode='markers',
    #                                             name='卖出点',
    #                                             marker=dict(color='red', size=10, symbol='triangle-down')
    #                                         ),
    #                                         row=1, col=1
    #                                     )
    #
    #                                 # 每日收益率
    #                                 if len(portfolio_values) > 1:
    #                                     daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    #                                     colors = ['green' if r >= 0 else 'red' for r in daily_returns]
    #
    #                                     fig.add_trace(
    #                                         go.Bar(
    #                                             x=dates[1:],
    #                                             y=daily_returns,
    #                                             name='日收益率',
    #                                             marker_color=colors,
    #                                             opacity=0.7
    #                                         ),
    #                                         row=2, col=1
    #                                     )
    #
    #                                 fig.update_layout(
    #                                     height=700,
    #                                     showlegend=True,
    #                                     template='plotly_white',
    #                                     title_text=f"{strategy_symbol} - {trading_strategy.strategies[selected_strategy]} 回测结果"
    #                                 )
    #
    #                                 st.plotly_chart(fig, use_container_width=True)
    #
    #                                 # 3. 显示交易记录
    #                                 st.subheader("交易记录")
    #
    #                                 if trades:
    #                                     trades_df = pd.DataFrame(trades)
    #                                     trades_df['profit_pct'] = trades_df['profit'] / (
    #                                                 trades_df['shares'] * trades_df['price'].shift(1))
    #                                     trades_df['profit_pct'] = trades_df['profit_pct'].fillna(0)
    #
    #                                     # 格式化显示
    #                                     display_cols = ['date', 'type', 'price', 'shares', 'value', 'profit',
    #                                                     'profit_pct']
    #                                     display_trades = trades_df[display_cols].copy()
    #                                     display_trades['date'] = display_trades['date'].dt.strftime('%Y-%m-%d')
    #                                     display_trades['profit_pct'] = display_trades['profit_pct'].apply(
    #                                         lambda x: f"{x:.2%}")
    #
    #                                     st.dataframe(
    #                                         display_trades.style.format({
    #                                             'price': '{:.4f}',
    #                                             'value': '{:.2f}',
    #                                             'profit': '{:.2f}'
    #                                         }).apply(
    #                                             lambda x: [
    #                                                 'background-color: lightgreen' if v > 0 else 'background-color: lightcoral'
    #                                                 for v in x] if x.name == 'profit' else [''] * len(x),
    #                                             axis=0
    #                                         ),
    #                                         use_container_width=True
    #                                     )
    #
    #                                     # 交易统计
    #                                     st.subheader("交易统计")
    #                                     col1, col2 = st.columns(2)
    #                                     with col1:
    #                                         st.write(
    #                                             f"**买入交易**: {len([t for t in trades if t['type'] == 'BUY'])} 次")
    #                                         st.write(
    #                                             f"**卖出交易**: {len([t for t in trades if t['type'] == 'SELL'])} 次")
    #                                         st.write(
    #                                             f"**盈利交易**: {len([t for t in trades if t.get('profit', 0) > 0])} 次")
    #                                         st.write(
    #                                             f"**亏损交易**: {len([t for t in trades if t.get('profit', 0) < 0])} 次")
    #                                     with col2:
    #                                         if len([t for t in trades if t.get('profit', 0) > 0]) > 0:
    #                                             avg_profit = np.mean(
    #                                                 [t.get('profit', 0) for t in trades if t.get('profit', 0) > 0])
    #                                             st.write(f"**平均盈利**: ¥{avg_profit:.2f}")
    #                                         if len([t for t in trades if t.get('profit', 0) < 0]) > 0:
    #                                             avg_loss = np.mean(
    #                                                 [t.get('profit', 0) for t in trades if t.get('profit', 0) < 0])
    #                                             st.write(f"**平均亏损**: ¥{avg_loss:.2f}")
    #
    #                                 else:
    #                                     st.info("本次回测没有产生交易")
    #
    #                                 # 4. 策略对比（如果有多个策略）
    #                                 st.subheader("策略对比建议")
    #
    #                                 # 计算基准收益（买入并持有策略）
    #                                 buy_hold_return = (backtest_df['close'].iloc[-1] - backtest_df['close'].iloc[0]) / \
    #                                                   backtest_df['close'].iloc[0]
    #
    #                                 comparison_data = {
    #                                     '策略': trading_strategy.strategies[selected_strategy],
    #                                     '总收益率': results.get('total_return', 0),
    #                                     '年化收益率': results.get('annual_return', 0),
    #                                     '最大回撤': results.get('max_drawdown', 0),
    #                                     '夏普比率': results.get('sharpe_ratio', 0)
    #                                 }
    #
    #                                 benchmark_data = {
    #                                     '策略': '买入并持有',
    #                                     '总收益率': buy_hold_return,
    #                                     '年化收益率': buy_hold_return,  # 简化处理
    #                                     '最大回撤': 0,  # 需要实际计算
    #                                     '夏普比率': 0  # 需要实际计算
    #                                 }
    #
    #                                 comparison_df = pd.DataFrame([comparison_data, benchmark_data])
    #                                 st.dataframe(comparison_df.style.format({
    #                                     '总收益率': '{:.2%}',
    #                                     '年化收益率': '{:.2%}',
    #                                     '最大回撤': '{:.2%}',
    #                                     '夏普比率': '{:.2f}'
    #                                 }), use_container_width=True)
    #
    #                                 # 策略优化建议
    #                                 st.subheader("策略优化建议")
    #
    #                                 suggestions = []
    #                                 if results.get('total_trades', 0) == 0:
    #                                     suggestions.append("🔍 策略信号过于保守，建议调整参数增加交易机会")
    #                                 elif results.get('win_rate', 0) < 0.4:
    #                                     suggestions.append("🔍 胜率较低，建议优化入场条件或增加过滤条件")
    #                                 elif results.get('max_drawdown', 0) < -0.15:
    #                                     suggestions.append("⚠️ 最大回撤较大，建议加强止损策略或降低仓位")
    #                                 elif results.get('sharpe_ratio', 0) < 0.5:
    #                                     suggestions.append("📈 夏普比率偏低，建议优化风险收益比")
    #                                 else:
    #                                     suggestions.append("✅ 策略表现良好，可以考虑实盘测试")
    #
    #                                 for suggestion in suggestions:
    #                                     st.write(suggestion)
    #
    #                             except Exception as e:
    #                                 st.error(f"❌ 回测失败: {e}")
    #                                 st.code(traceback.format_exc())
    #             else:
    #                 st.warning(f"⚠️ {strategy_symbol}数据不足，至少需要50条数据，当前只有{len(df)}条")

    # 页脚
    st.markdown("---")
    st.caption(
        f"量化分析系统 © {datetime.now().year} | 数据源: Tushare | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    st.caption("⚠️ 注意：股票预测结果和交易策略仅供参考，不构成投资建议。回测结果不代表未来表现，实际交易存在风险。")


# ==================== 运行应用 ====================
if __name__ == "__main__":
    main()