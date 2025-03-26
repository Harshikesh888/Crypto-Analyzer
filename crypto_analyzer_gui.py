import sys
import os
import requests
import pandas as pd
from datetime import datetime
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
                             QMessageBox, QComboBox, QProgressBar, QFrame,
                             QTabWidget, QLineEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QColor, QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import torch
import traceback
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# API Configuration
COINGECKO_API = "https://api.coingecko.com/api/v3"
RATE_LIMIT = 50  # requests per minute
REQUEST_INTERVAL = 60 / RATE_LIMIT  # seconds between requests
last_request_time = 0

# Configure requests session with retries
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Cache configuration
class DataCache:
    def __init__(self, cache_duration=300):  # 5 minutes default
        self.cache_duration = cache_duration
        self.cache = {}
        self.timestamps = {}

    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.cache_duration:
                return self.cache[key]
        return None

    def set(self, key, data):
        self.cache[key] = data
        self.timestamps[key] = time.time()

# Initialize caches
market_cache = DataCache(cache_duration=300)  # 5 minutes for market data
historical_cache = DataCache(cache_duration=3600)  # 1 hour for historical data

def rate_limit():
    global last_request_time
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    if time_since_last_request < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - time_since_last_request)
    last_request_time = time.time()

def fetch_crypto_data(force_refresh=False):
    """Fetch real-time cryptocurrency data from CoinGecko API with rate limiting and caching."""
    try:
        # Check cache first
        if not force_refresh:
            cached_data = market_cache.get('market_data')
            if cached_data is not None:
                print("Using cached market data")
                return cached_data

        print("Fetching data from CoinGecko API...")
        rate_limit()
        
        url = f"{COINGECKO_API}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': 100,
            'sparkline': False
        }
        
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            print("API returned empty data")
            return None

        df = pd.DataFrame(data)
        print(f"Raw data columns: {list(df.columns)}")

        # Map API fields to expected columns
        column_mapping = {
            'market_cap_rank': 'Rank',
            'name': 'Name',
            'symbol': 'Symbol',
            'current_price': 'Price (USD)',
            'price_change_percentage_24h': 'Raw 24h Change',
            'total_volume': '24h Volume (USD)'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
            else:
                print(f"Warning: '{old_col}' not found in API response")
                df[new_col] = 'N/A'

        # Handle missing 1h and 7d changes
        df['Raw 1h Change'] = pd.NA
        df['Raw 7d Change'] = pd.NA

        # Format percentage columns
        def format_percentage(x):
            if pd.isna(x) or not isinstance(x, (int, float)):
                return 'N/A'
            return f"{float(x):.2f}%"

        df['1h Change %'] = df['Raw 1h Change'].apply(format_percentage)
        df['24h Change %'] = df['Raw 24h Change'].apply(format_percentage)
        df['7d Change %'] = df['Raw 7d Change'].apply(format_percentage)

        # Cache the processed data
        market_cache.set('market_data', df)
        print(f"Processed DataFrame shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        traceback.print_exc()
        return None

def fetch_historical_data(coin_id):
    """Fetch historical data for a specific coin with rate limiting and caching."""
    try:
        # Check cache first
        cache_key = f'historical_{coin_id}'
        cached_data = historical_cache.get(cache_key)
        if cached_data is not None:
            print(f"Using cached historical data for {coin_id}")
            return cached_data

        print(f"Fetching historical data for {coin_id}...")
        rate_limit()
        
        url = f"{COINGECKO_API}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': 30,
            'interval': 'daily'
        }
        
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        market_caps = data.get('market_caps', [])

        if not prices or not volumes or not market_caps:
            print("Incomplete historical data returned")
            return None

        df = pd.DataFrame({
            'timestamp': [entry[0] / 1000 for entry in prices],
            'price': [entry[1] for entry in prices],
            'volume': [entry[1] for entry in volumes],
            'market_cap': [entry[1] for entry in market_caps]
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # Cache the data
        historical_cache.set(cache_key, df)
        print(f"Historical data shape: {df.shape}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing historical data: {str(e)}")
        traceback.print_exc()
        return None


def analyze_crypto(coin_id):
    """Basic analysis based on historical data."""
    try:
        print(f"Analyzing {coin_id}...")
        df = fetch_historical_data(coin_id)
        if df is None or df.empty:
            return ("No data available for analysis", 0, "Unknown", "gray")

        volatility = df['price'].pct_change().std() * 100
        risk_score = min(1.0, volatility / 10)
        risk_category = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
        risk_color = "green" if risk_score < 0.3 else "yellow" if risk_score < 0.7 else "red"

        report = f"Analysis for {coin_id}:\nVolatility (30-day): {volatility:.2f}%\nRisk Score: {risk_score:.2f}\nRisk Category: {risk_category}"
        return (report, risk_score, risk_category, risk_color)

    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return (f"Error analyzing {coin_id}: {str(e)}", 0, "Error", "gray")


# Thread classes
class DataFetchThread(QThread):
    data_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, force_refresh=False):
        super().__init__()
        self.force_refresh = force_refresh

    def run(self):
        try:
            print("Starting data fetch...")
            data = fetch_crypto_data(force_refresh=self.force_refresh)
            if data is not None:
                print(f"Data fetched: {type(data)}, shape: {data.shape if isinstance(data, pd.DataFrame) else 'N/A'}")
                self.data_ready.emit(data)
            else:
                print("Data fetch returned None")
                self.error.emit("Failed to fetch cryptocurrency data: No data returned")
        except Exception as e:
            error_msg = f"Data fetch error: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)


class AnalysisThread(QThread):
    analysis_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, coin_id):
        super().__init__()
        self.coin_id = coin_id

    def run(self):
        try:
            result = analyze_crypto(self.coin_id)
            if result:
                print(f"Analysis completed for {self.coin_id}")
                self.analysis_ready.emit(result)
            else:
                self.error.emit(f"Failed to analyze {self.coin_id}")
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)


# Main GUI class
class CryptoAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Cryptocurrency Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        self.crypto_data = None
        self.selected_crypto = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.fetch_thread = None
        self.analysis_thread = None
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)
        self.refresh_timer.start(5000)
        self.setup_ui()
        self.load_crypto_data()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search cryptocurrencies...")
        self.search_box.textChanged.connect(self.filter_crypto_list)
        control_layout.addWidget(self.search_box)
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_data)
        self.refresh_btn.setStyleSheet("padding: 8px;")
        control_layout.addWidget(self.refresh_btn)
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["5s", "10s", "30s", "1m"])
        self.interval_combo.currentTextChanged.connect(self.update_refresh_interval)
        control_layout.addWidget(self.interval_combo)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; padding: 5px;")
        control_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        main_layout.addWidget(control_panel)

        # Content splitter
        content_splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.crypto_table = QTableWidget()
        self.crypto_table.setColumnCount(7)
        self.crypto_table.setHorizontalHeaderLabels([
            "Rank", "Name", "Price", "1h Change", "24h Change", "7d Change", "24h Volume"
        ])
        self.crypto_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.crypto_table.verticalHeader().setVisible(False)
        self.crypto_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.crypto_table.setSelectionMode(QTableWidget.SingleSelection)
        self.crypto_table.itemSelectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.crypto_table)
        content_splitter.addWidget(left_panel)

        # Right panel: Tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        tab_widget = QTabWidget()
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        self.analysis_header = QLabel("Select a cryptocurrency to view analysis")
        self.analysis_header.setFont(QFont("Arial", 14, QFont.Bold))
        self.analysis_header.setAlignment(Qt.AlignCenter)
        overview_layout.addWidget(self.analysis_header)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        analysis_content = QWidget()
        self.analysis_layout = QVBoxLayout(analysis_content)
        self.analysis_text = QLabel()
        self.analysis_text.setWordWrap(True)
        self.analysis_text.setTextFormat(Qt.RichText)
        self.analysis_layout.addWidget(self.analysis_text)
        scroll.setWidget(analysis_content)
        overview_layout.addWidget(scroll)
        tab_widget.addTab(overview_tab, "Overview")

        charts_tab = QWidget()
        charts_layout = QVBoxLayout(charts_tab)
        self.figure, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        charts_layout.addWidget(self.canvas)
        tab_widget.addTab(charts_tab, "Charts")

        right_layout.addWidget(tab_widget)
        content_splitter.addWidget(right_panel)
        main_layout.addWidget(content_splitter)
        content_splitter.setSizes([400, 1000])

    def filter_crypto_list(self, text):
        if self.crypto_data is not None:
            for row in range(self.crypto_table.rowCount()):
                name_item = self.crypto_table.item(row, 1)
                if name_item:
                    name = name_item.text().lower()
                    self.crypto_table.setRowHidden(row, text.lower() not in name)

    def load_data(self):
        if self.fetch_thread and self.fetch_thread.isRunning():
            print("Fetch thread already running, skipping...")
            return
        try:
            print("Initiating data load...")
            self.refresh_btn.setEnabled(False)
            self.status_label.setText("Fetching data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.fetch_thread = DataFetchThread(force_refresh=True)
            self.fetch_thread.data_ready.connect(self.update_table)
            self.fetch_thread.error.connect(self.show_error)
            self.fetch_thread.finished.connect(self.on_fetch_complete)
            self.fetch_thread.start()
        except Exception as e:
            self.show_error(f"Load data error: {str(e)}")

    def update_table(self, data):
        try:
            self.crypto_data = data
            if data is None:
                self.show_error("No data received from fetch")
                return
            if not isinstance(data, pd.DataFrame):
                self.show_error(f"Invalid data type received: {type(data)}")
                return
            if data.empty:
                self.show_error("Received empty dataset")
                return
            print(f"Updating table with {len(data)} rows")
            self.crypto_table.setRowCount(len(data))
            for row, (_, crypto) in enumerate(data.iterrows()):
                rank_item = QTableWidgetItem(str(crypto.get('Rank', 'N/A')))
                rank_item.setTextAlignment(Qt.AlignCenter)
                self.crypto_table.setItem(row, 0, rank_item)
                name = f"{crypto.get('Name', 'Unknown')} ({crypto.get('Symbol', 'N/A').upper()})"
                self.crypto_table.setItem(row, 1, QTableWidgetItem(name))
                price_item = QTableWidgetItem(str(crypto.get('Price (USD)', 'N/A')))
                price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.crypto_table.setItem(row, 2, price_item)
                change_1h = float(crypto.get('Raw 1h Change', 0) if pd.notna(crypto.get('Raw 1h Change')) else 0)
                change_1h_item = QTableWidgetItem(str(crypto.get('1h Change %', 'N/A')))
                change_1h_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                change_1h_item.setForeground(QColor("#4CAF50" if change_1h >= 0 else "#F44336"))
                self.crypto_table.setItem(row, 3, change_1h_item)
                change_24h = float(crypto.get('Raw 24h Change', 0) if pd.notna(crypto.get('Raw 24h Change')) else 0)
                change_24h_item = QTableWidgetItem(str(crypto.get('24h Change %', 'N/A')))
                change_24h_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                change_24h_item.setForeground(QColor("#4CAF50" if change_24h >= 0 else "#F44336"))
                self.crypto_table.setItem(row, 4, change_24h_item)
                change_7d = float(crypto.get('Raw 7d Change', 0) if pd.notna(crypto.get('Raw 7d Change')) else 0)
                change_7d_item = QTableWidgetItem(str(crypto.get('7d Change %', 'N/A')))
                change_7d_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                change_7d_item.setForeground(QColor("#4CAF50" if change_7d >= 0 else "#F44336"))
                self.crypto_table.setItem(row, 5, change_7d_item)
                volume_item = QTableWidgetItem(str(crypto.get('24h Volume (USD)', 'N/A')))
                volume_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.crypto_table.setItem(row, 6, volume_item)
                self.crypto_table.item(row, 0).setData(Qt.UserRole, crypto.get('id', ''))
        except Exception as e:
            error_msg = f"Error updating table: {str(e)}"
            print(error_msg)
            self.show_error(error_msg)

    def show_error(self, message):
        QMessageBox.warning(self, "Error", message)
        self.status_label.setText("Error occurred")
        self.progress_bar.setVisible(False)
        self.refresh_btn.setEnabled(True)

    def on_fetch_complete(self):
        self.refresh_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Data fetched at {datetime.now().strftime('%H:%M:%S')}")
        print("Fetch completed")

    def auto_refresh(self):
        if self.fetch_thread and self.fetch_thread.isRunning():
            print("Auto-refresh skipped: Fetch in progress")
            return
        self.load_data()

    def on_selection_changed(self):
        selected_items = self.crypto_table.selectedIndexes()
        if selected_items:
            selected_row = selected_items[0].row()
            selected_crypto_id = self.crypto_table.item(selected_row, 0).data(Qt.UserRole)
            self.selected_crypto = selected_crypto_id
            name = self.crypto_table.item(selected_row, 1).text()
            self.analysis_header.setText(f"Analysis for {name}")
            self.load_analysis()

    def load_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            print("Analysis thread already running, skipping...")
            return
        try:
            self.status_label.setText("Analyzing data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.analysis_thread = AnalysisThread(self.selected_crypto)
            self.analysis_thread.analysis_ready.connect(self.update_analysis)
            self.analysis_thread.error.connect(self.show_error)
            self.analysis_thread.finished.connect(self.on_analysis_complete)
            self.analysis_thread.start()
        except Exception as e:
            self.show_error(f"Load analysis error: {str(e)}")

    def update_analysis(self, result):
        try:
            report, risk_score, risk_category, risk_color = result
            formatted_report = report.replace('\n', '<br>')
            self.analysis_text.setText(f"<pre>{formatted_report}</pre>")
            self.update_charts()
            self.status_label.setText("Analysis completed")
        except Exception as e:
            self.show_error(f"Update analysis error: {str(e)}")

    def update_charts(self):
        try:
            for ax in self.axes.flat:
                ax.clear()
            print(f"Fetching historical data for {self.selected_crypto}")
            df = fetch_historical_data(self.selected_crypto)
            if df is None or df.empty:
                print("No historical data returned")
                self.status_label.setText("No historical data available")
                self.canvas.draw()
                return
            print(f"Historical data shape: {df.shape}")

            ax1 = self.axes[0, 0]
            ax1.plot(df.index, df['price'], color='blue', linestyle='-', linewidth=1.5, label='Price')
            ax1.set_title('Price History', fontsize=12)
            ax1.set_xlabel('Date', fontsize=10)
            ax1.set_ylabel('Price (USD)', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()

            ax2 = self.axes[0, 1]
            sns.histplot(data=df['volume'], kde=True, ax=ax2, color='skyblue', edgecolor='black')
            ax2.set_title('Volume Distribution', fontsize=12)
            ax2.set_xlabel('Volume (USD)', fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)

            ax3 = self.axes[1, 0]
            daily_returns = df['price'].pct_change() * 100
            sns.histplot(data=daily_returns.dropna(), kde=True, ax=ax3, color='salmon', edgecolor='black')
            ax3.set_title('Daily Returns Distribution', fontsize=12)
            ax3.set_xlabel('Daily Return (%)', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)

            ax4 = self.axes[1, 1]
            ax4.plot(df.index, df['market_cap'], color='green', linestyle='-', linewidth=1.5, label='Market Cap')
            ax4.set_title('Market Cap History', fontsize=12)
            ax4.set_xlabel('Date', fontsize=10)
            ax4.set_ylabel('Market Cap (USD)', fontsize=10)
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()

            plt.tight_layout()
            self.canvas.draw()
        except Exception as e:
            error_msg = f"Error updating charts: {str(e)}"
            print(error_msg)
            self.show_error(error_msg)

    def on_analysis_complete(self):
        self.analysis_thread = None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Analysis completed at {datetime.now().strftime('%H:%M:%S')}")

    def load_crypto_data(self):
        try:
            print("Starting initial data load...")
            self.refresh_btn.setEnabled(False)
            self.status_label.setText("Fetching initial data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.fetch_thread = DataFetchThread(force_refresh=False)
            self.fetch_thread.data_ready.connect(self.update_table)
            self.fetch_thread.error.connect(self.show_error)
            self.fetch_thread.finished.connect(self.on_fetch_complete)
            self.fetch_thread.start()
        except Exception as e:
            self.show_error(f"Initial data load error: {str(e)}")

    def update_refresh_interval(self, text):
        try:
            interval_map = {"5s": 5000, "10s": 10000, "30s": 30000, "1m": 60000}
            interval = interval_map.get(text, 5000)
            self.refresh_timer.setInterval(interval)
            self.status_label.setText(f"Auto-refresh interval set to {text}")
        except Exception as e:
            self.show_error(f"Error updating refresh interval: {str(e)}")

    def closeEvent(self, event):
        self.refresh_timer.stop()
        if self.fetch_thread and self.fetch_thread.isRunning():
            self.fetch_thread.quit()
            self.fetch_thread.wait()
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        plt.close(self.figure)
        event.accept()


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}, Version: {torch.version.cuda}")
        else:
            print("CUDA not available, using CPU")
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = CryptoAnalyzerGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical Error: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")