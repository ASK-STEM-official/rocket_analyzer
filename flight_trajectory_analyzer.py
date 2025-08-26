import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
import os
import json
from datetime import datetime

class FlightAnalyzer:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.config = self.load_config()
        
    def load_config(self):
        """設定を読み込み"""
        default_config = {
            'filter_cutoff': 5.0,  # 加速度のカットオフ周波数
            'gravity': 9.81,
            'reference_pressure': 1013.25,
            'temperature_lapse': -0.0065,
            'plot_style': 'viridis',
            'animation_speed': 50,
            'complementary_alpha': 0.96,  # 相補フィルタのパラメータ
            'altitude_fusion_weight': 0.7  # 気圧高度の融合重み
        }
        
        config_path = 'analyzer_config.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return {**default_config, **config}
            except:
                pass
        return default_config
    
    def save_config(self):
        """設定を保存"""
        with open('analyzer_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_data(self, filepath):
        """センサーデータを読み込み"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"データ読み込み成功: {len(self.data)} 行")
            return True
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return False
    
    def load_event_log(self, event_log_path=None):
        """イベントログを読み込み"""
        if event_log_path is None:
            event_log_path = os.path.join('event_log', 'event_log2.txt')
        
        events = []
        if os.path.exists(event_log_path):
            try:
                with open(event_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if ':' in line:
                            time_str, event = line.split(':', 1)
                            try:
                                time_val = float(time_str)
                                events.append((time_val, event.strip()))
                            except ValueError:
                                continue
                print(f"イベントログ読み込み: {len(events)} イベント")
                return events
            except Exception as e:
                print(f"イベントログ読み込みエラー: {e}")
        return []
    
    def butter_lowpass_filter(self, data, cutoff, fs, order=4):
        """バターワースローパスフィルタ"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    def calculate_attitude_complementary(self, accel_data, gyro_data, dt):
        """相補フィルタによる姿勢推定"""
        N = len(accel_data)
        attitude = np.zeros((N, 3))  # roll, pitch, yaw
        
        # 初期姿勢を加速度から推定
        initial_acc = np.mean(accel_data[:5], axis=0)  # 最初の5点の平均
        initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
        
        attitude[0, 0] = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])  # roll
        attitude[0, 1] = np.arctan2(-initial_acc_norm[0], 
                                  np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))  # pitch
        attitude[0, 2] = 0.0  # yaw（初期値）
        
        # 相補フィルタのパラメータ
        alpha = 0.96  # ジャイロの信頼度（高周波特性が良い）
        beta = 1 - alpha  # 加速度の信頼度（低周波特性が良い）
        
        for i in range(1, N):
            # ジャイロによる予測（角速度積分）
            gyro_rad = np.radians(gyro_data[i])
            predicted = attitude[i-1] + gyro_rad * dt[i-1]
            
            # 加速度による修正（重力ベクトルから姿勢推定）
            acc_norm = accel_data[i] / (np.linalg.norm(accel_data[i]) + 1e-8)
            
            # 加速度から roll, pitch を計算
            roll_acc = np.arctan2(acc_norm[1], acc_norm[2])
            pitch_acc = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            
            # 相補フィルタで融合
            attitude[i, 0] = alpha * predicted[0] + beta * roll_acc
            attitude[i, 1] = alpha * predicted[1] + beta * pitch_acc
            attitude[i, 2] = predicted[2]  # yawはジャイロ積分のみ（磁力計がないため）
            
            # 角度の正規化（-π to π）
            for j in range(3):
                while attitude[i, j] > np.pi:
                    attitude[i, j] -= 2 * np.pi
                while attitude[i, j] < -np.pi:
                    attitude[i, j] += 2 * np.pi
        
        return attitude
    
    def rotation_matrix_from_euler(self, roll, pitch, yaw):
        """オイラー角から回転行列を作成"""
        # ZYX順の回転行列
        R_x = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def calculate_trajectory(self):
        """改良された軌跡計算（ジャイロ+加速度融合）"""
        if self.data is None:
            return None
        
        # 時間軸
        time = self.data['Time_s'].values
        dt = np.diff(time)
        dt = np.append(dt, dt[-1])
        fs = 1.0 / np.mean(dt)
        
        # センサーデータ取得
        accel_cols = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
        gyro_cols = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
        
        accel_data = self.data[accel_cols].values * self.config['gravity']
        gyro_data = self.data[gyro_cols].values
        
        # センサーデータのフィルタリング
        accel_filtered = np.zeros_like(accel_data)
        gyro_filtered = np.zeros_like(gyro_data)
        
        for i in range(3):
            accel_filtered[:, i] = self.butter_lowpass_filter(
                accel_data[:, i], self.config['filter_cutoff'], fs
            )
            gyro_filtered[:, i] = self.butter_lowpass_filter(
                gyro_data[:, i], 1.0, fs  # ジャイロは少し高めのカットオフ
            )
        
        # 静止状態の検出
        static_samples = min(20, len(accel_data) // 10)
        static_accel_mag = np.linalg.norm(accel_filtered[:static_samples], axis=1)
        static_threshold = self.config['gravity'] * 1.2
        static_indices = static_accel_mag < static_threshold
        
        # バイアス補正
        if np.any(static_indices):
            accel_bias = np.mean(accel_filtered[:static_samples][static_indices], axis=0)
            gyro_bias = np.mean(gyro_filtered[:static_samples][static_indices], axis=0)
            
            # 重力方向の補正
            gravity_vector = np.array([0, 0, self.config['gravity']])
            accel_bias -= gravity_vector
            
            accel_filtered -= accel_bias
            gyro_filtered -= gyro_bias
        
        # 相補フィルタによる姿勢推定
        attitude = self.calculate_attitude_complementary(accel_filtered, gyro_filtered, dt)
        
        # 座標変換（ボディ座標 → グローバル座標）
        # 水平方向のみ姿勢補正を適用、Z軸は後で気圧データで上書き
        global_accel = np.zeros_like(accel_filtered)
        
        for i in range(len(accel_filtered)):
            # 姿勢が安定している場合のみ座標変換を適用
            attitude_magnitude = np.sqrt(attitude[i, 0]**2 + attitude[i, 1]**2)
            if attitude_magnitude < np.radians(45):  # 45度以下の場合のみ変換
                R = self.rotation_matrix_from_euler(attitude[i, 0], attitude[i, 1], attitude[i, 2])
                global_accel[i] = R @ accel_filtered[i]
            else:
                # 姿勢が不安定な場合は変換なし
                global_accel[i] = accel_filtered[i]
        
        # 重力補正（Z軸のみ）
        global_accel[:, 2] -= self.config['gravity']
        
        # 適応的積分（水平方向のみ、高度は気圧データ使用）
        velocity = np.zeros_like(global_accel)
        position = np.zeros_like(global_accel)
        
        # X, Y軸の積分計算（水平方向）
        for i in range(1, len(global_accel)):
            # 水平方向の速度積分
            velocity[i, 0] = velocity[i-1, 0] + 0.5 * (global_accel[i-1, 0] + global_accel[i, 0]) * dt[i-1]
            velocity[i, 1] = velocity[i-1, 1] + 0.5 * (global_accel[i-1, 1] + global_accel[i, 1]) * dt[i-1]
            
            # 水平方向の位置積分
            position[i, 0] = position[i-1, 0] + 0.5 * (velocity[i-1, 0] + velocity[i, 0]) * dt[i-1]
            position[i, 1] = position[i-1, 1] + 0.5 * (velocity[i-1, 1] + velocity[i, 1]) * dt[i-1]
        
        # 気圧高度による Z軸（高度は気圧データを優先）
        if 'Pressure_hPa' in self.data.columns:
            pressure = self.data['Pressure_hPa'].values
            p0 = pressure[0]
            alt_baro = 44330 * (1 - (pressure / p0) ** (1/5.255))
            
            # 気圧高度をそのまま使用（最も信頼できる）
            position[:, 2] = alt_baro - alt_baro[0]  # 初期高度をゼロに正規化
            
            # 高度から速度を数値微分で計算
            for i in range(1, len(velocity)):
                velocity[i, 2] = (position[i, 2] - position[i-1, 2]) / dt[i-1]
        
        # 水平方向（X, Y軸）のみセンサー融合を適用
        # Z軸の加速度データは使わず、気圧高度のみを信頼
        
        # ドリフト補正（水平方向のみ適用）
        # 最後の20%のデータで着陸判定
        landing_start = int(len(velocity) * 0.8)
        if landing_start < len(velocity):
            # 水平速度が小さい場合はドリフト補正
            horizontal_vel_mag = np.sqrt(velocity[landing_start:, 0]**2 + velocity[landing_start:, 1]**2)
            if np.mean(horizontal_vel_mag) < 5.0:  # 平均水平速度が5m/s以下の場合
                # 水平方向の速度を徐々にゼロに収束
                for i in range(landing_start, len(velocity)):
                    decay_factor = 1.0 - 0.5 * (i - landing_start) / (len(velocity) - landing_start)
                    velocity[i, 0] *= decay_factor  # X方向
                    velocity[i, 1] *= decay_factor  # Y方向
                    # Z方向（高度方向）はそのまま（気圧データを信頼）
                
                # 水平位置も再計算
                for i in range(landing_start + 1, len(position)):
                    position[i, 0] = position[i-1, 0] + 0.5 * (velocity[i-1, 0] + velocity[i, 0]) * dt[i-1]
                    position[i, 1] = position[i-1, 1] + 0.5 * (velocity[i-1, 1] + velocity[i, 1]) * dt[i-1]
                    # Z方向は気圧データのまま変更しない
        
        # 異常値の制限（水平方向のみ）
        max_velocity = 100  # 最大速度 100m/s（現実的な値）
        max_horizontal_distance = 300  # 最大水平距離 300m
        
        # X, Y軸のみ制限（Z軸は気圧データを信頼）
        velocity[:, 0] = np.clip(velocity[:, 0], -max_velocity, max_velocity)
        velocity[:, 1] = np.clip(velocity[:, 1], -max_velocity, max_velocity)
        position[:, 0] = np.clip(position[:, 0], -max_horizontal_distance, max_horizontal_distance)
        position[:, 1] = np.clip(position[:, 1], -max_horizontal_distance, max_horizontal_distance)
        
        return {
            'time': time,
            'position': position,
            'velocity': velocity,
            'acceleration': global_accel,
            'attitude': attitude,
            'pressure_altitude': position[:, 2] if 'Pressure_hPa' in self.data.columns else None
        }
    
    def create_interactive_plot(self, save_html=True):
        """インタラクティブな3Dプロット作成"""
        if self.processed_data is None:
            self.processed_data = self.calculate_trajectory()
        
        if self.processed_data is None:
            print("データが処理できませんでした")
            return
        
        pos = self.processed_data['position']
        time = self.processed_data['time']
        
        # 速度の大きさを計算
        vel_magnitude = np.linalg.norm(self.processed_data['velocity'], axis=1)
        
        # メインの3D軌跡プロット
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d", "colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles=('飛行軌跡 3D', '高度変化', '速度変化'),
            vertical_spacing=0.1
        )
        
        # 3D軌跡
        fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                mode='lines+markers',
                line=dict(color=vel_magnitude, colorscale='Plasma', width=4),
                marker=dict(size=2, color=vel_magnitude, colorscale='Plasma'),
                name='軌跡',
                text=[f'時刻: {t:.2f}s<br>速度: {v:.2f}m/s' for t, v in zip(time, vel_magnitude)],
                hovertemplate='X: %{x:.2f}m<br>Y: %{y:.2f}m<br>Z: %{z:.2f}m<br>%{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 高度変化
        fig.add_trace(
            go.Scatter(
                x=time, y=pos[:, 2],
                mode='lines',
                line=dict(color='blue', width=2),
                name='高度'
            ),
            row=2, col=1
        )
        
        # 速度変化
        fig.add_trace(
            go.Scatter(
                x=time, y=vel_magnitude,
                mode='lines',
                line=dict(color='red', width=2),
                name='速度'
            ),
            row=2, col=2
        )
        
        # レイアウト設定
        fig.update_layout(
            title='ロケット飛行解析 - 詳細データ',
            height=800,
            showlegend=True
        )
        
        fig.update_scenes(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='高度 (m)',
            aspectmode='data'
        )
        
        fig.update_xaxes(title_text='時間 (s)', row=2, col=1)
        fig.update_yaxes(title_text='高度 (m)', row=2, col=1)
        fig.update_xaxes(title_text='時間 (s)', row=2, col=2)
        fig.update_yaxes(title_text='速度 (m/s)', row=2, col=2)
        
        if save_html:
            output_file = f'flight_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            fig.write_html(output_file)
            print(f"解析結果を保存しました: {output_file}")
        
        return fig
    
    def create_summary_report(self):
        """飛行サマリーレポート生成"""
        if self.processed_data is None:
            return None
        
        pos = self.processed_data['position']
        vel = self.processed_data['velocity']
        time = self.processed_data['time']
        
        max_altitude = np.max(pos[:, 2])
        max_velocity = np.max(np.linalg.norm(vel, axis=1))
        flight_time = time[-1] - time[0]
        landing_distance = np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2)
        
        report = {
            'max_altitude': max_altitude,
            'max_velocity': max_velocity,
            'flight_time': flight_time,
            'landing_distance': landing_distance,
            'launch_angle': np.degrees(np.arctan2(pos[10, 2] - pos[0, 2], 
                                                np.sqrt((pos[10, 0] - pos[0, 0])**2 + 
                                                       (pos[10, 1] - pos[0, 1])**2)))
        }
        
        return report

class FlightAnalyzerGUI:
    def __init__(self):
        self.analyzer = FlightAnalyzer()
        self.root = tk.Tk()
        self.root.title("ロケット飛行軌跡解析ソフト v2.0")
        self.root.geometry("800x600")
        self.setup_gui()
    
    def setup_gui(self):
        """GUI設定"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ファイル選択
        file_frame = ttk.LabelFrame(main_frame, text="データファイル", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(file_frame, text="CSVファイルを選択", 
                  command=self.select_file).grid(row=0, column=0, padx=5)
        
        self.file_label = ttk.Label(file_frame, text="ファイルが選択されていません")
        self.file_label.grid(row=0, column=1, padx=5)
        
        # 解析設定
        settings_frame = ttk.LabelFrame(main_frame, text="解析設定", padding="5")
        settings_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(settings_frame, text="フィルタカットオフ周波数:").grid(row=0, column=0)
        self.cutoff_var = tk.DoubleVar(value=self.analyzer.config['filter_cutoff'])
        ttk.Scale(settings_frame, from_=0.1, to=2.0, variable=self.cutoff_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1)
        
        # 実行ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="軌跡解析実行", 
                  command=self.run_analysis).grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="アニメーション表示", 
                  command=self.show_animation).grid(row=0, column=1, padx=5)
        
        ttk.Button(button_frame, text="レポート生成", 
                  command=self.generate_report).grid(row=0, column=2, padx=5)
        
        # 結果表示
        self.result_text = tk.Text(main_frame, height=20, width=80)
        self.result_text.grid(row=3, column=0, columnspan=2, pady=5)
        
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=3, column=2, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
    
    def select_file(self):
        """ファイル選択"""
        filename = filedialog.askopenfilename(
            title="CSVファイルを選択",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            if self.analyzer.load_data(filename):
                self.file_label.config(text=os.path.basename(filename))
                self.result_text.insert(tk.END, f"データ読み込み完了: {filename}\n")
                self.result_text.insert(tk.END, f"データ行数: {len(self.analyzer.data)}\n\n")
            else:
                messagebox.showerror("エラー", "ファイルの読み込みに失敗しました")
    
    def run_analysis(self):
        """解析実行"""
        if self.analyzer.data is None:
            messagebox.showwarning("警告", "データファイルを選択してください")
            return
        
        self.analyzer.config['filter_cutoff'] = self.cutoff_var.get()
        self.analyzer.save_config()
        
        self.result_text.insert(tk.END, "軌跡解析を開始...\n")
        self.root.update()
        
        try:
            fig = self.analyzer.create_interactive_plot()
            
            # サマリー生成
            report = self.analyzer.create_summary_report()
            if report:
                self.result_text.insert(tk.END, "\n=== 飛行サマリー ===\n")
                self.result_text.insert(tk.END, f"最大高度: {report['max_altitude']:.2f} m\n")
                self.result_text.insert(tk.END, f"最大速度: {report['max_velocity']:.2f} m/s\n")
                self.result_text.insert(tk.END, f"飛行時間: {report['flight_time']:.2f} s\n")
                self.result_text.insert(tk.END, f"着地距離: {report['landing_distance']:.2f} m\n")
                self.result_text.insert(tk.END, f"推定打ち上げ角: {report['launch_angle']:.1f} °\n\n")
            
            self.result_text.insert(tk.END, "解析完了！HTMLファイルが生成されました。\n")
            
        except Exception as e:
            messagebox.showerror("エラー", f"解析中にエラーが発生しました: {str(e)}")
            self.result_text.insert(tk.END, f"エラー: {str(e)}\n")
    
    def show_animation(self):
        """アニメーション表示"""
        if self.analyzer.processed_data is None:
            messagebox.showwarning("警告", "先に軌跡解析を実行してください")
            return
        
        self.result_text.insert(tk.END, "アニメーションを準備中...\n")
        # アニメーション機能は次のバージョンで実装予定
        messagebox.showinfo("情報", "アニメーション機能は開発中です")
    
    def generate_report(self):
        """詳細レポート生成"""
        if self.analyzer.processed_data is None:
            messagebox.showwarning("警告", "先に軌跡解析を実行してください")
            return
        
        # レポートをテキストファイルに出力
        report_file = f'flight_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ロケット飛行解析レポート\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            report = self.analyzer.create_summary_report()
            if report:
                f.write("飛行パフォーマンス:\n")
                f.write(f"  最大高度: {report['max_altitude']:.2f} m\n")
                f.write(f"  最大速度: {report['max_velocity']:.2f} m/s\n")
                f.write(f"  飛行時間: {report['flight_time']:.2f} s\n")
                f.write(f"  着地距離: {report['landing_distance']:.2f} m\n")
                f.write(f"  推定打ち上げ角: {report['launch_angle']:.1f} °\n")
        
        self.result_text.insert(tk.END, f"詳細レポートを生成しました: {report_file}\n")
    
    def run(self):
        """GUI実行"""
        self.root.mainloop()

def main():
    """メイン関数"""
    print("ロケット飛行軌跡解析ソフト v2.0")
    print("=" * 40)
    
    # コマンドライン引数でファイル指定があればCUI モード
    import sys
    if len(sys.argv) > 1:
        analyzer = FlightAnalyzer()
        if analyzer.load_data(sys.argv[1]):
            fig = analyzer.create_interactive_plot()
            report = analyzer.create_summary_report()
            if report:
                print("\n飛行サマリー:")
                print(f"最大高度: {report['max_altitude']:.2f} m")
                print(f"最大速度: {report['max_velocity']:.2f} m/s")
                print(f"飛行時間: {report['flight_time']:.2f} s")
                print(f"着地距離: {report['landing_distance']:.2f} m")
    else:
        # GUI モード
        app = FlightAnalyzerGUI()
        app.run()

if __name__ == "__main__":
    main()
