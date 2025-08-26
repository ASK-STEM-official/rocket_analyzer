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
            'filter_cutoff': 0.8,  # ローパスフィルタのアルファ値に変更
            'gravity': 9.81,
            'reference_pressure': 1013.25,
            'temperature_lapse': -0.0065,
            'plot_style': 'viridis',
            'animation_speed': 50
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
    
    def calculate_attitude(self, accel_data, gyro_data, dt):
        """姿勢推定（相補フィルタ）"""
        N = len(accel_data)
        attitude = np.zeros((N, 3))  # roll, pitch, yaw
        
        # 初期姿勢を加速度から推定
        initial_acc = np.mean(accel_data[:10], axis=0)
        initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
        
        attitude[0, 0] = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])  # roll
        attitude[0, 1] = np.arctan2(-initial_acc_norm[0], 
                                  np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))  # pitch
        attitude[0, 2] = 0.0  # yaw
        
        alpha = 0.98  # 相補フィルタの係数
        
        for i in range(1, N):
            # ジャイロ積分による姿勢予測
            gyro_rad = np.radians(gyro_data[i])
            predicted = attitude[i-1] + gyro_rad * dt
            
            # 加速度による姿勢推定
            acc_norm = accel_data[i] / (np.linalg.norm(accel_data[i]) + 1e-8)
            roll_acc = np.arctan2(acc_norm[1], acc_norm[2])
            pitch_acc = np.arctan2(-acc_norm[0], np.sqrt(acc_norm[1]**2 + acc_norm[2]**2))
            
            # 相補フィルタで融合
            attitude[i, 0] = alpha * predicted[0] + (1 - alpha) * roll_acc
            attitude[i, 1] = alpha * predicted[1] + (1 - alpha) * pitch_acc
            attitude[i, 2] = predicted[2]  # yawは積分のみ
            
        return attitude
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """オイラー角から回転行列を計算"""
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

    def low_pass_filter(self, data, alpha=0.8):
        """ローパスフィルタ（data1.pyから参考）"""
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
        return filtered

    def calculate_trajectory(self):
        """軌跡計算（data1.pyの手法を参考に改良）"""
        if self.data is None:
            return None
        
        # 時間軸
        time = self.data['Time_s'].values
        dt = np.mean(np.diff(time))
        
        # センサーデータ取得
        accel_cols = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
        gyro_cols = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
        
        # 加速度データを取得し、単位をgからm/s^2に変換
        acc_data = self.data[accel_cols].values * self.config['gravity']
        gyro_data = self.data[gyro_cols].values
        
        # ジャイロデータをラジアンに変換
        gyro_data_rad = np.deg2rad(gyro_data)
        
        # 初期姿勢推定（最初の数秒の加速度から推定）
        initial_samples = min(10, len(acc_data))
        initial_acc = np.mean(acc_data[:initial_samples], axis=0)
        initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
        
        # 初期のロール・ピッチを推定
        initial_roll = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])
        initial_pitch = np.arctan2(-initial_acc_norm[0], 
                                  np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))
        initial_yaw = 0.0  # 初期ヨー角は0と仮定
        
        # 姿勢角の計算（data1.pyの手法）
        N = len(time)
        attitude = np.zeros((N, 3))
        attitude[0] = [initial_roll, initial_pitch, initial_yaw]
        
        # より正確な姿勢推定（ジャイロ積分）
        for i in range(1, N):
            attitude[i] = attitude[i-1] + gyro_data_rad[i] * dt
        
        # 加速度のローパスフィルタリング（ノイズ除去）
        acc_data_filtered = np.zeros_like(acc_data)
        for axis in range(3):
            acc_data_filtered[:, axis] = self.low_pass_filter(acc_data[:, axis])
        
        # 各時刻における加速度をグローバル座標系に変換
        global_acc = np.zeros_like(acc_data_filtered)
        for i in range(N):
            roll = attitude[i, 0]
            pitch = attitude[i, 1]
            yaw = attitude[i, 2]
            R_mat = self.euler_to_rotation_matrix(roll, pitch, yaw)
            global_acc[i] = R_mat @ acc_data_filtered[i]
        
        # 重力加速度を除去（Z軸方向）
        global_acc[:, 2] = global_acc[:, 2] - self.config['gravity']
        
        # 静止時の加速度バイアスを除去（data1.pyの手法）
        static_samples = min(20, len(global_acc))
        bias = np.mean(global_acc[:static_samples], axis=0)
        global_acc = global_acc - bias
        
        # 加速度の積分により速度、さらに速度の積分により位置を計算
        velocity = cumulative_trapezoid(global_acc, dx=dt, initial=0, axis=0)
        position = cumulative_trapezoid(velocity, dx=dt, initial=0, axis=0)
        
        # 気圧高度でZ軸位置を上書き（より信頼性が高い）
        if 'Pressure_hPa' in self.data.columns:
            pressure = self.data['Pressure_hPa'].values
            p0 = pressure[0]  # 基準気圧
            alt_baro = 44330 * (1 - (pressure / p0) ** (1/5.255))
            position[:, 2] = alt_baro - alt_baro[0]  # 高度（初期値をゼロに）
        
        # 水平位置の制限（異常値を防ぐ - analyzer.pyの手法を参考）
        max_altitude = np.max(position[:, 2])
        horizontal_drift_limit = min(max_altitude * 0.5, 100.0)  # 高度の50%または100m以内
        
        # 水平位置を制限
        position[:, 0] = np.clip(position[:, 0], -horizontal_drift_limit, horizontal_drift_limit)
        position[:, 1] = np.clip(position[:, 1], -horizontal_drift_limit, horizontal_drift_limit)
        
        # 速度も制限（物理的に妥当な範囲）
        max_horizontal_velocity = 100.0  # 100m/s以内
        velocity[:, 0] = np.clip(velocity[:, 0], -max_horizontal_velocity, max_horizontal_velocity)
        velocity[:, 1] = np.clip(velocity[:, 1], -max_horizontal_velocity, max_horizontal_velocity)
        
        return {
            'time': time,
            'position': position,
            'velocity': velocity,
            'acceleration': global_acc,
            'attitude': attitude,
            'pressure_altitude': position[:, 2] if 'Pressure_hPa' in self.data.columns else None
        }
    
    def create_interactive_plot(self, save_html=True):
        """インタラクティブな3Dプロット作成（analyzer.pyの手法を採用）"""
        if self.processed_data is None:
            self.processed_data = self.calculate_trajectory()
        
        if self.processed_data is None:
            print("データが処理できませんでした")
            return
        
        pos = self.processed_data['position']
        time = self.processed_data['time']
        vel = self.processed_data['velocity']
        
        # 速度の大きさを計算
        vel_magnitude = np.linalg.norm(vel, axis=1)
        
        # 加速度の大きさを計算
        acc_magnitude = np.linalg.norm(self.processed_data['acceleration'], axis=1)
        
        # 飛行サマリー計算
        max_altitude = np.max(pos[:, 2])
        max_velocity = np.max(vel_magnitude)
        flight_time = time[-1] - time[0]
        horizontal_distance = np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2)
        max_acceleration = np.max(acc_magnitude)
        
        print("\n=== 飛行解析サマリー ===")
        print(f"最大高度: {max_altitude:.2f} m")
        print(f"最大速度: {max_velocity:.2f} m/s")
        print(f"飛行時間: {flight_time:.2f} s")
        print(f"水平移動距離: {horizontal_distance:.2f} m")
        print(f"最大加速度: {max_acceleration:.2f} m/s²")
        
        # カスタムデータを準備（ホバー表示用）
        if 'Temperature_C' in self.data.columns:
            temperature_data = self.data['Temperature_C'].values
        else:
            temperature_data = np.full(len(time), 20.0)  # デフォルト値
            
        if 'Humidity_%' in self.data.columns:
            humidity_data = self.data['Humidity_%'].values
        else:
            humidity_data = np.full(len(time), 50.0)  # デフォルト値
            
        if 'Pressure_hPa' in self.data.columns:
            pressure_data = self.data['Pressure_hPa'].values
        else:
            pressure_data = np.full(len(time), 1013.25)  # デフォルト値
        
        custom_data = np.column_stack([
            acc_magnitude,
            temperature_data,
            humidity_data,
            pressure_data,
            time,
            vel_magnitude
        ])
        
        # 3D軌跡トレースの作成（analyzer.pyの手法を採用）
        flight_path_trace = go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode='lines+markers',
            line=dict(
                color=acc_magnitude,
                colorscale='Plasma',  # より鮮やかなカラーマップ
                width=6,
                colorbar=dict(
                    title=dict(text='加速度 (m/s²)', font=dict(size=14)),
                    tickfont=dict(size=12)
                )
            ),
            marker=dict(size=3, color=acc_magnitude, colorscale='Plasma'),
            name='飛行軌跡',
            customdata=custom_data,
            hovertemplate='<b>飛行データ</b><br>' +
                          '時刻: %{customdata[4]:.2f} s<br>' +
                          '加速度: %{customdata[0]:.2f} m/s²<br>' +
                          '速度: %{customdata[5]:.2f} m/s<br>' +
                          '温度: %{customdata[1]:.1f} °C<br>' +
                          '湿度: %{customdata[2]:.1f} %<br>' +
                          '気圧: %{customdata[3]:.1f} hPa<br>' +
                          'X: %{x:.2f} m<br>' +
                          'Y: %{y:.2f} m<br>' +
                          'Z: %{z:.2f} m<extra></extra>'
        )
        
        # サマリー情報を注釈として追加
        summary_text = (f"最大高度: {max_altitude:.2f} m<br>"
                       f"最大速度: {max_velocity:.2f} m/s<br>"
                       f"飛行時間: {flight_time:.2f} s<br>"
                       f"水平距離: {horizontal_distance:.2f} m<br>"
                       f"最大加速度: {max_acceleration:.2f} m/s²")
        
        # 単一の3Dプロットとして作成（analyzer.pyの手法）
        fig = go.Figure(
            data=[flight_path_trace],
            layout=go.Layout(
                title=dict(
                    text="ロケット飛行軌跡解析（改良版）",
                    font=dict(size=20)
                ),
                scene=dict(
                    xaxis=dict(title='X (m)'),
                    yaxis=dict(title='Y (m)'),
                    zaxis=dict(title='Z (m)'),
                    aspectmode='data'
                ),
                annotations=[
                    dict(
                        text=summary_text,
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor="left", yanchor="top",
                        font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                ],
                width=1200,
                height=800
            )
        )
        
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
        
        ttk.Label(settings_frame, text="ローパスフィルタ係数:").grid(row=0, column=0)
        self.cutoff_var = tk.DoubleVar(value=self.analyzer.config['filter_cutoff'])
        ttk.Scale(settings_frame, from_=0.1, to=0.95, variable=self.cutoff_var, 
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
