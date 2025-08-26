import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid
import os

def estimate_altitude(data, target_pressure):
    """
    data: [(標高, 気圧), ...] のリスト（標高は m、気圧は hPa）
    target_pressure: 推定したい気圧 (hPa)
    return: 推定高度 (m)
    """
    # 気圧の降順（高→低）でソート
    data_sorted = sorted(data, key=lambda x: x[1], reverse=True)

    # データ点を順に見て、target_pressure が含まれる区間を探す
    for i in range(len(data_sorted) - 1):
        h1, p1 = data_sorted[i]
        h2, p2 = data_sorted[i+1]

        # 区間内の場合
        if p1 >= target_pressure >= p2:
            return h1 + (p1 - target_pressure) / (p1 - p2) * (h2 - h1)

    # 範囲外 → 端の2点で外挿する
    h1, p1 = data_sorted[0]
    h2, p2 = data_sorted[-1]
    return h1 + (p1 - target_pressure) / (p1 - p2) * (h2 - h1)

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    return R

def process_sensor_data_corrected(df, reference_data=None):
    """
    reference_data: 当日の気圧観測データ [(標高, 気圧), ...]
    デフォルトでは提供されたデータを使用
    """
    if reference_data is None:
        # デフォルトの観測データ
        reference_data = [
            (33.6, 1012.29),
            (51.0, 1009.00),
        ]
    
    print(f"データ件数: {len(df)}")
    print(f"気圧参照データ: {reference_data}")
    
    # --- 改善された気圧高度計算（相対高度に変更） ---
    pressure = df['Pressure_hPa'].to_numpy()
    
    # 各気圧値に対して実測データから絶対標高を推定
    alt_absolute = np.array([estimate_altitude(reference_data, p) for p in pressure])
    
    # 最初の地点の標高を0とした相対高度に変換
    initial_altitude = alt_absolute[0]
    alt_corrected = alt_absolute - initial_altitude
    
    # 従来の国際標準大気式も計算（比較用、相対高度）
    P0 = pressure[0]
    alt_standard_absolute = 44330 * (1 - (pressure / P0) ** (1/5.255))
    alt_standard = alt_standard_absolute - alt_standard_absolute[0]
    
    print(f"高度計算結果（相対高度）:")
    print(f"  開始地点の絶対標高: {initial_altitude:.2f} m")
    print(f"  補正高度範囲: {alt_corrected.min():.2f} - {alt_corrected.max():.2f} m")
    print(f"  標準大気高度範囲: {alt_standard.min():.2f} - {alt_standard.max():.2f} m")
    print(f"  高度差（最大）: {np.max(np.abs(alt_corrected - alt_standard)):.2f} m")
    
    # 以降は補正された相対高度を使用
    alt_pressure = alt_corrected
    
    # --- 既存の処理（加速度・姿勢など） ---
    acc_columns = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
    gyro_columns = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
    time = df['Time_s'].values
    dt = np.mean(np.diff(time))
    
    # イベントログ処理
    event_log_path = os.path.join('event_log', 'event_log2.txt')
    events = []
    flight_start_idx = None
    descent_idx = None
    parachute_idx = None
    
    if os.path.exists(event_log_path):
        try:
            with open(event_log_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    tstr, msg = line.split(':', 1)
                    try:
                        t = float(tstr)
                    except ValueError:
                        continue
                    events.append((t, msg.strip()))
            
            for t, msg in events:
                idx = int(np.argmin(np.abs(time - t)))
                if 'Flight start' in msg and flight_start_idx is None:
                    flight_start_idx = idx
                elif '下降検知' in msg or 'descent' in msg and descent_idx is None:
                    descent_idx = idx
                elif 'Parachute' in msg or 'parachute' in msg and parachute_idx is None:
                    parachute_idx = idx
                    
            print(f"主要イベント: flight_start={flight_start_idx}, descent={descent_idx}, parachute={parachute_idx}")
        except Exception as e:
            print(f"イベントログ読み込み失敗: {e}")
    
    # 加速度・ジャイロデータ処理
    acc_data = df[acc_columns].to_numpy() * 9.81
    gyro_data = df[gyro_columns].to_numpy()
    
    initial_samples = min(10, len(acc_data))
    initial_acc = np.mean(acc_data[:initial_samples], axis=0)
    initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
    initial_roll = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])
    initial_pitch = np.arctan2(-initial_acc_norm[0], np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))
    initial_yaw = 0.0
    
    gyro_data_rad = np.deg2rad(gyro_data)
    N = len(time)

    def low_pass_filter(data, alpha=0.8):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
        return filtered

    # 姿勢推定
    acc_data_filtered = np.zeros_like(acc_data)
    for axis in range(3):
        acc_data_filtered[:, axis] = low_pass_filter(acc_data[:, axis])

    attitude = np.zeros((N, 3))
    attitude[0] = [initial_roll, initial_pitch, initial_yaw]
    cf_alpha = 0.98
    
    for i in range(1, N):
        pred = attitude[i-1] + gyro_data_rad[i] * dt
        a = acc_data_filtered[i]
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        roll_a = np.arctan2(a_norm[1], a_norm[2])
        pitch_a = np.arctan2(-a_norm[0], np.sqrt(a_norm[1]**2 + a_norm[2]**2))
        yaw_pred = pred[2]
        
        attitude[i, 0] = cf_alpha * pred[0] + (1 - cf_alpha) * roll_a
        attitude[i, 1] = cf_alpha * pred[1] + (1 - cf_alpha) * pitch_a
        attitude[i, 2] = yaw_pred

    # 地球座標系変換
    global_acc = np.zeros_like(acc_data_filtered)
    for i in range(N):
        roll, pitch, yaw = attitude[i]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data_filtered[i]
    
    global_acc[:, 2] = global_acc[:, 2] - 9.81

    # 静止区間判定
    if 'FlightStarted' in df.columns:
        static_mask = df['FlightStarted'] == 0
    else:
        static_mask = np.ones(len(df), dtype=bool)
        if flight_start_idx is not None:
            static_mask[flight_start_idx:] = False

    # バイアス補正
    if np.any(static_mask):
        bias = np.mean(global_acc[static_mask], axis=0)
    else:
        bias = np.mean(global_acc[:20], axis=0)
    global_acc = global_acc - bias

    # --- 改善された3D位置・速度計算 ---
    position = np.zeros((N, 3))
    velocity_3d = np.zeros((N, 3))
    
    # Z軸：補正された相対高度を使用
    position[:, 2] = alt_pressure
    
    # Z軸速度：相対高度の変化率
    velocity_3d[:, 2] = np.gradient(alt_pressure, time)
    
    # X/Y軸：加速度積分 + 姿勢補正
    for axis in range(2):
        # 基本的な加速度積分
        vel_integrated = cumulative_trapezoid(global_acc[:, axis], time, initial=0)
        pos_integrated = cumulative_trapezoid(vel_integrated, time, initial=0)
        
        # 静止区間リセット
        vel_integrated[static_mask] = 0.0
        pos_integrated[static_mask] = 0.0
        
        # ドリフト除去
        vel_lp = low_pass_filter(vel_integrated, alpha=0.9)
        vel_hp = vel_integrated - vel_lp
        vel_hp = np.clip(vel_hp, -15.0, 15.0)
        
        # 姿勢による補正（推力方向考慮）
        if flight_start_idx is not None and descent_idx is not None:
            flight_indices = range(flight_start_idx, min(descent_idx, len(time)))
            
            # 飛行中の姿勢補正
            for i in flight_indices:
                # ピッチ角による水平推力成分
                pitch_effect = np.sin(attitude[i, 1]) * 0.5  # 補正係数
                if axis == 0:  # X軸
                    vel_hp[i] += pitch_effect * np.cos(attitude[i, 2])
                else:  # Y軸
                    vel_hp[i] += pitch_effect * np.sin(attitude[i, 2])
        
        # 最終的な位置計算
        position[:, axis] = cumulative_trapezoid(vel_hp, time, initial=0)
        velocity_3d[:, axis] = vel_hp

    # 速度の大きさ
    speed = np.linalg.norm(velocity_3d, axis=1)
    
    # 加速度の大きさ
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)

    print(f"最終結果（相対座標系）:")
    print(f"  位置範囲: X={position[:,0].min():.2f}~{position[:,0].max():.2f}, Y={position[:,1].min():.2f}~{position[:,1].max():.2f}, Z={position[:,2].min():.2f}~{position[:,2].max():.2f}")
    print(f"  最大スピード: {speed.max():.2f}m/s")
    print(f"  総水平飛行距離: {np.linalg.norm(position[-1, :2]):.1f}m")
    print(f"  最大到達高度: {position[:,2].max():.1f}m")
    
    return {
        'position': position,
        'velocity_3d': velocity_3d,
        'speed': speed,
        'acc_magnitude': acc_magnitude,
        'attitude': attitude,
        'altitude_corrected': alt_pressure,
        'altitude_standard': alt_standard,
        'initial_altitude': initial_altitude,
        'temperature': df['Temperature_C'].to_numpy(),
        'humidity': df['Humidity_%'].to_numpy(),
        'pressure': df['Pressure_hPa'].to_numpy(),
        'time': time
    }

def plot_corrected_trajectory(data):
    if data is None:
        print("データ処理に失敗しました")
        return
    
    pos = data['position']
    vel = data['velocity_3d']
    speed = data['speed']
    time = data['time']
    attitude = data['attitude']
    
    # サブプロット設定
    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{"type": "scatter3d", "colspan": 2}, None],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter", "colspan": 2}, None]],
        subplot_titles=('補正された飛行軌跡 3D（相対高度）', '高度比較（相対）', 'スピード変化', 
                       '水平位置変化', '姿勢角変化', '速度成分変化'),
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # 3D軌跡
    flight_path = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='lines',
        line=dict(color=speed, colorscale='Viridis', width=6),
        name='補正軌跡（相対）'
    )
    fig.add_trace(flight_path, row=1, col=1)
    
    # 開始点をマーク
    start_point = go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
        mode='markers',
        marker=dict(color='red', size=8),
        name='開始点（高度0）'
    )
    fig.add_trace(start_point, row=1, col=1)
    
    # 高度比較（補正 vs 標準、両方とも相対高度）
    fig.add_trace(go.Scatter(x=time, y=data['altitude_corrected'], 
                            name='補正相対高度', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=data['altitude_standard'], 
                            name='標準大気相対高度', line=dict(color='red', dash='dash')), row=2, col=1)
    
    # スピード
    fig.add_trace(go.Scatter(x=time, y=speed, name='スピード', 
                            line=dict(color='green')), row=2, col=2)
    
    # 水平位置（開始点が原点）
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode='lines', 
                            name='水平軌跡', line=dict(color='purple')), row=3, col=1)
    # 開始点をマーク
    fig.add_trace(go.Scatter(x=[pos[0, 0]], y=[pos[0, 1]], mode='markers',
                            marker=dict(color='red', size=8), name='開始点', showlegend=False), row=3, col=1)
    
    # 姿勢角
    fig.add_trace(go.Scatter(x=time, y=np.rad2deg(attitude[:, 1]), 
                            name='ピッチ', line=dict(color='orange')), row=3, col=2)
    fig.add_trace(go.Scatter(x=time, y=np.rad2deg(attitude[:, 0]), 
                            name='ロール', line=dict(color='cyan')), row=3, col=2)
    
    # 速度成分
    fig.add_trace(go.Scatter(x=time, y=vel[:, 0], name='Vx', 
                            line=dict(color='red')), row=4, col=1)
    fig.add_trace(go.Scatter(x=time, y=vel[:, 1], name='Vy', 
                            line=dict(color='green')), row=4, col=1)
    fig.add_trace(go.Scatter(x=time, y=vel[:, 2], name='Vz', 
                            line=dict(color='blue')), row=4, col=1)
    
    # タイトル更新（相対高度を明記）
    title = f'気圧補正を適用したロケット軌道解析（相対高度）<br>開始地点標高: {data["initial_altitude"]:.1f}m, 最大到達高度: {pos[:,2].max():.1f}m'
    fig.update_layout(title=title, height=1200)
    
    # 軸ラベル
    fig.update_scenes(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='相対高度 (m)')
    fig.update_xaxes(title_text='時間 (s)', row=2, col=1)
    fig.update_yaxes(title_text='相対高度 (m)', row=2, col=1)
    fig.update_xaxes(title_text='時間 (s)', row=2, col=2)
    fig.update_yaxes(title_text='スピード (m/s)', row=2, col=2)
    fig.update_xaxes(title_text='X (m)', row=3, col=1)
    fig.update_yaxes(title_text='Y (m)', row=3, col=1)
    fig.update_xaxes(title_text='時間 (s)', row=3, col=2)
    fig.update_yaxes(title_text='角度 (°)', row=3, col=2)
    fig.update_xaxes(title_text='時間 (s)', row=4, col=1)
    fig.update_yaxes(title_text='速度 (m/s)', row=4, col=1)

    fig.write_html("corrected_trajectory_relative.html")
    print("corrected_trajectory_relative.html をブラウザで開いてください")

if __name__ == "__main__":
    # カスタム気圧データがある場合は指定
    custom_reference = [
         (33.6, 1012.29),
         (51.0, 1009.00),
        # 追加のデータポイントがあれば追加
    ]
    
    #df = pd.read_csv("sensor.csv")
    df = pd.read_csv("rocket_flight_data_realistic_1756199645046.csv")
    data = process_sensor_data_corrected(df, custom_reference)
    plot_corrected_trajectory(data)