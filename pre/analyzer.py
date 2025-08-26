
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
from datetime import datetime
import os

# オイラー角（roll, pitch, yaw）から回転行列を計算する関数
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

# --- データ処理部 ---
def process_sensor_data(df):
    # --- 気圧から高度を計算（国際標準大気式） ---
    P0 = 1013.25  # 海面気圧[hPa]
    pressure = df['Pressure_hPa'].to_numpy()
    alt_pressure = 44330 * (1 - (pressure / P0) ** (1/5.255))
    print(f"データ件数: {len(df)}")
    acc_columns = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
    gyro_columns = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
    acc_data = df[acc_columns].to_numpy() * 9.81
    gyro_data = df[gyro_columns].to_numpy()
    initial_samples = min(10, len(acc_data))
    initial_acc = np.mean(acc_data[:initial_samples], axis=0)
    initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
    print(f"データ件数: {len(df)}")
    acc_columns = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
    gyro_columns = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
    time = df['Time_s'].values
    dt = np.mean(np.diff(time))
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
    attitude = np.zeros((N, 3))
    attitude[0] = [initial_roll, initial_pitch, initial_yaw]
    for i in range(1, N):
        attitude[i] = attitude[i-1] + gyro_data_rad[i] * dt
    def low_pass_filter(data, alpha=0.8):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
        return filtered
    acc_data_filtered = np.zeros_like(acc_data)
    for axis in range(3):
        acc_data_filtered[:, axis] = low_pass_filter(acc_data[:, axis])
    global_acc = np.zeros_like(acc_data_filtered)
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data_filtered[i]
    global_acc[:, 2] = global_acc[:, 2] - 9.81
    # --- バイアス算出区間を静止区間（FlightStarted==0）に限定 ---
    if 'FlightStarted' in df.columns:
        static_mask = df['FlightStarted'] == 0
        if np.any(static_mask):
            bias = np.mean(global_acc[static_mask], axis=0)
        else:
            bias = np.mean(global_acc[:20], axis=0)
    else:
        bias = np.mean(global_acc[:20], axis=0)
    global_acc = global_acc - bias
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)
    velocity = cumulative_trapezoid(global_acc, dx=dt, initial=0, axis=0)
    position = cumulative_trapezoid(velocity, dx=dt, initial=0, axis=0)
    # --- Z軸はGPS高度(Alt_m)が使える場合はそちらを優先 ---
    # --- Z軸は気圧高度のみを使用 ---
    position[:,2] = alt_pressure
    # --- まとめてログ出力 ---
    print(f"初期加速度: {initial_acc}")
    print(f"初期加速度ノルム: {initial_acc_norm}")
    print(f"初期ロール: {initial_roll:.3f}, 初期ピッチ: {initial_pitch:.3f}, 初期ヨー: {initial_yaw:.3f}")
    print(f"加速度バイアス: {bias}")
    print(f"位置計算: 初期={position[0]}, 最終={position[-1]}")
    print(f"位置範囲: X={position[:,0].min():.2f}～{position[:,0].max():.2f}, Y={position[:,1].min():.2f}～{position[:,1].max():.2f}, Z={position[:,2].min():.2f}～{position[:,2].max():.2f}")
    # GPS高度は使わない
    print(f"気圧高度: 初期={alt_pressure[0]:.2f}, 最終={alt_pressure[-1]:.2f}, 最大={alt_pressure.max():.2f}")
    static_samples = min(20, len(global_acc))
    bias = np.mean(global_acc[:static_samples], axis=0)
    global_acc = global_acc - bias
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)
    velocity = cumulative_trapezoid(global_acc, dx=dt, initial=0, axis=0)
    position = cumulative_trapezoid(velocity, dx=dt, initial=0, axis=0)
    return {
        'position': position,
        'acc_magnitude': acc_magnitude,
        'temperature': df['Temperature_C'].to_numpy(),
        'humidity': df['Humidity_%'].to_numpy(),
        'pressure': df['Pressure_hPa'].to_numpy(),
        'time': time
    }

# --- 表示部 ---
def plot_trajectory(data):
    if data is None:
        print("データ処理に失敗しました")
        return
    
    # 飛行サマリー計算
    position = data['position']
    time = data['time']
    velocity = data['velocity']
    acc_magnitude = data['acc_magnitude']
    
    max_altitude = np.max(position[:, 2])
    max_velocity = np.max(np.linalg.norm(velocity, axis=1))
    flight_time = time[-1] - time[0]
    horizontal_distance = np.sqrt(position[-1, 0]**2 + position[-1, 1]**2)
    max_acceleration = np.max(acc_magnitude)
    
    print("\n=== 飛行解析サマリー ===")
    print(f"最大高度: {max_altitude:.2f} m")
    print(f"最大速度: {max_velocity:.2f} m/s")
    print(f"飛行時間: {flight_time:.2f} s")
    print(f"水平移動距離: {horizontal_distance:.2f} m")
    print(f"最大加速度: {max_acceleration:.2f} m/s²")
    
    velocity_magnitude = np.linalg.norm(velocity, axis=1)
    custom_data = np.column_stack([
        acc_magnitude,
        data['temperature'],
        data['humidity'],
        data['pressure'],
        time,
        velocity_magnitude
    ])
    
    flight_path_trace = go.Scatter3d(
        x=position[:, 0],
        y=position[:, 1],
        z=position[:, 2],
        mode='lines+markers',
        line=dict(
            color=acc_magnitude,
            colorscale='Plasma',
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
    
    fig = go.Figure(
        data=[flight_path_trace],
        layout=go.Layout(
            title=dict(
                text="ロケット飛行ログ解析（改良版）",
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title='X (m)', titlefont=dict(size=14)),
                yaxis=dict(title='Y (m)', titlefont=dict(size=14)),
                zaxis=dict(title='Z (m)', titlefont=dict(size=14)),
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
    
    # HTML書き出し
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"flight_analysis_{timestamp}.html"
    fig.write_html(output_file)
    print(f"解析結果をHTMLファイルに出力しました: {output_file}")
    
    # ブラウザでも表示
    fig.show()
    
    return fig

if __name__ == "__main__":
    df = pd.read_csv("../sensor_data50.csv")  # 正しいファイルパスに修正
    data = process_sensor_data(df)
    plot_trajectory(data)
