
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid

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
    print(f"データ件数: {len(df)}")
    print(f"初期加速度: {initial_acc}")
    print(f"初期加速度ノルム: {initial_acc_norm}")
    print(f"初期ロール: {initial_roll:.3f}, 初期ピッチ: {initial_pitch:.3f}, 初期ヨー: {initial_yaw:.3f}")
    print(f"加速度バイアス: {bias}")
    print(f"位置計算: 初期={position[0]}, 最終={position[-1]}")
    print(f"位置範囲: X={position[:,0].min():.2f}～{position[:,0].max():.2f}, Y={position[:,1].min():.2f}～{position[:,1].max():.2f}, Z={position[:,2].min():.2f}～{position[:,2].max():.2f}")
    required_columns = ['Time_s', 'AccelX_g', 'AccelY_g', 'AccelZ_g', 'GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s', 'Temperature_C', 'Humidity_%', 'Pressure_hPa']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"警告: 以下のカラムが見つかりません: {missing}")
        return None
    time = df['Time_s'].values
    dt = np.mean(np.diff(time))
    acc_columns = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
    gyro_columns = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
    acc_data = df[acc_columns].to_numpy() * 9.81
    gyro_data = df[gyro_columns].to_numpy()
    gyro_data_rad = np.deg2rad(gyro_data)
    initial_samples = min(10, len(acc_data))
    initial_acc = np.mean(acc_data[:initial_samples], axis=0)
    initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
    initial_roll = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])
    initial_pitch = np.arctan2(-initial_acc_norm[0], np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))
    initial_yaw = 0.0
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
    custom_data = np.column_stack([
        data['acc_magnitude'],
        data['temperature'],
        data['humidity'],
        data['pressure'],
        data['time']
    ])
    flight_path_trace = go.Scatter3d(
        x=data['position'][:, 0],
        y=data['position'][:, 1],
        z=data['position'][:, 2],
        mode='lines',
        line=dict(
            color=data['acc_magnitude'],
            colorscale='Viridis',
            width=5,
            colorbar=dict(
                title=dict(text='加速度 (m/s^2)'),
                tickfont=dict(size=10)
            )
        ),
        name='飛行軌跡',
        customdata=custom_data,
        hovertemplate='<b>飛行データ</b><br>' +
                      '時刻: %{customdata[4]:.2f} s<br>' +
                      '加速度: %{customdata[0]:.2f} m/s²<br>' +
                      '温度: %{customdata[1]:.1f} °C<br>' +
                      '湿度: %{customdata[2]:.1f} %<br>' +
                      '気圧: %{customdata[3]:.1f} hPa<br>' +
                      'X: %{x:.2f} m<br>' +
                      'Y: %{y:.2f} m<br>' +
                      'Z: %{z:.2f} m<extra></extra>'
    )
    fig = go.Figure(
        data=[flight_path_trace],
        layout=go.Layout(
            title="ロケット飛行ログ解析（静的軌道プロット）",
            scene=dict(
                xaxis=dict(title='X (m)'),
                yaxis=dict(title='Y (m)'),
                zaxis=dict(title='Z (m)'),
                aspectmode='data'
            )
        )
    )
    fig.show()

if __name__ == "__main__":
    df = pd.read_csv("sensor.csv")
    data = process_sensor_data(df)
    plot_trajectory(data)
