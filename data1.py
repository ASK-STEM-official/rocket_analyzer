# coding: utf-8
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid

# オイラー角（roll, pitch, yaw）から回転行列を計算する関数
def euler_to_rotation_matrix(roll, pitch, yaw):
    # X軸回りの回転行列
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    # Y軸回りの回転行列
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    # Z軸回りの回転行列
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    # 3軸の回転行列を掛け合わせて最終的な回転行列を得る
    R = R_z @ R_y @ R_x
    return R

# メインの処理を行う関数
def main():
    # CSVファイルからセンサーデータを読み込む
    df = pd.read_csv("sensor.csv")

    # 'Time_s' カラムが存在すればその値を使用、なければ dt=0.01 秒でタイムスタンプを生成
    if 'Time_s' in df.columns:
        time = df['Time_s'].values
        dt = np.mean(np.diff(time))
    else:
        dt = 0.01
        time = np.arange(len(df)) * dt

    # 新しいCSVファイルに対応したカラム名を指定
    acc_columns = ['AccelX_g', 'AccelY_g', 'AccelZ_g']
    gyro_columns = ['GyroX_deg_s', 'GyroY_deg_s', 'GyroZ_deg_s']
    
    # 必要なカラムが存在するかチェック
    missing_columns = []
    for col in acc_columns + gyro_columns:
        if col not in df.columns:
            missing_columns.append(col)
    
    if missing_columns:
        raise ValueError(f"必要なカラムが見つかりません: {missing_columns}")
    
    # 加速度データを取得し、単位を g から m/s^2 に変換
    acc_data = df[acc_columns].to_numpy() * 9.81  
    # ジャイロデータ（単位: deg/s）を取得
    gyro_data = df[gyro_columns].to_numpy()   
    
    # ジャイロデータをラジアンに変換
    gyro_data_rad = np.deg2rad(gyro_data)
    
    # 初期姿勢推定（最初の数秒の加速度から推定）
    initial_samples = min(10, len(acc_data))
    initial_acc = np.mean(acc_data[:initial_samples], axis=0)
    initial_acc_norm = initial_acc / np.linalg.norm(initial_acc)
    
    # 初期のロール・ピッチを推定
    initial_roll = np.arctan2(initial_acc_norm[1], initial_acc_norm[2])
    initial_pitch = np.arctan2(-initial_acc_norm[0], np.sqrt(initial_acc_norm[1]**2 + initial_acc_norm[2]**2))
    initial_yaw = 0.0  # 初期ヨー角は0と仮定
    
    # 姿勢角の計算（改良版）
    N = len(time)
    attitude = np.zeros((N, 3))
    attitude[0] = [initial_roll, initial_pitch, initial_yaw]
    
    # より正確な姿勢推定
    for i in range(1, N):
        # 前の姿勢角に角速度を積分して加算
        attitude[i] = attitude[i-1] + gyro_data_rad[i] * dt
    
    # 加速度のローパスフィルタリング（ノイズ除去）
    def low_pass_filter(data, alpha=0.8):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
        return filtered
    
    # 各軸の加速度にローパスフィルタを適用
    acc_data_filtered = np.zeros_like(acc_data)
    for axis in range(3):
        acc_data_filtered[:, axis] = low_pass_filter(acc_data[:, axis])
    
    global_acc = np.zeros_like(acc_data_filtered)
    # 各時刻における加速度をグローバル座標系に変換
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data_filtered[i]
    
    # 重力加速度を除去（Z軸方向）
    global_acc[:, 2] = global_acc[:, 2] - 9.81
    
    # 静止時の加速度バイアスを除去
    static_samples = min(20, len(global_acc))
    bias = np.mean(global_acc[:static_samples], axis=0)
    global_acc = global_acc - bias
    
    # 各時刻のグローバル加速度の大きさを計算
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # 加速度の積分により速度、さらに速度の積分により位置を計算（台形公式）
    velocity = cumulative_trapezoid(global_acc, dx=dt, initial=0, axis=0)
    position = cumulative_trapezoid(velocity, dx=dt, initial=0, axis=0)
    
    # アニメーションで使用するフレーム数の上限を設定
    max_frames = 100
    step = max(1, N // max_frames)
    indices = list(range(0, N, step))
    if indices[-1] != N - 1:
        indices.append(N - 1)
    
    frames = []
    axis_length = 1.0  # 局所座標軸の長さ（メートル）
    
    # 飛行軌跡のトレースを作成
    custom_data = np.column_stack([
        acc_magnitude,
        df['Temperature_C'].to_numpy(),
        df['Humidity_%'].to_numpy(),
        df['Pressure_hPa'].to_numpy(),
        time
    ])
    
    flight_path_trace = go.Scatter3d(
        x=position[:, 0],
        y=position[:, 1],
        z=position[:, 2],
        mode='lines',
        line=dict(
            color=acc_magnitude, 
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
    
    # 各フレームごとに現在位置と局所座標軸を表示するフレームを作成
    for idx in indices:
        pos_current = position[idx]
        roll = attitude[idx, 0]
        pitch = attitude[idx, 1]
        yaw = attitude[idx, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        
        # 局所座標軸の終点を計算
        x_axis_end = pos_current + R_mat @ np.array([axis_length, 0, 0])
        y_axis_end = pos_current + R_mat @ np.array([0, axis_length, 0])
        z_axis_end = pos_current + R_mat @ np.array([0, 0, axis_length])
        
        # 現在位置を示すマーカー
        current_pos_trace = go.Scatter3d(
            x=[pos_current[0]],
            y=[pos_current[1]],
            z=[pos_current[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='現在位置'
        )
        
        # 局所座標軸の表示
        x_axis_trace = go.Scatter3d(
            x=[pos_current[0], x_axis_end[0]],
            y=[pos_current[1], x_axis_end[1]],
            z=[pos_current[2], x_axis_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='X軸'
        )
        y_axis_trace = go.Scatter3d(
            x=[pos_current[0], y_axis_end[0]],
            y=[pos_current[1], y_axis_end[1]],
            z=[pos_current[2], y_axis_end[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Y軸'
        )
        z_axis_trace = go.Scatter3d(
            x=[pos_current[0], z_axis_end[0]],
            y=[pos_current[1], z_axis_end[1]],
            z=[pos_current[2], z_axis_end[2]],
            mode='lines',
            line=dict(color='orange', width=5),
            name='Z軸'
        )
        
        frame_data = [flight_path_trace, current_pos_trace, x_axis_trace, y_axis_trace, z_axis_trace]
        frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))
    
    # 初期フレームのデータを作成
    init_idx = indices[0]
    pos_init = position[init_idx]
    roll_init = attitude[init_idx, 0]
    pitch_init = attitude[init_idx, 1]
    yaw_init = attitude[init_idx, 2]
    R_init = euler_to_rotation_matrix(roll_init, pitch_init, yaw_init)
    
    x_axis_end_init = pos_init + R_init @ np.array([axis_length, 0, 0])
    y_axis_end_init = pos_init + R_init @ np.array([0, axis_length, 0])
    z_axis_end_init = pos_init + R_init @ np.array([0, 0, axis_length])
    
    current_pos_trace_init = go.Scatter3d(
        x=[pos_init[0]],
        y=[pos_init[1]],
        z=[pos_init[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='現在位置'
    )
    x_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], x_axis_end_init[0]],
        y=[pos_init[1], x_axis_end_init[1]],
        z=[pos_init[2], x_axis_end_init[2]],
        mode='lines',
        line=dict(color='red', width=5),
        name='X軸'
    )
    y_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], y_axis_end_init[0]],
        y=[pos_init[1], y_axis_end_init[1]],
        z=[pos_init[2], y_axis_end_init[2]],
        mode='lines',
        line=dict(color='green', width=5),
        name='Y軸'
    )
    z_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], z_axis_end_init[0]],
        y=[pos_init[1], z_axis_end_init[1]],
        z=[pos_init[2], z_axis_end_init[2]],
        mode='lines',
        line=dict(color='orange', width=5),
        name='Z軸'
    )
    
    # 図のレイアウトおよびアニメーションの設定
    fig = go.Figure(
        data=[
            flight_path_trace,
            current_pos_trace_init,
            x_axis_trace_init,
            y_axis_trace_init,
            z_axis_trace_init
        ],
        layout=go.Layout(
            title="ロケット飛行ログ解析（改良版軌道計算）",
            scene=dict(
                xaxis=dict(title='X (m)'),
                yaxis=dict(title='Y (m)'),
                zaxis=dict(title='Z (m)'),
                aspectmode='data'
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "再生",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 50, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}]
                        }
                    ]
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "時刻: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [[f"frame{idx}"],
                                     {"frame": {"duration": 50, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                            "label": f"{time[idx]:.2f}s",
                            "method": "animate"
                        }
                        for idx in indices
                    ]
                }
            ]
        ),
        frames=frames
    )
    
    # 凡例の表示設定を改善
    fig.update_layout(
        legend=dict(
            title="凡例",
            orientation="h",
            x=0.5,
            y=1.15,
            xanchor="center",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=2
        )
    )
    
    # 作成した図を表示
    fig.show()

if __name__ == "__main__":
    main()
