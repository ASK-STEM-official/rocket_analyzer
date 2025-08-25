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
    df = pd.read_csv("before_flight2.csv")

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
    acc_data = data[acc_columns].to_numpy() * 9.81
    # 加速度データの異常値処理（±50gを超える値をクリップ）
    acc_data = np.clip(acc_data, -50*9.81, 50*9.81)
    
    # ジャイロデータ（単位: deg/s）を取得
    gyro_data = data[gyro_columns].to_numpy()   
    
    # ジャイロデータの異常値処理（±250度/秒を超える値をクリップ）
    gyro_data = np.clip(gyro_data, -200, 200)
    
    # センサーバイアス（オフセット）を除去（静止時の平均値を引く）
    # 最初の10データポイントを静止状態と仮定
    acc_bias = np.mean(acc_data[:10], axis=0)
    acc_bias[2] -= 9.81  # Z軸は重力を考慮
    acc_data = acc_data - acc_bias
    
    gyro_bias = np.mean(gyro_data[:10], axis=0)
    gyro_data = gyro_data - gyro_bias
    
    # 簡易ローパスフィルター（移動平均）でノイズを軽減
    window_size = 3
    for i in range(3):
        acc_data[:, i] = np.convolve(acc_data[:, i], np.ones(window_size)/window_size, mode='same')
        gyro_data[:, i] = np.convolve(gyro_data[:, i], np.ones(window_size)/window_size, mode='same')
    
    # データの品質チェック（デバッグ情報）
    print(f"データ点数: {len(time)}")
    print(f"時間範囲: {time[0]:.3f} - {time[-1]:.3f} 秒")
    print(f"平均時間間隔: {np.mean(np.diff(time)):.4f} 秒")
    print(f"加速度バイアス: X:{acc_bias[0]:.2f}, Y:{acc_bias[1]:.2f}, Z:{acc_bias[2]:.2f} m/s^2")
    print(f"ジャイロバイアス: X:{gyro_bias[0]:.1f}, Y:{gyro_bias[1]:.1f}, Z:{gyro_bias[2]:.1f} deg/s")
    print(f"加速度範囲: X[{acc_data[:,0].min():.2f}, {acc_data[:,0].max():.2f}] Y[{acc_data[:,1].min():.2f}, {acc_data[:,1].max():.2f}] Z[{acc_data[:,2].min():.2f}, {acc_data[:,2].max():.2f}] m/s^2")
    print(f"ジャイロ範囲: X[{gyro_data[:,0].min():.1f}, {gyro_data[:,0].max():.1f}] Y[{gyro_data[:,1].min():.1f}, {gyro_data[:,1].max():.1f}] Z[{gyro_data[:,2].min():.1f}, {gyro_data[:,2].max():.1f}] deg/s")
    
    # ジャイロデータをラジアンに変換
    gyro_data_rad = np.deg2rad(gyro_data)
    
    # 実際の時間差分を計算（不均一な時間間隔に対応）
    dt_array = np.diff(time)
    dt_array = np.append(dt_array, dt_array[-1])  # 最後の要素用
    
    # 各時刻でのオイラー角を積分により計算（簡易版、実際の時間差を使用）
    attitude = np.zeros_like(gyro_data_rad)
    for i in range(1, len(attitude)):
        attitude[i] = attitude[i-1] + gyro_data_rad[i] * dt_array[i-1]
    
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
        global_acc[i] = R_mat @ acc_data[i]
    
    # 重力加速度を正しく除去（グローバル座標系でZ軸負方向に9.81）
    gravity_vector = np.array([0, 0, 9.81])  # グローバル座標系での重力
    global_acc = global_acc - gravity_vector
    
    # 小さな加速度を無視（ノイズ除去）
    threshold = 1.0  # 1.0 m/s^2以下の加速度を0にする（より積極的）
    global_acc[np.abs(global_acc) < threshold] = 0
    
    # 静止時の加速度バイアスを除去
    static_samples = min(20, len(global_acc))
    bias = np.mean(global_acc[:static_samples], axis=0)
    global_acc = global_acc - bias
    
    # 各時刻のグローバル加速度の大きさを計算
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # 実際の時間差を使って加速度の積分により速度、さらに速度の積分により位置を計算
    velocity = np.zeros_like(global_acc)
    position = np.zeros_like(global_acc)
    
    for i in range(1, N):
        # 台形公式で積分
        velocity[i] = velocity[i-1] + (global_acc[i-1] + global_acc[i]) * dt_array[i-1] / 2
        position[i] = position[i-1] + (velocity[i-1] + velocity[i]) * dt_array[i-1] / 2
    
    # 速度ドリフト補正（最終的に停止すると仮定）
    final_velocity_drift = velocity[-1] / len(velocity)
    for i in range(len(velocity)):
        velocity[i] = velocity[i] - final_velocity_drift * i
    
    # 位置を再計算（ドリフト補正後の速度を使用）
    position = np.zeros_like(global_acc)
    for i in range(1, N):
        position[i] = position[i-1] + (velocity[i-1] + velocity[i]) * dt_array[i-1] / 2
    
    # 座標軸の修正: Y→Z（高度）, Z→Y（前後方向）
    # position[:, [0,1,2]] を position[:, [0,2,1]] に変換
    position_corrected = position[:, [0, 2, 1]]  # X, Z→Y, Y→Z
    velocity_corrected = velocity[:, [0, 2, 1]]
    
    # 最終的な飛行距離の情報（修正後の座標系）
    final_position = position_corrected[-1]
    max_altitude = np.max(position_corrected[:, 2])  # Z軸が高度
    flight_distance_2d = np.sqrt(final_position[0]**2 + final_position[1]**2)
    
    print(f"\n=== 飛行結果（座標軸修正後） ===")
    print(f"最終位置: X={final_position[0]:.2f}m, Y={final_position[1]:.2f}m, Z={final_position[2]:.2f}m")
    print(f"最大高度: {max_altitude:.2f}m")
    print(f"水平飛行距離: {flight_distance_2d:.2f}m")
    print(f"最大速度: {np.max(np.linalg.norm(velocity_corrected, axis=1)):.2f}m/s")
    print(f"最大加速度: {np.max(acc_magnitude):.2f}m/s^2")
    
    # 以降のプロット用に修正後の座標を使用
    position = position_corrected
    velocity = velocity_corrected
    
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
