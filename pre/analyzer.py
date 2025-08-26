import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid
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
    pressure = df['Pressure_hPa'].to_numpy()
    P0 = pressure[0]  # 初期気圧を基準とする
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
    # --- イベントログがあれば読み込んでセンサ時刻にマッピング ---
    event_log_path = os.path.join('event_log', 'event_log2.txt')
    events = []
    if os.path.exists(event_log_path):
        try:
            with open(event_log_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                    # split at first ':'
                    tstr, msg = line.split(':', 1)
                    try:
                        t = float(tstr)
                    except ValueError:
                        continue
                    events.append((t, msg.strip()))
            print(f"イベントログ読み込み: {len(events)} エントリ (path={event_log_path})")
        except Exception as e:
            print(f"イベントログ読み込み失敗: {e}")
    else:
        print(f"イベントログが見つかりません: {event_log_path}")

    # マップされたイベントのインデックスやタイムを初期化
    event_idx_map = {}
    if len(events) > 0:
        event_times = np.array([e[0] for e in events])
        # for each event, find nearest sensor index
        for t, msg in events:
            idx = int(np.argmin(np.abs(time - t)))
            event_idx_map.setdefault(msg, []).append((t, idx))
        # 簡易的に重要イベントを抽出
        # Flight start
        flight_start_idx = None
        for t, msg in events:
            if 'Flight start' in msg or 'Flight start:' in msg or 'Flight start' in msg:
                flight_start_idx = int(np.argmin(np.abs(time - t)))
                break
        # 下降検知
        descent_indices = [int(np.argmin(np.abs(time - t))) for t, msg in events if '下降検知' in msg or 'descent' in msg]
        descent_idx = descent_indices[0] if len(descent_indices) > 0 else None
        # パラシュート関連
        parachute_indices = [int(np.argmin(np.abs(time - t))) for t, msg in events if 'Parachute' in msg or 'parachute' in msg or 'parachute' in msg]
        parachute_idx = parachute_indices[0] if len(parachute_indices) > 0 else None
        print(f"マップされた主要イベント: flight_start_idx={flight_start_idx}, descent_idx={descent_idx}, parachute_idx={parachute_idx}")
    else:
        flight_start_idx = None
        descent_idx = None
        parachute_idx = None
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

    # 低域通過フィルタ（指数移動平均）関数
    def low_pass_filter(data, alpha=0.8):
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * filtered[i-1] + (1 - alpha) * data[i]
        return filtered

    # フィルタ済み加速度（姿勢推定で使用）
    acc_data_filtered = np.zeros_like(acc_data)
    for axis in range(3):
        acc_data_filtered[:, axis] = low_pass_filter(acc_data[:, axis])

    attitude = np.zeros((N, 3))
    attitude[0] = [initial_roll, initial_pitch, initial_yaw]
    # complementary filter params
    cf_alpha = 0.98
    for i in range(1, N):
        # gyro integrated prediction
        pred = attitude[i-1] + gyro_data_rad[i] * dt
        # accel-based roll/pitch (use filtered accel)
        a = acc_data_filtered[i]
        # avoid div0
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        roll_a = np.arctan2(a_norm[1], a_norm[2])
        pitch_a = np.arctan2(-a_norm[0], np.sqrt(a_norm[1]**2 + a_norm[2]**2))
        # yaw cannot be observed from accel; keep integrated yaw
        yaw_pred = pred[2]
        # fuse
        attitude[i, 0] = cf_alpha * pred[0] + (1 - cf_alpha) * roll_a
        attitude[i, 1] = cf_alpha * pred[1] + (1 - cf_alpha) * pitch_a
        attitude[i, 2] = yaw_pred
    global_acc = np.zeros_like(acc_data_filtered)
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data_filtered[i]
    global_acc[:, 2] = global_acc[:, 2] - 9.81
    # --- バイアス算出区間を静止区間（FlightStarted==0）に限定。イベントログがあればそれを優先して FlightStarted を補完 ---
    if 'FlightStarted' in df.columns:
        static_mask = df['FlightStarted'] == 0
    else:
        # デフォルトで静止は True。イベントログに基づき flight_start を上書き
        static_mask = np.ones(len(df), dtype=bool)
        if flight_start_idx is not None:
            static_mask[flight_start_idx:] = False
    if np.any(static_mask):
        bias = np.mean(global_acc[static_mask], axis=0)
    else:
        bias = np.mean(global_acc[:20], axis=0)
    global_acc = global_acc - bias
    if 'TotalAccel' in df.columns:
        acc_magnitude = df['TotalAccel'].to_numpy() * 9.81
    else:
        acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # --- ハイブリッド相対推定（データ駆動 + モデル駆動） ---
    N = len(df)
    position = np.zeros((N, 3))

    # Z軸は気圧高度（信頼性が高い）
    position[:, 2] = alt_pressure

    # --- モード設定と閾値（調整可能） ---
    alpha_lpf = 0.9          # 加速度/速度のLPF係数（1に近いほど遅い）
    a_thresh = 0.3           # m/s^2: 有意な横加速度の閾値
    window_len_s = 0.7       # s: 閾値継続評価ウィンドウ
    vel_clip = 10.0          # m/s

    max_altitude = np.max(alt_pressure)
    flight_duration = 0.0
    if 'FlightStarted' in df.columns:
        flight_indices = np.where(df['FlightStarted'] == 1)[0]
        if len(flight_indices) > 0:
            flight_duration = (flight_indices[-1] - flight_indices[0]) * dt

    # 簡易モデルの水平ドリフト（高度に比例する概形）
    horizontal_drift = min(max_altitude * 0.3, 20.0)
    pos_model = np.zeros((N, 2))
    for i in range(N):
        if max_altitude > 0:
            progress = alt_pressure[i] / max_altitude
        else:
            progress = 0.0
        pos_model[i, 0] = horizontal_drift * progress * 0.6
        pos_model[i, 1] = horizontal_drift * progress * 0.8

    # 地球座標での水平加速度ノルム
    a_h = np.linalg.norm(global_acc[:, :2], axis=1)

    # ウィンドウ長（サンプル数）
    window_len = max(1, int(round(window_len_s / max(dt, 1e-6))))
    # 有意な横加速度が継続しているかを判定（移動平均）
    import pandas as _pd
    active_series = _pd.Series(a_h > a_thresh)
    active_mask = active_series.rolling(window=window_len, center=True, min_periods=1).mean() > 0.5
    active_mask = active_mask.to_numpy()

    # データ駆動（短時間積分）による速度・位置推定（X/Y）
    vel = np.zeros((N, 2))
    pos_data = np.zeros((N, 2))
    # 積分（台形）で速度を求める
    vel[:, 0] = cumulative_trapezoid(global_acc[:, 0], time, initial=0)
    vel[:, 1] = cumulative_trapezoid(global_acc[:, 1], time, initial=0)

    # 静止区間では速度をゼロにする（バイアス補正のため）
    if 'FlightStarted' in df.columns:
        static_mask = df['FlightStarted'] == 0
        vel[static_mask, :] = 0.0

    # 速度の低周波（ドリフト）成分を除去するために単純なハイパス（vel - LPF(vel)）を適用
    vel_hp = np.zeros_like(vel)
    for axis in range(2):
        vel_lp = low_pass_filter(vel[:, axis], alpha=alpha_lpf)
        vel_hp[:, axis] = vel[:, axis] - vel_lp
    # 速度クリッピング
    vel_hp = np.clip(vel_hp, -vel_clip, vel_clip)

    # 位置はハイパス後の速度を積分
    pos_data[:, 0] = cumulative_trapezoid(vel_hp[:, 0], time, initial=0)
    pos_data[:, 1] = cumulative_trapezoid(vel_hp[:, 1], time, initial=0)

    # ブレンドウェイト（信頼度）：有意加速度で1、そうでなければ0へ線形に移行
    w = active_mask.astype(float)
    w_series = _pd.Series(w)
    w_smooth = w_series.rolling(window=window_len, center=True, min_periods=1).mean().to_numpy()

    # 最終位置はデータ駆動とモデル駆動をブレンド
    for i in range(N):
        position[i, 0] = w_smooth[i] * pos_data[i, 0] + (1 - w_smooth[i]) * pos_model[i, 0]
        position[i, 1] = w_smooth[i] * pos_data[i, 1] + (1 - w_smooth[i]) * pos_model[i, 1]
    # 下降/パラシュート検出後はモデル寄与を減らし、速度・位置の急な変化をリセット
    if 'descent_idx' in locals() and descent_idx is not None:
        reset_idx = descent_idx
        # 下降以降はモデルのスケールを減らす
        for i in range(reset_idx, N):
            # 緩やかにモデルを0へスケールダウン
            factor = max(0.0, 1.0 - (i - reset_idx) / max(1, int(5.0 / max(dt, 1e-6))))
            position[i, 0] = w_smooth[i] * pos_data[i, 0] + (1 - w_smooth[i]) * (pos_model[i, 0] * factor)
            position[i, 1] = w_smooth[i] * pos_data[i, 1] + (1 - w_smooth[i]) * (pos_model[i, 1] * factor)
        # 下降直後に速度リセット相当の処理（pos_data を基に直近をゼロ近傍に保つ）
        # ここでは簡易に下降時点以降のデータ駆動成分を減衰させる
        for axis in range(2):
            pos_data[reset_idx:, axis] *= 0.5
    # ログ用にモード情報も出す
    mode_flags = np.where(w_smooth > 0.5, 'data', 'model')
    # イベントがある場合、パラシュートや下降開始で速度リセットやモード切替の参考にする
    if descent_idx is not None:
        print(f"下降開始イベントが検出されました。センサー時刻 idx={descent_idx}, time={time[descent_idx]:.3f}s")
    if parachute_idx is not None:
        print(f"パラシュート関連イベントが検出されました。idx={parachute_idx}, time={time[parachute_idx]:.3f}s")
    # --- まとめてログ出力 ---
    print(f"初期加速度: {initial_acc}")
    print(f"初期加速度ノルム: {initial_acc_norm}")
    print(f"初期ロール: {initial_roll:.3f}, 初期ピッチ: {initial_pitch:.3f}, 初期ヨー: {initial_yaw:.3f}")
    print(f"加速度バイアス: {bias}")
    print(f"位置計算: 初期={position[0]}, 最終={position[-1]}")
    print(f"位置範囲: X={position[:,0].min():.2f}～{position[:,0].max():.2f}, Y={position[:,1].min():.2f}～{position[:,1].max():.2f}, Z={position[:,2].min():.2f}～{position[:,2].max():.2f}")
    print(f"気圧高度: 初期={alt_pressure[0]:.2f}, 最終={alt_pressure[-1]:.2f}, 最大={alt_pressure.max():.2f}")
    
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
    # 3D軌跡と高度変化グラフを作成
    if data is None:
        print("データ処理に失敗しました")
        return
    pos = data['position']
    time = data['time']
    acc_magnitude = data['acc_magnitude']
    # カスタムデータ準備
    custom_data = np.column_stack([
        acc_magnitude,
        data['temperature'],
        data['humidity'],
        data['pressure'],
        time
    ])
    # サブプロット設定
    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scatter3d"}],
               [{"type": "scatter"}]],
        subplot_titles=('飛行軌跡 3D', '高度変化'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    # 3D軌跡トレース
    flight_path = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='lines',
        line=dict(color=acc_magnitude, colorscale='Viridis', width=5,
                  colorbar=dict(title=dict(text='加速度 (m/s^2)'))),
        customdata=custom_data,
        hovertemplate='時刻:%{customdata[4]:.2f}s<br>' +
                     '加速度:%{customdata[0]:.2f}m/s²<br>' +
                     '温度:%{customdata[1]:.1f}°C<br>' +
                     '湿度:%{customdata[2]:.1f}%<br>' +
                     '気圧:%{customdata[3]:.1f}hPa<extra></extra>'
    )
    fig.add_trace(flight_path, row=1, col=1)
    # 高度変化トレース
    height_trace = go.Scatter(
        x=time, y=pos[:, 2], mode='lines',
        line=dict(color='blue', width=2), name='高度'
    )
    fig.add_trace(height_trace, row=2, col=1)
    # レイアウト調整
    fig.update_layout(title='ロケット飛行ログ解析', height=800)
    fig.update_scenes(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='高度 (m)')
    fig.update_xaxes(title_text='時間 (s)', row=2, col=1)
    fig.update_yaxes(title_text='高度 (m)', row=2, col=1)
    # HTML出力
    fig.write_html("output.html")
    print("output.html をブラウザで開いてください")
    input("Enterキーで終了します...")

if __name__ == "__main__":
    df = pd.read_csv("sensor.csv")
    data = process_sensor_data(df)
    plot_trajectory(data)
