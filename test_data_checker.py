import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class RocketFlightVisualizer:
    def __init__(self, csv_filename):
        """
        CSVファイルからロケット飛行データを読み込み、可視化を行う
        """
        self.df = pd.read_csv(csv_filename)
        self.filename = csv_filename
        print(f"データ読み込み完了: {len(self.df)} データポイント")
        print(f"時間範囲: {self.df['Time_s'].min():.1f} - {self.df['Time_s'].max():.1f} 秒")
        
    def create_comprehensive_visualization(self):
        """包括的な可視化（Plotly使用）"""
        # データ準備
        time = self.df['Time_s']
        altitude = self.df['altitude']
        velocity = self.df['velocity']
        
        # 水平位置の計算（姿勢角から推定）
        theta_rad = np.radians(self.df['theta'])
        # 簡単な積分で水平移動を推定
        dt = np.mean(np.diff(time))
        vx_est = velocity * np.sin(theta_rad) * 0.1  # 推力の水平成分
        vy_est = np.random.normal(0, 0.5, len(time))  # 横風効果
        
        x_pos = np.cumsum(vx_est) * dt
        y_pos = np.cumsum(vy_est) * dt
        
        # 色分け用の飛行段階
        flight_phase = []
        for i, t in enumerate(time):
            if t <= 3.0:
                flight_phase.append('推力段階')
            elif velocity.iloc[i] > 0:
                flight_phase.append('上昇段階')
            elif self.df['parachute_deployed'].iloc[i]:
                flight_phase.append('パラシュート降下')
            else:
                flight_phase.append('自由降下')
        
        # サブプロット作成 - 修正版
        fig = make_subplots(
            rows=3, cols=3,
            specs=[
                [{"type": "scatter3d", "colspan": 2}, None, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]
            ],
            subplot_titles=[
                '3D飛行軌道', '', '高度変化',
                '速度変化', '加速度（3軸）', '角速度（3軸）',
                '姿勢角変化', '気圧・温度', '推力・質量・抗力'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. 3D軌道プロット
        colors = {'推力段階': 'red', '上昇段階': 'orange', '自由降下': 'blue', 'パラシュート降下': 'green'}
        for phase in ['推力段階', '上昇段階', '自由降下', 'パラシュート降下']:
            mask = [p == phase for p in flight_phase]
            if any(mask):
                indices = [i for i, m in enumerate(mask) if m]
                fig.add_trace(go.Scatter3d(
                    x=[x_pos[i] for i in indices],
                    y=[y_pos[i] for i in indices],
                    z=[altitude.iloc[i] for i in indices],
                    mode='lines+markers',
                    line=dict(color=colors[phase], width=6),
                    marker=dict(size=3),
                    name=phase,
                    showlegend=True
                ), row=1, col=1)
        
        # スタート・ゴール地点をマーク
        start_idx = 0
        end_idx = len(altitude) - 1
        
        fig.add_trace(go.Scatter3d(
            x=[x_pos[start_idx]], y=[y_pos[start_idx]], z=[altitude.iloc[start_idx]],
            mode='markers',
            marker=dict(color='black', size=10, symbol='diamond'),
            name='スタート地点'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=[x_pos[end_idx]], y=[y_pos[end_idx]], z=[altitude.iloc[end_idx]],
            mode='markers',
            marker=dict(color='purple', size=10, symbol='square'),
            name='着地地点'
        ), row=1, col=1)
        
        # 2. 高度変化
        fig.add_trace(go.Scatter(
            x=time, y=altitude,
            mode='lines',
            line=dict(color='blue', width=2),
            name='高度',
            showlegend=False
        ), row=1, col=3)
        
        # 3. 速度変化
        fig.add_trace(go.Scatter(
            x=time, y=velocity,
            mode='lines',
            line=dict(color='green', width=2),
            name='速度',
            showlegend=False
        ), row=2, col=1)
        
        # 4. 加速度（3軸）
        fig.add_trace(go.Scatter(x=time, y=self.df['AccelX_g'], name='AccelX', line=dict(color='red'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=time, y=self.df['AccelY_g'], name='AccelY', line=dict(color='green'), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=time, y=self.df['AccelZ_g'], name='AccelZ', line=dict(color='blue'), showlegend=False), row=2, col=2)
        
        # 5. 角速度（3軸）
        fig.add_trace(go.Scatter(x=time, y=self.df['GyroX_deg_s'], name='GyroX', line=dict(color='red', dash='dash'), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time, y=self.df['GyroY_deg_s'], name='GyroY', line=dict(color='green', dash='dash'), showlegend=False), row=2, col=3)
        fig.add_trace(go.Scatter(x=time, y=self.df['GyroZ_deg_s'], name='GyroZ', line=dict(color='blue', dash='dash'), showlegend=False), row=2, col=3)
        
        # 6. 姿勢角変化
        fig.add_trace(go.Scatter(
            x=time, y=self.df['theta'],
            mode='lines',
            line=dict(color='purple', width=2),
            name='姿勢角',
            showlegend=False
        ), row=3, col=1)
        
        # 7. 気圧・温度 (secondary y-axis not directly supported in subplots, using separate traces)
        fig.add_trace(go.Scatter(
            x=time, y=self.df['Pressure_hPa'],
            mode='lines',
            line=dict(color='orange', width=2),
            name='気圧 (hPa)',
            showlegend=False
        ), row=3, col=2)
        
        # 温度を正規化して同じ軸に表示
        temp_normalized = (self.df['Temperature_C'] - self.df['Temperature_C'].min()) / (self.df['Temperature_C'].max() - self.df['Temperature_C'].min()) * 100 + self.df['Pressure_hPa'].min()
        fig.add_trace(go.Scatter(
            x=time, y=temp_normalized,
            mode='lines',
            line=dict(color='cyan', width=2),
            name='温度 (正規化)',
            showlegend=False
        ), row=3, col=2)
        
        # 8. 推力・質量・抗力
        fig.add_trace(go.Scatter(x=time, y=self.df['thrust'], name='推力 (N)', line=dict(color='red', width=3), showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time, y=self.df['mass'], name='質量 (kg)', line=dict(color='black', dash='dot'), showlegend=False), row=3, col=3)
        fig.add_trace(go.Scatter(x=time, y=self.df['drag'], name='抗力 (N)', line=dict(color='brown'), showlegend=False), row=3, col=3)
        
        # レイアウト調整
        last_time = time.iloc[len(time)-1] if len(time) > 0 else 0
        fig.update_layout(
            title=f'ロケット飛行データ包括解析 - {self.filename}<br>最高高度: {altitude.max():.1f}m, 最大速度: {velocity.max():.1f}m/s, 飛行時間: {last_time:.1f}s',
            height=1000,
            showlegend=True
        )
        
        # 軸ラベル設定
        fig.update_scenes(
            xaxis_title="X座標 (m)",
            yaxis_title="Y座標 (m)",
            zaxis_title="高度 (m)",
            row=1, col=1
        )
        
        # 各軸のラベル設定
        fig.update_xaxes(title_text='時間 (s)', row=1, col=3)
        fig.update_yaxes(title_text='高度 (m)', row=1, col=3)
        
        fig.update_xaxes(title_text='時間 (s)', row=2, col=1)
        fig.update_yaxes(title_text='速度 (m/s)', row=2, col=1)
        
        fig.update_xaxes(title_text='時間 (s)', row=2, col=2)
        fig.update_yaxes(title_text='加速度 (g)', row=2, col=2)
        
        fig.update_xaxes(title_text='時間 (s)', row=2, col=3)
        fig.update_yaxes(title_text='角速度 (deg/s)', row=2, col=3)
        
        fig.update_xaxes(title_text='時間 (s)', row=3, col=1)
        fig.update_yaxes(title_text='姿勢角 (deg)', row=3, col=1)
        
        fig.update_xaxes(title_text='時間 (s)', row=3, col=2)
        fig.update_yaxes(title_text='気圧/温度', row=3, col=2)
        
        fig.update_xaxes(title_text='時間 (s)', row=3, col=3)
        fig.update_yaxes(title_text='力・質量', row=3, col=3)
        
        # HTMLファイルとして保存
        last_time = time.iloc[len(time)-1] if len(time) > 0 else 0
        output_filename = f'rocket_flight_comprehensive_{int(last_time*1000)}.html'
        fig.write_html(output_filename)
        print(f"包括的可視化を保存しました: {output_filename}")
        
        return fig
    
    def create_trajectory_animation(self):
        """3D軌道のアニメーション作成"""
        time = self.df['Time_s']
        altitude = self.df['altitude']
        
        # 水平位置の推定
        theta_rad = np.radians(self.df['theta'])
        dt = np.mean(np.diff(time))
        vx_est = self.df['velocity'] * np.sin(theta_rad) * 0.1
        vy_est = np.random.normal(0, 0.5, len(time))
        
        x_pos = np.cumsum(vx_est) * dt
        y_pos = np.cumsum(vy_est) * dt
        
        # アニメーション用のフレームを作成
        frames = []
        for i in range(0, len(self.df), max(1, len(self.df)//100)):  # 100フレーム程度に調整
            frame_data = go.Scatter3d(
                x=x_pos[:i+1],
                y=y_pos[:i+1],
                z=altitude[:i+1],
                mode='lines+markers',
                line=dict(color='blue', width=4),
                marker=dict(size=4, color=time[:i+1], colorscale='Viridis'),
                name=f'時刻: {time.iloc[i]:.1f}s'
            )
            frames.append(go.Frame(data=[frame_data], name=str(i)))
        
        # 初期プロット
        start_idx = 0
        fig = go.Figure(
            data=[go.Scatter3d(
                x=[x_pos[start_idx]], y=[y_pos[start_idx]], z=[altitude.iloc[start_idx]],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='ロケット位置'
            )],
            frames=frames
        )
        
        # アニメーション設定
        fig.update_layout(
            title='ロケット3D軌道アニメーション',
            scene=dict(
                xaxis_title="X座標 (m)",
                yaxis_title="Y座標 (m)",
                zaxis_title="高度 (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[{
                'buttons': [
                    {'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}], 'label': '▶️', 'method': 'animate'},
                    {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': '⏸️', 'method': 'animate'}
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        animation_filename = f'rocket_trajectory_animation_{int(time.iloc[0] if len(time) > 0 else 0)*1000}.html'
        fig.write_html(animation_filename)
        print(f"軌道アニメーションを保存しました: {animation_filename}")
        
        return fig
    
    def create_matplotlib_plots(self):
        """Matplotlib版の詳細プロット"""
        fig = plt.figure(figsize=(20, 15))
        
        # 日本語フォント設定（必要に応じて）
        plt.rcParams['font.size'] = 10
        
        # 1. 3D軌道
        ax1 = plt.subplot(3, 4, (1, 2), projection='3d')
        time = self.df['Time_s']
        altitude = self.df['altitude']
        
        # 簡易的な水平位置計算
        theta_rad = np.radians(self.df['theta'])
        x_pos = np.cumsum(self.df['velocity'] * np.sin(theta_rad)) * 0.01
        y_pos = np.cumsum(np.random.normal(0, 0.1, len(time)))
        
        # 軌道を段階別に色分け
        thrust_mask = time <= 3.0
        ascent_mask = (time > 3.0) & (self.df['velocity'] > 0)
        descent_mask = (time > 3.0) & (self.df['velocity'] <= 0) & (~self.df['parachute_deployed'])
        chute_mask = self.df['parachute_deployed']
        
        ax1.plot(x_pos[thrust_mask], y_pos[thrust_mask], altitude[thrust_mask], 'r-', linewidth=3, label='推力段階')
        ax1.plot(x_pos[ascent_mask], y_pos[ascent_mask], altitude[ascent_mask], 'orange', linewidth=2, label='上昇段階')
        ax1.plot(x_pos[descent_mask], y_pos[descent_mask], altitude[descent_mask], 'b-', linewidth=2, label='降下段階')
        ax1.plot(x_pos[chute_mask], y_pos[chute_mask], altitude[chute_mask], 'g-', linewidth=2, label='パラシュート')
        
        start_idx = 0
        end_idx = len(altitude) - 1
        ax1.scatter([x_pos[start_idx]], [y_pos[start_idx]], [altitude.iloc[start_idx]], color='black', s=100, marker='o', label='発射地点')
        ax1.scatter([x_pos[end_idx]], [y_pos[end_idx]], [altitude.iloc[end_idx]], color='purple', s=100, marker='s', label='着地地点')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('高度 (m)')
        ax1.set_title('3D飛行軌道')
        ax1.legend()
        
        # 2. 高度-時間
        plt.subplot(3, 4, 3)
        plt.plot(time, altitude, 'b-', linewidth=2)
        plt.xlabel('時間 (s)')
        plt.ylabel('高度 (m)')
        plt.title('高度変化')
        plt.grid(True)
        
        # 3. 速度-時間
        plt.subplot(3, 4, 4)
        plt.plot(time, self.df['velocity'], 'g-', linewidth=2)
        plt.xlabel('時間 (s)')
        plt.ylabel('速度 (m/s)')
        plt.title('速度変化')
        plt.grid(True)
        
        # 4. 加速度（3軸）
        plt.subplot(3, 4, 5)
        plt.plot(time, self.df['AccelX_g'], 'r-', label='X軸')
        plt.plot(time, self.df['AccelY_g'], 'g-', label='Y軸')
        plt.plot(time, self.df['AccelZ_g'], 'b-', label='Z軸')
        plt.xlabel('時間 (s)')
        plt.ylabel('加速度 (g)')
        plt.title('3軸加速度')
        plt.legend()
        plt.grid(True)
        
        # 5. 角速度（3軸）
        plt.subplot(3, 4, 6)
        plt.plot(time, self.df['GyroX_deg_s'], 'r--', label='X軸')
        plt.plot(time, self.df['GyroY_deg_s'], 'g--', label='Y軸')
        plt.plot(time, self.df['GyroZ_deg_s'], 'b--', label='Z軸')
        plt.xlabel('時間 (s)')
        plt.ylabel('角速度 (deg/s)')
        plt.title('3軸角速度')
        plt.legend()
        plt.grid(True)
        
        # 6. 姿勢角
        plt.subplot(3, 4, 7)
        plt.plot(time, self.df['theta'], 'purple', linewidth=2)
        plt.xlabel('時間 (s)')
        plt.ylabel('姿勢角 (deg)')
        plt.title('姿勢角変化')
        plt.grid(True)
        
        # 7. 気圧・温度
        plt.subplot(3, 4, 8)
        ax_pressure = plt.gca()
        ax_temp = ax_pressure.twinx()
        
        ax_pressure.plot(time, self.df['Pressure_hPa'], 'orange', linewidth=2, label='気圧')
        ax_temp.plot(time, self.df['Temperature_C'], 'cyan', linewidth=2, label='温度')
        
        ax_pressure.set_xlabel('時間 (s)')
        ax_pressure.set_ylabel('気圧 (hPa)', color='orange')
        ax_temp.set_ylabel('温度 (°C)', color='cyan')
        ax_pressure.set_title('気圧・温度変化')
        ax_pressure.grid(True)
        
        # 8. 推力・質量・抗力
        plt.subplot(3, 4, 9)
        ax_force = plt.gca()
        ax_mass = ax_force.twinx()
        
        ax_force.plot(time, self.df['thrust'], 'r-', linewidth=3, label='推力')
        ax_force.plot(time, self.df['drag'], 'brown', linewidth=2, label='抗力')
        ax_mass.plot(time, self.df['mass'], 'k--', linewidth=2, label='質量')
        
        ax_force.set_xlabel('時間 (s)')
        ax_force.set_ylabel('力 (N)', color='red')
        ax_mass.set_ylabel('質量 (kg)', color='black')
        ax_force.set_title('推力・抗力・質量')
        ax_force.grid(True)
        ax_force.legend(loc='upper left')
        ax_mass.legend(loc='upper right')
        
        # 9. 飛行段階表示
        plt.subplot(3, 4, 10)
        flight_phase = np.zeros_like(time)
        flight_phase[time <= 3.0] = 1  # 推力段階
        flight_phase[(time > 3.0) & (self.df['velocity'] > 0)] = 2  # 上昇段階
        flight_phase[(time > 3.0) & (self.df['velocity'] <= 0) & (~self.df['parachute_deployed'])] = 3  # 降下段階
        flight_phase[self.df['parachute_deployed']] = 4  # パラシュート段階
        
        plt.plot(time, flight_phase, 'k-', linewidth=3)
        plt.xlabel('時間 (s)')
        plt.ylabel('飛行段階')
        plt.title('飛行段階')
        plt.yticks([1, 2, 3, 4], ['推力', '上昇', '降下', 'パラシュート'])
        plt.grid(True)
        
        # 10. 総加速度
        plt.subplot(3, 4, 11)
        plt.plot(time, self.df['TotalAccel'], 'magenta', linewidth=2)
        plt.xlabel('時間 (s)')
        plt.ylabel('総加速度 (g)')
        plt.title('総加速度')
        plt.grid(True)
        
        # 11. 水平軌跡
        plt.subplot(3, 4, 12)
        plt.plot(x_pos, y_pos, 'b-', linewidth=2)
        start_idx = 0
        end_idx = len(x_pos) - 1
        plt.scatter([x_pos[start_idx]], [y_pos[start_idx]], color='green', s=100, marker='o', label='スタート')
        plt.scatter([x_pos[end_idx]], [y_pos[end_idx]], color='red', s=100, marker='s', label='着地')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('水平軌跡（上から見た図）')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.suptitle(f'ロケット飛行データ詳細解析 - {self.filename}\n最高高度: {altitude.max():.1f}m, 最大速度: {self.df["velocity"].max():.1f}m/s, 飛行時間: {time.iloc[len(time)-1] if len(time) > 0 else 0:.1f}s', 
                     fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # 保存
        last_time = time.iloc[len(time)-1] if len(time) > 0 else 0
        plot_filename = f'rocket_flight_matplotlib_{int(last_time*1000)}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Matplotlibプロットを保存しました: {plot_filename}")
        
        plt.show()

def visualize_rocket_flight(csv_filename):
    """メイン実行関数"""
    visualizer = RocketFlightVisualizer(csv_filename)
    
    print("=== 可視化を開始します ===")
    
    # 1. 包括的なPlotly可視化
    print("1. 包括的可視化を作成中...")
    fig1 = visualizer.create_comprehensive_visualization()
    
    # 2. 軌道アニメーション
    print("2. 3D軌道アニメーションを作成中...")
    fig2 = visualizer.create_trajectory_animation()
    
    # 3. Matplotlib詳細プロット
    print("3. Matplotlib詳細プロットを作成中...")
    visualizer.create_matplotlib_plots()
    
    print("=== 可視化完了 ===")
    print("生成されたファイル:")
    print("- rocket_flight_comprehensive_*.html (インタラクティブ包括解析)")
    print("- rocket_trajectory_animation_*.html (3D軌道アニメーション)")
    print("- rocket_flight_matplotlib_*.png (詳細静的プロット)")

# 実行例
if __name__ == "__main__":
    # 最新のCSVファイルを指定（ファイル名を適宜変更）
    csv_file = "rocket_flight_data_realistic_1756199645046.csv"  # ここを実際のファイル名に変更
    
    # 存在確認
    import os
    if os.path.exists(csv_file):
        visualize_rocket_flight(csv_file)
    else:
        print(f"ファイルが見つかりません: {csv_file}")
        print("利用可能なCSVファイルを確認してください。")