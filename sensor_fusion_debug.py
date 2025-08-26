import numpy as np
import matplotlib.pyplot as plt
from flight_trajectory_analyzer import FlightAnalyzer

def debug_sensor_fusion():
    analyzer = FlightAnalyzer()
    
    if not analyzer.load_data('sensor_data50.csv'):
        print("データ読み込み失敗")
        return
    
    print("=== センサー融合デバッグ ===")
    print(f"データ点数: {len(analyzer.data)}")
    
    # 軌跡計算実行
    trajectory = analyzer.calculate_trajectory()
    
    if trajectory is None:
        print("軌跡計算失敗")
        return
    
    # 結果の確認
    time = trajectory['time']
    pos = trajectory['position']
    vel = trajectory['velocity']
    att = trajectory['attitude']
    
    print(f"\n=== 結果サマリー ===")
    print(f"飛行時間: {time[-1] - time[0]:.2f} s")
    print(f"最大高度: {np.max(pos[:, 2]):.2f} m")
    print(f"最大速度: {np.max(np.linalg.norm(vel, axis=1)):.2f} m/s")
    print(f"最終位置: X={pos[-1, 0]:.2f}m, Y={pos[-1, 1]:.2f}m, Z={pos[-1, 2]:.2f}m")
    print(f"水平移動距離: {np.sqrt(pos[-1, 0]**2 + pos[-1, 1]**2):.2f} m")
    
    print(f"\n=== 姿勢データ（度） ===")
    att_deg = np.degrees(att)
    print(f"Roll範囲: {np.min(att_deg[:, 0]):.1f} ~ {np.max(att_deg[:, 0]):.1f} °")
    print(f"Pitch範囲: {np.min(att_deg[:, 1]):.1f} ~ {np.max(att_deg[:, 1]):.1f} °")
    print(f"Yaw範囲: {np.min(att_deg[:, 2]):.1f} ~ {np.max(att_deg[:, 2]):.1f} °")
    
    # 簡易プロット
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(time, pos[:, 2])
    plt.title('高度変化')
    plt.xlabel('時間 (s)')
    plt.ylabel('高度 (m)')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    vel_mag = np.linalg.norm(vel, axis=1)
    plt.plot(time, vel_mag)
    plt.title('速度の大きさ')
    plt.xlabel('時間 (s)')
    plt.ylabel('速度 (m/s)')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(pos[:, 0], pos[:, 1])
    plt.title('水平軌跡 (XY)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(time, att_deg[:, 0], label='Roll')
    plt.plot(time, att_deg[:, 1], label='Pitch')
    plt.plot(time, att_deg[:, 2], label='Yaw')
    plt.title('姿勢変化')
    plt.xlabel('時間 (s)')
    plt.ylabel('角度 (°)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(time, vel[:, 0], label='Vx')
    plt.plot(time, vel[:, 1], label='Vy')
    plt.plot(time, vel[:, 2], label='Vz')
    plt.title('速度成分')
    plt.xlabel('時間 (s)')
    plt.ylabel('速度 (m/s)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # 3D軌跡の投影
    plt.plot(time, np.sqrt(pos[:, 0]**2 + pos[:, 1]**2), label='水平距離')
    plt.plot(time, pos[:, 2], label='高度')
    plt.title('軌跡概要')
    plt.xlabel('時間 (s)')
    plt.ylabel('距離 (m)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensor_fusion_debug.png', dpi=150)
    plt.show()
    
    return trajectory

if __name__ == "__main__":
    debug_sensor_fusion()
