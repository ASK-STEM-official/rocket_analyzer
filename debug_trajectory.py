import pandas as pd
import numpy as np
from flight_trajectory_analyzer import FlightAnalyzer

# テスト用のデータ分析スクリプト
def test_trajectory_calculation():
    analyzer = FlightAnalyzer()
    
    # データを読み込み
    if not analyzer.load_data('sensor_data50.csv'):
        print("データ読み込み失敗")
        return
    
    print(f"データ行数: {len(analyzer.data)}")
    print(f"データ列: {list(analyzer.data.columns)}")
    print("\n最初の5行:")
    print(analyzer.data.head())
    
    # 軌跡計算
    trajectory = analyzer.calculate_trajectory()
    
    if trajectory:
        print("\n計算結果:")
        print(f"時間範囲: {trajectory['time'][0]:.2f} - {trajectory['time'][-1]:.2f} s")
        print(f"位置範囲 X: {np.min(trajectory['position'][:, 0]):.2f} - {np.max(trajectory['position'][:, 0]):.2f} m")
        print(f"位置範囲 Y: {np.min(trajectory['position'][:, 1]):.2f} - {np.max(trajectory['position'][:, 1]):.2f} m")
        print(f"位置範囲 Z: {np.min(trajectory['position'][:, 2]):.2f} - {np.max(trajectory['position'][:, 2]):.2f} m")
        
        print(f"\n速度範囲 X: {np.min(trajectory['velocity'][:, 0]):.2f} - {np.max(trajectory['velocity'][:, 0]):.2f} m/s")
        print(f"速度範囲 Y: {np.min(trajectory['velocity'][:, 1]):.2f} - {np.max(trajectory['velocity'][:, 1]):.2f} m/s")
        print(f"速度範囲 Z: {np.min(trajectory['velocity'][:, 2]):.2f} - {np.max(trajectory['velocity'][:, 2]):.2f} m/s")
        
        # 加速度の確認
        accel_mag = np.linalg.norm(trajectory['acceleration'], axis=1)
        print(f"\n加速度の大きさ範囲: {np.min(accel_mag):.2f} - {np.max(accel_mag):.2f} m/s²")
        
        # 生データの加速度と比較
        raw_accel = analyzer.data[['AccelX_g', 'AccelY_g', 'AccelZ_g']].values * 9.81
        raw_accel_mag = np.linalg.norm(raw_accel, axis=1)
        print(f"生データ加速度の大きさ範囲: {np.min(raw_accel_mag):.2f} - {np.max(raw_accel_mag):.2f} m/s²")
        
        # 姿勢データの確認
        attitude_deg = np.degrees(trajectory['attitude'])
        print(f"\n姿勢範囲 Roll: {np.min(attitude_deg[:, 0]):.1f} - {np.max(attitude_deg[:, 0]):.1f} °")
        print(f"姿勢範囲 Pitch: {np.min(attitude_deg[:, 1]):.1f} - {np.max(attitude_deg[:, 1]):.1f} °")
        print(f"姿勢範囲 Yaw: {np.min(attitude_deg[:, 2]):.1f} - {np.max(attitude_deg[:, 2]):.1f} °")
        
        # X-Y平面での移動距離
        horizontal_distance = np.sqrt(trajectory['position'][:, 0]**2 + trajectory['position'][:, 1]**2)
        print(f"\n水平移動距離の最大: {np.max(horizontal_distance):.2f} m")
        
        return trajectory
    else:
        print("軌跡計算失敗")
        return None

if __name__ == "__main__":
    test_trajectory_calculation()
