import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import time

class RocketFlightSimulator:
    def __init__(self):
        # 物理定数
        self.g = 9.81  # 重力加速度 [m/s²]
        self.rho_0 = 1.225  # 海面気圧での空気密度 [kg/m³]
        self.p_0 = 101325  # 海面気圧 [Pa]
        self.T_0 = 288.15  # 海面温度 [K]
        self.L = 0.0065  # 気温減率 [K/m]
        self.R = 287.05  # 気体定数 [J/kg/K]
        
        # ロケット諸元
        self.m_prop = 2.5  # 推進剤質量 [kg]
        self.m_dry = 5.0   # 乾燥質量 [kg]
        self.burn_time = 3.0  # 燃焼時間 [s]
        self.thrust = 200.0   # 推力 [N]
        self.cd = 0.5  # 抗力係数
        self.area = 0.01  # 断面積 [m²]
        self.cd_parachute = 1.3  # パラシュート抗力係数
        self.area_parachute = 1.0  # パラシュート面積 [m²]
        
        # シミュレーション設定
        self.dt = 0.02  # タイムステップ [s]
        self.parachute_deploy_altitude = 500  # パラシュート展開高度 [m]
        
    def atmospheric_conditions(self, altitude):
        """高度による大気条件の変化"""
        if altitude < 11000:
            T = self.T_0 - self.L * altitude
            p = self.p_0 * (T / self.T_0) ** (self.g / (self.R * self.L))
            rho = p / (self.R * T)
        else:
            # 成層圏での近似
            T = 216.65
            p = 22632 * math.exp(-self.g * (altitude - 11000) / (self.R * T))
            rho = p / (self.R * T)
        
        return p, rho, T
    
    def thrust_curve(self, t):
        """推力曲線（燃焼時間内で一定推力）"""
        if t <= self.burn_time:
            return self.thrust
        else:
            return 0.0
    
    def mass(self, t):
        """時間による質量変化"""
        if t <= self.burn_time:
            return self.m_dry + self.m_prop * (1 - t / self.burn_time)
        else:
            return self.m_dry
    
    def equations_of_motion(self, state, t, parachute_deployed):
        """運動方程式"""
        y, vy, theta, omega = state
        
        # 大気条件
        p, rho, T = self.atmospheric_conditions(y)
        
        # 質量と推力
        m = self.mass(t)
        F_thrust = self.thrust_curve(t)
        
        # 抗力
        if parachute_deployed and vy < 0:
            # パラシュート展開時
            F_drag = 0.5 * rho * (vy**2) * self.cd_parachute * self.area_parachute * np.sign(vy)
        else:
            # 通常の抗力
            F_drag = 0.5 * rho * (vy**2) * self.cd * self.area * np.sign(vy)
        
        # 運動方程式
        dydt = vy
        dvydt = (F_thrust - m * self.g - F_drag) / m
        dthetadt = omega
        
        # 空力モーメント（簡易モデル）
        if abs(vy) > 10:  # 十分な速度がある場合のみ
            domegadt = -0.1 * omega + np.random.normal(0, 0.1)  # 減衰 + ノイズ
        else:
            domegadt = np.random.normal(0, 0.05)  # 低速時のランダムな回転
        
        return [dydt, dvydt, dthetadt, domegadt]
    
    def simulate_flight(self):
        """飛行シミュレーション実行"""
        # 初期条件
        y0 = 0.0    # 高度 [m]
        vy0 = 0.0   # 鉛直速度 [m/s]
        theta0 = 0.0  # 姿勢角 [rad]
        omega0 = 0.0  # 角速度 [rad/s]
        
        initial_state = [y0, vy0, theta0, omega0]
        
        # 時間配列
        t_max = 60.0  # 最大シミュレーション時間
        t = np.arange(0, t_max, self.dt)
        
        # データ保存用
        results = []
        parachute_deployed = False
        parachute_deploy_time = None
        
        current_state = initial_state
        
        for i, time_val in enumerate(t):
            # パラシュート展開判定
            altitude = current_state[0]
            velocity = current_state[1]
            
            if (not parachute_deployed and 
                altitude > self.parachute_deploy_altitude and 
                velocity < 0 and 
                time_val > self.burn_time):
                parachute_deployed = True
                parachute_deploy_time = time_val
                print(f"パラシュート展開: 時刻 {time_val:.1f}s, 高度 {altitude:.1f}m")
            
            # 運動方程式を積分（数値安定性を向上）
            if i < len(t) - 1:
                dt_step = t[i+1] - t[i]
                try:
                    # より安定な積分設定
                    next_state = odeint(self.equations_of_motion, current_state, 
                                     [time_val, time_val + dt_step], 
                                     args=(parachute_deployed,),
                                     rtol=1e-6, atol=1e-9)
                    current_state = next_state[1]
                except:
                    # 積分が失敗した場合は簡易オイラー法で近似
                    derivatives = self.equations_of_motion(current_state, time_val, parachute_deployed)
                    current_state = [current_state[i] + derivatives[i] * dt_step for i in range(4)]
            
            # 大気条件
            p, rho, T = self.atmospheric_conditions(altitude)
            
            # 加速度計算
            m = self.mass(time_val)  # ← 修正箇所1: timeをtime_valに変更
            F_thrust = self.thrust_curve(time_val)  # ← 修正箇所2: timeをtime_valに変更
            
            if parachute_deployed and velocity < 0:
                F_drag = 0.5 * rho * (velocity**2) * self.cd_parachute * self.area_parachute * np.sign(velocity)
            else:
                F_drag = 0.5 * rho * (velocity**2) * self.cd * self.area * np.sign(velocity)
            
            acceleration = (F_thrust - m * self.g - F_drag) / m
            
            # 3軸加速度（簡易モデル）
            ax = acceleration * np.sin(current_state[2]) + np.random.normal(0, 0.1)
            ay = np.random.normal(0, 0.1)  # 横方向の微小振動
            az = acceleration * np.cos(current_state[2]) + self.g + np.random.normal(0, 0.1)
            
            # 3軸角速度
            gx = current_state[3] + np.random.normal(0, 0.01)
            gy = np.random.normal(0, 0.01)
            gz = np.random.normal(0, 0.01)
            
            # データ記録（解析コードと互換性のある形式）
            results.append({
                'Time_s': time_val,
                'AccelX_g': ax / 9.81,  # g単位に変換
                'AccelY_g': ay / 9.81,
                'AccelZ_g': az / 9.81,
                'GyroX_deg_s': gx,
                'GyroY_deg_s': gy,
                'GyroZ_deg_s': gz,
                'Pressure_hPa': p / 100,  # PaをhPaに変換
                'Temperature_C': T - 273.15,
                'Humidity_%': 50 + np.random.normal(0, 10),  # 仮の湿度データ
                'TotalAccel': np.sqrt(ax**2 + ay**2 + az**2) / 9.81,  # g単位
                'FlightStarted': 1 if time_val > 0.5 else 0,  # 飛行開始フラグ
                # 追加情報（解析には使用されないが参考用）
                'altitude': altitude,
                'velocity': velocity,
                'theta': math.degrees(current_state[2]),
                'angular_velocity': math.degrees(current_state[3]),
                'mass': m,
                'thrust': F_thrust,
                'drag': F_drag,
                'parachute_deployed': parachute_deployed
            })
            
            # 着地判定
            if altitude <= 0 and time_val > 5:
                print(f"着地: 時刻 {time_val:.1f}s")
                break
        
        return pd.DataFrame(results)

# シミュレーション実行
def run_simulation():
    simulator = RocketFlightSimulator()
    df = simulator.simulate_flight()
    
    # CSVファイルに保存（解析コードと互換性のある形式）
    output_filename = f'rocket_flight_data_realistic_{int(time.time() * 1000)}.csv'
    df.to_csv(output_filename, index=False)
    print(f"データを保存しました: {output_filename} ({len(df)} データポイント)")
    
    # 基本統計
    print(f"\n=== 飛行統計 ===")
    print(f"最高高度: {df['altitude'].max():.1f} m")
    print(f"最高速度: {df['velocity'].max():.1f} m/s")
    print(f"飛行時間: {df['Time_s'].iloc[-1]:.1f} s")
    print(f"最低気圧: {df['Pressure_hPa'].min():.1f} hPa")
    print(f"最大加速度: {df['TotalAccel'].max():.1f} g")
    
    # グラフ描画
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ロケット飛行データ', fontsize=16)
    
    # 高度
    axes[0,0].plot(df['Time_s'], df['altitude'])
    axes[0,0].set_title('高度 vs 時間')
    axes[0,0].set_xlabel('時間 [s]')
    axes[0,0].set_ylabel('高度 [m]')
    axes[0,0].grid(True)
    
    # 速度
    axes[0,1].plot(df['Time_s'], df['velocity'])
    axes[0,1].set_title('速度 vs 時間')
    axes[0,1].set_xlabel('時間 [s]')
    axes[0,1].set_ylabel('速度 [m/s]')
    axes[0,1].grid(True)
    
    # 加速度
    axes[0,2].plot(df['Time_s'], df['AccelX_g'], label='ax')
    axes[0,2].plot(df['Time_s'], df['AccelY_g'], label='ay') 
    axes[0,2].plot(df['Time_s'], df['AccelZ_g'], label='az')
    axes[0,2].set_title('3軸加速度')
    axes[0,2].set_xlabel('時間 [s]')
    axes[0,2].set_ylabel('加速度 [g]')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # 角速度
    axes[1,0].plot(df['Time_s'], df['GyroX_deg_s'], label='gx')
    axes[1,0].plot(df['Time_s'], df['GyroY_deg_s'], label='gy')
    axes[1,0].plot(df['Time_s'], df['GyroZ_deg_s'], label='gz')
    axes[1,0].set_title('3軸角速度')
    axes[1,0].set_xlabel('時間 [s]')
    axes[1,0].set_ylabel('角速度 [deg/s]')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # 気圧
    axes[1,1].plot(df['Time_s'], df['Pressure_hPa'])
    axes[1,1].set_title('気圧 vs 時間')
    axes[1,1].set_xlabel('時間 [s]')
    axes[1,1].set_ylabel('気圧 [hPa]')
    axes[1,1].grid(True)
    
    # 温度
    axes[1,2].plot(df['Time_s'], df['Temperature_C'])
    axes[1,2].set_title('温度 vs 時間')
    axes[1,2].set_xlabel('時間 [s]')
    axes[1,2].set_ylabel('温度 [°C]')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('rocket_flight_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

# 実行
if __name__ == "__main__":
    df = run_simulation()
    print("\nデータの最初の10行:")
    print(df.head(10))