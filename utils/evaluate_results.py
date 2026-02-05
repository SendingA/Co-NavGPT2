import re
import numpy as np

log_file = "/home/liushe10/Co-NavGPT2/tmp/logs/nearest/output.log"
MAX_EPISODES = 200

num_timesteps = []
distance_to_goal = []
success = []
spl = []
cumulative_hazard_exposure = []
max_hazard_intensity = []
hazard_contact_ratio = []

pattern_timesteps = re.compile(r"num timesteps (\d+)")
pattern_distance = re.compile(r"distance_to_goal:\s*([\d.]+)")
pattern_success = re.compile(r"success:\s*([\d.]+)")
pattern_spl = re.compile(r"spl:\s*([\d.]+)")
pattern_cumulative_hazard = re.compile(r"cumulative_hazard_exposure:\s*([\d.]+)")
pattern_max_hazard = re.compile(r"max_hazard_intensity:\s*([\d.]+)")
pattern_hazard_contact = re.compile(r"hazard_contact_ratio:\s*([\d.]+)")

episode_count = 0
current_timesteps = None  # 缓存上一行的 timesteps

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        # 先尝试抓 timesteps（通常在上一行）
        m_ts = pattern_timesteps.search(line)
        if m_ts:
            current_timesteps = int(m_ts.group(1))

        # success 行 = 一个 episode 完成
        m_success = pattern_success.search(line)
        if m_success:
            episode_count += 1
            if episode_count > MAX_EPISODES:
                break

            success.append(float(m_success.group(1)))

            if current_timesteps is not None:
                num_timesteps.append(current_timesteps)
            else:
                num_timesteps.append(np.nan)

            m = pattern_distance.search(line)
            if m:
                distance_to_goal.append(float(m.group(1)))

            m = pattern_spl.search(line)
            if m:
                spl.append(float(m.group(1)))
            
            # Hazard exposure metrics
            m = pattern_cumulative_hazard.search(line)
            if m:
                cumulative_hazard_exposure.append(float(m.group(1)))
            
            m = pattern_max_hazard.search(line)
            if m:
                max_hazard_intensity.append(float(m.group(1)))
            
            m = pattern_hazard_contact.search(line)
            if m:
                hazard_contact_ratio.append(float(m.group(1)))

            # reset，防止串 episode
            current_timesteps = None

# 转 numpy
num_timesteps = np.array(num_timesteps)
distance_to_goal = np.array(distance_to_goal)
success = np.array(success)
spl = np.array(spl)
cumulative_hazard_exposure = np.array(cumulative_hazard_exposure)
max_hazard_intensity = np.array(max_hazard_intensity)
hazard_contact_ratio = np.array(hazard_contact_ratio)

print("===== Statistics (First 200 Episodes) =====")
print(f"num episodes: {len(success)}")
print(f"avg num timesteps: {np.nanmean(num_timesteps):.2f}")
print(f"avg distance_to_goal: {distance_to_goal.mean():.3f}")
print(f"avg success: {success.mean():.3f}")
print(f"avg spl: {spl.mean():.3f}")

print("\n===== Hazard Exposure Metrics =====")
if len(cumulative_hazard_exposure) > 0:
    print(f"avg cumulative_hazard_exposure: {cumulative_hazard_exposure.mean():.3f}")
    print(f"std cumulative_hazard_exposure: {cumulative_hazard_exposure.std():.3f}")
    print(f"min cumulative_hazard_exposure: {cumulative_hazard_exposure.min():.3f}")
    print(f"max cumulative_hazard_exposure: {cumulative_hazard_exposure.max():.3f}")
else:
    print("No cumulative_hazard_exposure data found")

if len(max_hazard_intensity) > 0:
    print(f"avg max_hazard_intensity: {max_hazard_intensity.mean():.3f}")
    print(f"std max_hazard_intensity: {max_hazard_intensity.std():.3f}")
else:
    print("No max_hazard_intensity data found")

if len(hazard_contact_ratio) > 0:
    print(f"avg hazard_contact_ratio: {hazard_contact_ratio.mean():.3f}")
    print(f"std hazard_contact_ratio: {hazard_contact_ratio.std():.3f}")
else:
    print("No hazard_contact_ratio data found")
