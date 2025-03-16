import os
import sys
import fastf1 # (3.5.2)
import matplotlib.cm as cm
from IPython.display import clear_output
try:
    fastf1.Cache.enable_cache(sys.path[0]+"/fastf1_cache")
except:
    os.makedirs(sys.path[0]+"/fastf1_cache")
    fastf1.Cache.enable_cache(sys.path[0]+"/fastf1_cache")
    
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import datetime
import seaborn as sns
sns.set_style("darkgrid")
import pandas as pd
import numpy as np
from windrose import WindroseAxes
pd.set_option('display.max_columns', None)

# Custom input function with default value
def input_with_default(prompt, default):
    user_input = input(f"{prompt} - per default [{default}]: ")
    return user_input if user_input else default

# input session and year
#year = input_with_default("Enter the year", 2025)
year = int(input_with_default("Enter the year",2025))
session = str(input_with_default("Session Type",'Qualifying'))  # 'Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Sprint Qualifying', 'Race'

# Function to list events for a given year
def list_events_for_year(year):
    # Get the event schedule for the specified year
    events = fastf1.get_event_schedule(year,include_testing=False)

    # Debugging: Print the entire events list to understand its structure
    #print(f"Raw events data for {year}:", events)

    # Convert the DataFrame to a list of dictionaries
    events_list = events.to_dict(orient='records')

    # Print the list of events with their round numbers, names, and locations
    print(f"Events for {year}:")
    for idx, event in enumerate(events_list, start=1):
        print(f"{idx}. {event['EventName']} - {event['Location']} (Round {event['RoundNumber']})")

    return events_list

def select_event_by_round_number(events, round_number):
    # Find the event with the specified round number
    for event in events:
        if event['RoundNumber'] == round_number:
            return event
    return None

print(f"Selected Session : {session}")

# Prompt the user to enter the round number of the event they want to select
events = list_events_for_year(year)   
print(f"{list_events_for_year}")
round_number = int(input("Enter the round number of the event you want to select: "))
selected_event = select_event_by_round_number(events, round_number)

# Print the select event for the year
if selected_event:
    print(f"-------------------------------------------------------------------------------------------------------------------")
    print(f"Selected {year} Event: {selected_event['EventName']} - {selected_event['Location']} (Round {selected_event['RoundNumber']})")
    print(f"Selected Session : {session}")
    print(f"-------------------------------------------------------------------------------------------------------------------")
else:
    print("No event found with the specified round number.")
    
race = fastf1.get_session(year, round_number, session)
race.load(weather=True)

# load race laps
race_name = race.event.OfficialEventName
df = race.laps

# Create the necessary directories
img_folder = f"{year}/{race_name}/{session}/png"
os.makedirs(img_folder, exist_ok=True)

# load dataframe of df (by Final Position in ascending order)
df = df[df['Deleted']==False] # only included laps that are legit (not deleted by race control)
df = df.sort_values(by=['LapNumber','Position'], ascending=[False, True]).reset_index(drop=True)

# fill in empty laptime records and convert to seconds
df.LapTime = df.LapTime.fillna(df['Sector1Time']+df['Sector2Time']+df['Sector3Time'])
df.LapTime = df.LapTime.dt.total_seconds()
df.Sector1Time = df.Sector1Time.dt.total_seconds()
df.Sector2Time = df.Sector2Time.dt.total_seconds()
df.Sector3Time = df.Sector3Time.dt.total_seconds()

# weather
df_weather = race.weather_data.copy()
df_weather['Time'] = df_weather['Time'].dt.total_seconds()/60
df_weather = df_weather.rename(columns={'Time':'SessionTime(Minutes)'})

# Rain Indicator
rain = df_weather.Rainfall.eq(True).any()
# Results
if session in ('Q','SQ','Qualifying','Sprint Qualifying'):
    df_results = race.results
    df_results['BestQTime'] = round(df_results.apply(lambda row: min(row.Q1, row.Q2, row.Q3), axis=1).dt.total_seconds(),3)
else:
    df_results = df.loc[df.groupby(['Driver'])['LapTime'].idxmin()].sort_values('LapTime', ascending=True).rename(columns={'Driver':'Abbreviation'}).reset_index()
    df_results['BestQTime'] = df_results['LapTime']

df_results['GapToBest'] = df_results.apply(lambda row: round(row.BestQTime - df_results.BestQTime.min(),3), axis=1)
df_results = df_results.sort_values('GapToBest')

# for driver color palette
driver_color = {}
for index, row in df_results.iterrows():
    driver = row['Abbreviation']
    try:
        if driver == 'RIC':
            driver_color[driver] = fastf1.plotting.get_driver_color('ALO',session=race)
        else:
            driver_color[driver] = fastf1.plotting.get_driver_color(driver,session=race)
    except Exception as e:
        print(f"Error getting color for driver {driver}: {e}")
        driver_color[driver] = 'pink'  # Default color in case of error

abs_diff_activated = 'enable'
### Plotting
plt.style.use('dark_background')
plt.figure(figsize=(10,10))
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
sns.barplot(data=df_results, x='GapToBest', y='Abbreviation', palette=driver_color, hue='Abbreviation', edgecolor='black', legend=False)
plt.ylabel('Driver')
plt.xlabel('Time(s)')
plt.title(f'Gap to Best Driver(s) \nFastest Driver: {df_results.Abbreviation.iloc[0]} {int(df_results.BestQTime.iloc[0]/60)}:{round(df_results.BestQTime.iloc[0]-60,3)}\n{race_name}\n{session}', fontsize=14, color='white')

# to add data labels
plt.text(df_results.GapToBest.iloc[0]+0.01, 0.15, f'{int(df_results.BestQTime.iloc[0]/60)}:{round(df_results.BestQTime.iloc[0]-60,3)}', ha = 'left') # for 1st driver
for i in range(1,len(df_results)):
    plt.text(df_results.GapToBest.iloc[i]+0.01, i+0.15, df_results.GapToBest.iloc[i], ha = 'left')
plt.tight_layout()
plt.savefig(f"{img_folder}/01-Gap_to_Best_Driver.png")

# get telemetry data
driver_1_lap = df.loc[df.pick_drivers(df_results.Abbreviation.iloc[0])['LapTime'].idxmin()]
driver_1_tel = driver_1_lap.get_telemetry()
driver_1_tel['Driver'] = df_results.Abbreviation.iloc[0]
driver_2_lap = df.loc[df.pick_drivers(df_results.Abbreviation.iloc[1])['LapTime'].idxmin()]
driver_2_tel = driver_2_lap.get_telemetry()
driver_2_tel['Driver'] = df_results.Abbreviation.iloc[1]
driver_3_lap = df.loc[df.pick_drivers(df_results.Abbreviation.iloc[2])['LapTime'].idxmin()]
driver_3_tel = driver_3_lap.get_telemetry()
driver_3_tel['Driver'] = df_results.Abbreviation.iloc[2]

# Data transform

# Join all 3 drivers data
telemetry = pd.concat([driver_1_tel, driver_2_tel, driver_3_tel])

# Create minisectors
total_minisectors = 40
telemetry['Minisector'] = pd.cut(telemetry['Distance'], total_minisectors, labels=False) + 1

# Calculate time delta for top 2 drivers
time_used = telemetry.groupby(['Minisector', 'Driver'])['Time'].mean().reset_index().rename(columns={'Time': 'BestTime'})
time_used['BestDriverTime'] = time_used.apply(lambda x: time_used.loc[(time_used.Driver == df_results.Abbreviation.iloc[0]) & (time_used.Minisector == x.Minisector), 'BestTime'].min(), axis=1)
time_used['TimeDelta'] = time_used['BestTime'].dt.total_seconds() - time_used['BestDriverTime'].dt.total_seconds()

# Convert Speed from km/h to m/s
driver_1_tel['Speed_m_s'] = driver_1_tel['Speed'] * (1000 / 3600)
driver_2_tel['Speed_m_s'] = driver_2_tel['Speed'] * (1000 / 3600)
driver_3_tel['Speed_m_s'] = driver_3_tel['Speed'] * (1000 / 3600)

# Calculate acceleration in m/s² (difference in speed over difference in time)
driver_1_tel['Accel_m_s2'] = driver_1_tel['Speed_m_s'].diff() / driver_1_tel['Time'].diff().dt.total_seconds()
driver_2_tel['Accel_m_s2'] = driver_2_tel['Speed_m_s'].diff() / driver_2_tel['Time'].diff().dt.total_seconds()
driver_3_tel['Accel_m_s2'] = driver_3_tel['Speed_m_s'].diff() / driver_3_tel['Time'].diff().dt.total_seconds()

# Convert acceleration to G (1 G = 9.81 m/s²)
driver_1_tel['Accel_G'] = driver_1_tel['Accel_m_s2'] / 9.81
driver_2_tel['Accel_G'] = driver_2_tel['Accel_m_s2'] / 9.81
driver_3_tel['Accel_G'] = driver_3_tel['Accel_m_s2'] / 9.81

# Create the figure and axes (4x2 grid)
fig, ax = plt.subplots(4, 2, figsize=(30, 20))  # 4 rows, 2 columns
fig.suptitle(f"Telemetry Data of 2 Fastest Drivers - {session}\n{race_name}", fontsize=14, color='white')
fig.supxlabel('Distance', color='white')
for axis in ax.flat:
    axis.grid(True, linestyle='-', color='gray', alpha=0.5)

# Time Delta (row 0, column 0)
second_driver_time_delta = [0] + time_used.loc[time_used.Driver == df_results.Abbreviation.iloc[1], 'TimeDelta'].to_list() + [df_results.GapToBest.iloc[1]]
distance_array = [0] + [i * telemetry.Distance.max() / total_minisectors for i in range(1, total_minisectors + 1)] + [telemetry.Distance.max()]
ax[0, 0].plot(distance_array, second_driver_time_delta, color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[0, 0].axhline(y=0, color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0])

# Annotating the max value for the second driver
max_time_delta = max(second_driver_time_delta)
ax[0, 0].annotate(f'Max: {max_time_delta:.2f}s',
                  xy=(distance_array[second_driver_time_delta.index(max_time_delta)], max_time_delta),
                  xytext=(10, 5),
                  textcoords='offset points',
                  arrowprops=dict(arrowstyle="->", color='red'),
                  color='red')

ax[0, 0].text(max(distance_array) + 20, df_results.GapToBest.iloc[1], f"{df_results.GapToBest.iloc[1]}s", color='white')
ax[0, 0].legend(loc='lower left', facecolor='black', edgecolor='white')
ax[0, 0].set_title('Time Delta', fontweight="bold", color='white')
ax[0, 0].set_ylabel('Seconds(s)', color='white')
ax[0, 0].set_xlim(0, telemetry.Distance.max())

# Speed (row 0, column 1)
ax[0, 1].plot(driver_2_tel['Distance'], driver_2_tel['Speed'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[0, 1].plot(driver_1_tel['Distance'], driver_1_tel['Speed'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')

if abs_diff_activated =='enable':
    # Create secondary y-axis for absolute difference
    ax2 = ax[0, 1].twinx()
    abs_diff = np.abs(driver_2_tel['Speed'] - driver_1_tel['Speed'])
    ax2.plot(driver_2_tel['Distance'], abs_diff, color='white', linestyle='dashed', label='Absolute Difference')
    ax2.set_ylabel('Absolute Difference', color='white')

# Annotating the max value for both drivers
max_speed_1 = driver_1_tel['Speed'].max()
max_speed_2 = driver_2_tel['Speed'].max()

ax[0, 1].scatter(driver_1_tel['Distance'][driver_1_tel['Speed'].idxmax()], max_speed_1, color='blue', zorder=5)
ax[0, 1].scatter(driver_2_tel['Distance'][driver_2_tel['Speed'].idxmax()], max_speed_2, color='orange', zorder=5)

ax[0, 1].text(driver_1_tel['Distance'][driver_1_tel['Speed'].idxmax()] + 10, max_speed_1, f'Max: {max_speed_1:.2f} km/h', color='blue')
ax[0, 1].text(driver_2_tel['Distance'][driver_2_tel['Speed'].idxmax()] + 10, max_speed_2, f'Max: {max_speed_2:.2f} km/h', color='orange')
ax[0, 1].set_xlim(0, telemetry.Distance.max())
ax[0, 1].legend(loc='lower left', facecolor='black', edgecolor='white')
ax[0, 1].set_title('Speed', fontweight="bold", color='white')
ax[0, 0].set_ylabel('Speed(kph)', color='white')

# DRS (row 1, column 0)
driver_1_tel['DRS_Activated'] = driver_1_tel['DRS'].map(lambda x: 'ON' if x >= 10 else 'OFF')
driver_2_tel['DRS_Activated'] = driver_2_tel['DRS'].map(lambda x: 'ON' if x >= 10 else 'OFF')
ax[1, 0].plot(driver_2_tel['Distance'], driver_2_tel['DRS_Activated'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[1, 0].plot(driver_1_tel['Distance'], driver_1_tel['DRS_Activated'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')
ax[1, 0].invert_yaxis()
ax[1, 0].set_xlim(0, telemetry.Distance.max())
ax[1, 0].set_title('DRS', fontweight="bold", color='white')
ax[1, 0].set_ylabel('DRS', color='white')

# Throttle (row 1, column 1)
ax[1, 1].plot(driver_2_tel['Distance'], driver_2_tel['Throttle'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[1, 1].plot(driver_1_tel['Distance'], driver_1_tel['Throttle'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')
ax[1, 1].set_title('Throttle', fontweight="bold", color='white')
ax[1, 1].set_ylabel('Throttle(%)', color='white')
ax[1, 1].set_xlim(0, telemetry.Distance.max())

if abs_diff_activated =='enable':
    # Create secondary y-axis for absolute difference
    ax2 = ax[1, 1].twinx()
    abs_diff = np.abs(driver_2_tel['Throttle'] - driver_1_tel['Throttle'])
    ax2.plot(driver_2_tel['Distance'], abs_diff, color='white', linestyle='dashed', label='Absolute Difference')
    ax2.set_ylabel('Absolute Difference', color='white')
    
# Brake (row 2, column 0)
ax[2, 0].plot(driver_2_tel['Distance'], driver_2_tel['Brake'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[2, 0].plot(driver_1_tel['Distance'], driver_1_tel['Brake'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')
ax[2, 0].set_yticks(ticks=[1, 0], labels=["YES", "NO"])
ax[2, 0].set_xlim(0, telemetry.Distance.max())
ax[2, 0].set_title('Brake', fontweight="bold", color='white')
ax[2, 0].set_ylabel('Brake(YES/NO)', color='white')

if abs_diff_activated =='enable':
    # Create secondary y-axis for absolute difference
    ax2 = ax[2, 0].twinx()
    abs_diff = np.abs(driver_2_tel['Brake'] - driver_1_tel['Brake'])
    ax2.plot(driver_2_tel['Distance'], abs_diff, color='white', linestyle='dashed', label='Absolute Difference')
    ax2.set_ylabel('Absolute Difference', color='white')

# Gear (row 2, column 1)
ax[2, 1].plot(driver_2_tel['Distance'], driver_2_tel['nGear'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[2, 1].plot(driver_1_tel['Distance'], driver_1_tel['nGear'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')
ax[2, 1].set_title('Gear', fontweight="bold", color='white')
ax[2, 1].set_ylabel('Gear', color='white')
ax[2, 1].set_xlim(0, telemetry.Distance.max())

# RPM (row 3, column 0)
ax[3, 0].plot(driver_2_tel['Distance'], driver_2_tel['RPM'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[3, 0].plot(driver_1_tel['Distance'], driver_1_tel['RPM'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')

if abs_diff_activated =='enable':
    # Create secondary y-axis for absolute difference
    ax2 = ax[3, 0].twinx()
    abs_diff = np.abs(driver_2_tel['RPM'] - driver_1_tel['RPM'])
    ax2.plot(driver_2_tel['Distance'], abs_diff, color='white', linestyle='dashed', label='Absolute Difference')
    ax2.set_ylabel('Absolute Difference', color='white')


# Annotating the max RPM value for both drivers.
max_rpm_1 = driver_1_tel['RPM'].max()
max_rpm_2 = driver_2_tel['RPM'].max()
ax[3, 0].scatter(driver_1_tel['Distance'][driver_1_tel['RPM'].idxmax()], max_rpm_1, color='blue', zorder=5)
ax[3, 0].scatter(driver_2_tel['Distance'][driver_2_tel['RPM'].idxmax()], max_rpm_2, color='orange', zorder=5)
ax[3, 0].set_xlim(0, telemetry.Distance.max())
ax[3, 0].set_title('RPM', fontweight="bold", color='white')
ax[3, 0].set_ylabel('RPM', color='white')

# Acceleration in G (row 3, column 1)
ax[3, 1].plot(driver_2_tel['Distance'], driver_2_tel['Accel_G'], color=driver_color[df_results.Abbreviation.iloc[1]], label=df_results.Abbreviation.iloc[1])
ax[3, 1].plot(driver_1_tel['Distance'], driver_1_tel['Accel_G'], color=driver_color[df_results.Abbreviation.iloc[0]], label=df_results.Abbreviation.iloc[0], alpha=0.6, linestyle='dotted')

# Annotating the max acceleration value for both drivers
max_accel_1 = driver_1_tel['Accel_G'].max()
max_accel_2 = driver_2_tel['Accel_G'].max()
min_accel_1 = driver_1_tel['Accel_G'].min()
min_accel_2 = driver_2_tel['Accel_G'].min()

ax[3, 1].scatter(driver_1_tel['Distance'][driver_1_tel['Accel_G'].idxmax()], max_accel_1, color='blue', zorder=5)
ax[3, 1].scatter(driver_2_tel['Distance'][driver_2_tel['Accel_G'].idxmax()], max_accel_2, color='orange', zorder=5)
ax[3, 1].scatter(driver_1_tel['Distance'][driver_1_tel['Accel_G'].idxmin()], min_accel_1, color='green', zorder=5)
ax[3, 1].scatter(driver_2_tel['Distance'][driver_2_tel['Accel_G'].idxmin()], min_accel_2, color='black', zorder=5)

ax[3, 1].text(driver_1_tel['Distance'][driver_1_tel['Accel_G'].idxmax()] + 10, max_accel_1, f'Max: {max_accel_1:.2f} G', color='blue')
ax[3, 1].text(driver_2_tel['Distance'][driver_2_tel['Accel_G'].idxmax()] + 10, max_accel_2, f'Max: {max_accel_2:.2f} G', color='orange')
ax[3, 1].text(driver_1_tel['Distance'][driver_1_tel['Accel_G'].idxmin()] + 10, min_accel_1, f'Min: {min_accel_1:.2f} G', color='green')
ax[3, 1].text(driver_2_tel['Distance'][driver_2_tel['Accel_G'].idxmin()] + 10, min_accel_2, f'Min: {min_accel_2:.2f} G', color='black')

ax[3, 1].set_title('Longitudinal Acceleration', fontweight="bold", color='white')
ax[3, 1].set_ylabel('Acceleration (G)', color='white')
ax[3, 1].set_xlim(0, telemetry.Distance.max())

# Adjust layout for better spacing
plt.tight_layout()  # To ensure the title is not cut off
plt.savefig(f"{img_folder}/02-Telemetry_Data.png")

### Track Dominance
fig, ax = plt.subplots(1, figsize=(10, 10)) 
average_speed = telemetry.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()
best_sectors = average_speed.groupby(['Minisector'])['Speed'].max().reset_index()
best_sectors = best_sectors.merge(average_speed[['Speed','Driver']], on=['Speed']).rename(columns={'Driver':'FastestSectorDriver', 'Speed':'FastestSectorSpeed'})
track_dominance = best_sectors.FastestSectorDriver.value_counts(normalize=True)
best_sectors = best_sectors.merge(telemetry, on=['Minisector'])

# Get Lap Data
single_lap = telemetry.loc[telemetry['Driver'] == df_results.Abbreviation.iloc[0]]
lap_x = np.array(single_lap['X'].values)
lap_y = np.array(single_lap['Y'].values)

# points and segments for drawing lap
points = np.array([lap_x, lap_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Get Fastest Driver of each minisector
best_sectors['FastestSectorDriver'] = best_sectors['FastestSectorDriver'].map({df_results.Abbreviation.iloc[0]:1,df_results.Abbreviation.iloc[1]:2,df_results.Abbreviation.iloc[2]:3})
map_sectors = best_sectors.loc[best_sectors['Driver'] == df_results.Abbreviation.iloc[0]]

# getting colormap
colors = [driver_color[df_results.Abbreviation.iloc[i]] for i in range(3)]
cmap = ListedColormap(colors)

# coordinates
lc_comp = LineCollection(segments, norm = plt.Normalize(1, cmap.N), cmap = cmap)
lc_comp.set_array(map_sectors['FastestSectorDriver'])
lc_comp.set_linewidth(4)

### Plot Track Dominance
plt.rcParams['figure.figsize'] = [10,10]
plt.title(f"Track Dominance- {session}\n{race_name}", fontsize=20)
plt.xlabel('On Fastest Lap of Top 3 Drivers')
plt.gca().add_collection(lc_comp)
plt.arrow(lap_x[0], lap_y[0], lap_x[5]-lap_x[0], lap_y[5]-lap_y[0], width=150, color='black', zorder=100)
plt.axis('equal')
plt.grid(False)
plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
legend_lines = [Line2D([0], [0], color = driver_color[df_results.Abbreviation.iloc[i]], lw = 1) for i in range(3)]
plt.legend(legend_lines,
           [f"{df_results.Abbreviation.iloc[0]} | {round(track_dominance[df_results.Abbreviation.iloc[0]]*100,1)}%",
            f"{df_results.Abbreviation.iloc[1]} | {round(track_dominance[df_results.Abbreviation.iloc[1]]*100,1)}%",
            f"{df_results.Abbreviation.iloc[2]} | {round(track_dominance[df_results.Abbreviation.iloc[2]]*100,1)}%"])
plt.tight_layout()
plt.savefig(f"{img_folder}/03-Track_Dominance.png")

### Get the top 10 fastest Drivers time per Sector 1/2/3
top_10_sector1 = df.groupby(['Driver'])['Sector1Time'].min().sort_values().head(10).reset_index()
top_10_sector2 = df.groupby(['Driver'])['Sector2Time'].min().sort_values().head(10).reset_index()
top_10_sector3 = df.groupby(['Driver'])['Sector3Time'].min().sort_values().head(10).reset_index()

fig, ax = plt.subplots(1,3, figsize=(20, 10))
fig.suptitle(f"Fastest Sector Time - {session}\n{race_name}", fontsize=20)
for axis in ax.flat:
    axis.grid(True, linestyle='--', color='gray', alpha=0.5)

# Sector 1
sns.barplot(x=top_10_sector1['Sector1Time'], y=top_10_sector1['Driver'], palette=driver_color, hue=top_10_sector1['Driver'], ax=ax[0], edgecolor='black', legend=False)
ax[0].bar_label(ax[0].containers[0], padding=3)
ax[0].set_xlim(top_10_sector1.Sector1Time.iloc[0]-0.1,top_10_sector1.Sector1Time.iloc[9]+0.1)
ax[0].set_title('Sector 1', fontweight="bold")

# Sector 2
sns.barplot(x=top_10_sector2['Sector2Time'], y=top_10_sector2['Driver'], palette=driver_color, hue=top_10_sector2['Driver'], ax=ax[1], edgecolor='black', legend=False)
ax[1].bar_label(ax[1].containers[0], padding=3)
ax[1].set_xlim(top_10_sector2.Sector2Time.iloc[0]-0.1,top_10_sector2.Sector2Time.iloc[9]+0.1)
ax[1].set_title('Sector 2', fontweight="bold")

# Sector 3
sns.barplot(x=top_10_sector3['Sector3Time'], y=top_10_sector3['Driver'], palette=driver_color, hue=top_10_sector3['Driver'], ax=ax[2], edgecolor='black', legend=False)
ax[2].bar_label(ax[2].containers[0], padding=3)
ax[2].set_xlim(top_10_sector3.Sector3Time.iloc[0]-0.1,top_10_sector3.Sector3Time.iloc[9]+0.1)
ax[2].set_title('Sector 3', fontweight="bold")
plt.tight_layout()
plt.savefig(f"{img_folder}/04-Fastest_Sector_Time.png")

### Top speed and min speed of each team

team_max_speed = {}
team_min_speed = {}

for team in set(df.Team):
    team_max_speed[team] = df.pick_teams(team).pick_fastest().get_telemetry().Speed.max()
    team_min_speed[team] = df.pick_teams(team).pick_fastest().get_telemetry().Speed.min()

team_max_speed = pd.DataFrame(team_max_speed.items(), columns=['Team', 'Max Speed']).sort_values('Max Speed', ascending=False).reset_index()
team_min_speed = pd.DataFrame(team_min_speed.items(), columns=['Team', 'Min Speed']).sort_values('Min Speed', ascending=False).reset_index()

# for colour palette
team_color = {}
for team in team_max_speed.Team:
    team_color[team] = fastf1.plotting.get_team_color(team,session=race)

fig, ax = plt.subplots(2, figsize=(15, 15))
fig.suptitle(f"Max and Min Speed (Fastest Lap) - {session}\n{race_name}", fontsize=20)

# Max Speed
sns.barplot(data=team_max_speed, x='Team', y='Max Speed', palette=team_color, hue='Team', ax=ax[0], edgecolor='black',legend=False)
ax[0].set_ylim(team_max_speed['Max Speed'].min()-5, team_max_speed['Max Speed'].max()+1)
ax[0].set_title('Maximum Speed(km/h)', fontweight="bold")
# to add data labels
for i in range(len(team_max_speed)):
    ax[0].text(i, team_max_speed['Max Speed'][i]+0.2, team_max_speed['Max Speed'][i], ha = 'center')  

# Min Speed
sns.barplot(data=team_min_speed, x='Team', y='Min Speed', palette=team_color, hue='Team', ax=ax[1], edgecolor='black',legend=False)
ax[1].set_ylim(team_min_speed['Min Speed'].min()-5, team_min_speed['Min Speed'].max()+1)
ax[1].set_title('Minimum Speed(km/h)', fontweight="bold")
ax[1].invert_yaxis()
ax[1].xaxis.tick_top()
ax[1].xaxis.set_label_position('top')
 
# to add data labels
for i in range(len(team_min_speed)):
    ax[1].text(i, team_min_speed['Min Speed'][i]+0.7, team_min_speed['Min Speed'][i], ha = 'center')  
plt.tight_layout()
plt.savefig(f"{img_folder}/05-Teams_Max_Min_Speed.png")

#### Weather Data

fig, ax = plt.subplots(2,2, figsize=(15, 15))
fig.suptitle("Weather Data & Track Evolution - {session}\n{race_name}", fontsize=30)
for axis in ax.flat:
    axis.grid(True, linestyle='--', color='gray', alpha=0.5)

sns.lineplot(data = df_weather, x='SessionTime(Minutes)', y='TrackTemp', label = 'TrackTemp', ax = ax[0,0])
sns.lineplot(data = df_weather, x='SessionTime(Minutes)', y='AirTemp', label = 'AirTemp', ax = ax[0,0])
if rain:
    ax[0,0].fill_between(df_weather[df_weather.Rainfall == True]['SessionTime(Minutes)'], df_weather.TrackTemp.max()+0.5, df_weather.AirTemp.min()-0.5, facecolor="blue", color='blue', alpha=0.1, zorder=0, label = 'Rain')
ax[0,0].legend(loc='upper right')
ax[0,0].set_ylabel('Temperature')
ax[0,0].title.set_text('Track Temperature & Air Temperature (°C)')
ax[0, 0].set_xlim(0, df_weather['SessionTime(Minutes)'].max())

# Humidity
sns.lineplot(df_weather, x='SessionTime(Minutes)', y='Humidity', ax=ax[0,1])
ax[0,1].set_ylabel('Humidity (%)')
if rain:
    ax[0,1].fill_between(df_weather[df_weather.Rainfall == True]['SessionTime(Minutes)'], df_weather.Humidity.max()+0.5, df_weather.Humidity.min()-0.5, facecolor="blue", color='blue', alpha=0.1, zorder=0, label = 'Rain')
    ax[0,1].legend(loc='upper right')
ax[0,1].title.set_text('Track Humidity (%)')
ax[0, 1].set_xlim(0, df_weather['SessionTime(Minutes)'].max())

# Pressure
# Define your constant reference
standard_pressure = 1013.25  # mbar
max_wind_speed = df_weather['WindSpeed'].max()  # example max wind speed
pressure_normalized = standard_pressure / 1013.25 * max_wind_speed  # simple normalization (adjust as necessary)

sns.lineplot(data = df_weather, x='SessionTime(Minutes)', y='Pressure', ax = ax[1,0])
plt.tight_layout()
# Add the reference line for 1013.25 mbar (standard atmospheric pressure)
ax[1,0].axhline(y=standard_pressure, color='red', linestyle='--', label='1013.25 mbar (Standard Pressure)')
ax[1,0].title.set_text('Air Pressure (mbar)')
ax[1,0].set_xlim(0, df_weather['SessionTime(Minutes)'].max())

# Wind Direction & Speed
rect = ax[1, 1].get_position()
wax = WindroseAxes(fig, rect)

fig.delaxes(ax[1, 1]) # Remove the previous axes at [1, 1] (the Cartesian axes)
fig.add_axes(wax) # Add the windrose axes in place of the original subplot

wax.set_title('Wind Direction (°) and Speed(m/s)\n\n', fontsize=12, color='white')
wax.bar(df_weather.WindDirection, df_weather.WindSpeed, normed=True, opening=0.8, edgecolor='white')
wax.set_legend()

plt.savefig(f"{img_folder}/06-Weather_Data.png")

print(f"Images saved in {img_folder}")