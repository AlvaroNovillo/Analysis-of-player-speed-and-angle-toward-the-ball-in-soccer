
# -*- coding: utf-8 -*-
"""
Speed study of the players

@author: alvaro
"""


from load_single_team_data import games_played,team_data
from load_data import load_single_game
import matplotlib.colors
import numpy as np
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics
import fnmatch 
import seaborn as sns
import re
from fig_config import (
    add_grid,
    figure_features,  
    )
"""
USER VARIABLES:
    Team_name -> Name of the team you want to study
"""


#USER VARIABLES
Team_names = ['Alavés','Athletic Club','Eibar','Espanyol','Getafe','Granada CF','Osasuna','Real Betis','Sevilla','Real Madrid','Barcelona','Valencia CF','Levante','Atlético de Madrid','Villarreal','Real Sociedad','Leganés','Real Valladolid','Mallorca','Celta de Vigo']
dir_path = 'C:/Users/alvaro/Documents/DATOS/tracking'

divisions = 50
max_speed = 37.5            
dt=1/(5*3600)                    

c_bins = np.linspace(0,max_speed,100)



parts = ['Part 1', 'Part 2']
positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
attributes = ['all', 'attacking', 'defending']

ball_vel = {part:np.array([]) for part in parts}

ball_distance = {part: [] for part in parts}

velocities = {position: {part: {attribute: {'[0, 3)': np.array([]), '[3, 10)': np.array([]), '[10, inf)': np.array([])} for attribute in attributes} for part in parts} for position in positions}
velocities_cont = {position: {part: {attribute: {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] } for attribute in attributes} for part in parts} for position in positions}
velocities_cont_player = {position: {part: {attribute: {} for attribute in attributes} for part in parts} for position in positions}
angle_cont = {position: {part: {attribute: {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1 )] } for attribute in attributes} for part in parts} for position in positions}
angle_cont_player = {position: {part: {attribute: {} for attribute in attributes} for part in parts} for position in positions}

 
velocities_player = {}
for position in positions:
    velocities_player[position] = {}
    for part in parts:
        velocities_player[position][part] = {}
        for attribute in attributes:
            velocities_player[position][part][attribute] = {}
 
           
import math
    
def angles_between_vectors(v1,v2):

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    
    def angle(v1, v2): 
        """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

    
    v1 = np.array(v1)
    v2 = np.array(v2)
    angles = []
    for i in range(v1.shape[1] - 1):
        vector_ball = [v2[0,i]-v1[0,i],v2[1, i] - v1[1, i]]
        vector_vel = [v1[0,i+1] - v1[0,i],v1[1,i+1] - v1[1,i]]


        ang = angle(vector_vel,vector_ball)
        
        angles.append(ang)
    
    # Append the last angle as the same as the penultimate angle
    angles.append(angles[-1])
    
    
    return angles



      
dir_path = 'C:/Users/alvaro/Documents/DATOS/tracking'

Position =  'All' #['Defender', 'Goalkeeper', 'Midfielder', 'Striker', 'Substitute','All']
 #Index of the data (0 -> 1 as in Matlab)


jump_gap = 12 #Desired FPS 
field_div = 11 #Number of divisions (per coordinate)

only_ball_alive = True



#Load data
game_list = []
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        if path.startswith('._'): continue
        if not path.endswith('rawmetadata.xml'): continue
        game_list.append(re.findall(r'\b\d+\b',path)[0])


frames = 0
for k,games in enumerate(game_list):
    

    
    Team = 'Both'
    desired_game = '*' + games + '*'
    pitch_info = str(games) + '-rawmetadata.xml'
    
    # list to store xml files 
    file_name = []
    desired_file_name = []
    
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file and it is not a csv file
        if os.path.isfile(os.path.join(dir_path, path)) and path.endswith('.csv') and not path.startswith('._'):
            file_name.append(path)
    
    # Find the desired match among all files and store it in desired_file_name
    desired_file_name = [file for file in file_name if fnmatch.fnmatchcase(file, desired_game)]
    


    
   
    players_home,players_away,ball,game_info,home_data,away_data = load_single_game(desired_game,dir_path,Team,Position,jump_gap,only_ball_alive, parts,5)
    
    (data_referee,_,_,ref_data) = load_single_game(desired_game,dir_path,'Referee',Position,jump_gap,only_ball_alive, parts,5)
    #Extract the main info about the match
    Team = 'Home'
    Team_name = game_info['Game'][game_info['Game'].Status == Team].Name.tolist()[0]

    

    rival_team = 'Away'
    rival_name = game_info['Game'][game_info['Game'].Status == rival_team].Name.tolist()[0]

    
    def dist_speed(dist, vel):
        """
        Function to divide the velocities into 3 intervals based on distances
        
        Parameters:
        dist (list or numpy array): a list or numpy array of distances
        vel (list or numpy array): a list or numpy array of velocities, corresponding to each distance
        
        Returns:
        intervals (dict): a dictionary with 3 key-value pairs, each key describes an interval and its corresponding value is the list of velocities in that interval
        
        """
        # convert inputs to numpy arrays
        vel = np.array(vel)
        dist = np.array(dist)
        
        # define the 3 intervals
        R = pd.arrays.IntervalArray.from_breaks([0.01, 3, 10, np.inf])
        
        # determine which interval each distance belongs to
        dist_idx = np.digitize(dist, R.right)
   
        # divide velocities into 3 intervals based on the distances
        intervals = {'[0, 3)': [], '[3, 10)': [], '[10, inf)': []}
        for i, idx in enumerate(dist_idx):
            intervals[list(intervals.keys())[idx]].append(vel[i])
        
        return intervals
    def dist_speed_cont(dist, vel):
        """
        Function to divide the velocities into 3 intervals based on distances
        
        Parameters:
        dist (list or numpy array): a list or numpy array of distances
        vel (list or numpy array): a list or numpy array of velocities, corresponding to each distance
        
        Returns:
        intervals (dict): a dictionary with 3 key-value pairs, each key describes an interval and its corresponding value is the list of velocities in that interval
        
        """
        # convert inputs to numpy arrays
        vel = np.array(vel)
        dist = np.array(dist)
        
        # define the 3 intervals
        R = pd.arrays.IntervalArray.from_breaks(np.linspace(0, divisions ,divisions + 1))
        
        # determine which interval each distance belongs to
        dist_idx = np.digitize(dist, R.right)
   
        # divide velocities into 3 intervals based on the distances
        labels = ["%g" % x for x in np.linspace(0, divisions ,divisions + 1)]
        
        intervals = {key:[] for key in labels}
        for i, idx in enumerate(dist_idx):
            intervals[list(intervals.keys())[idx]].append(vel[i])
        
        return intervals
    





            
    for part in parts:
        ball_vel[part] = np.concatenate((ball[part]['mod_speed'][ball[part]['mod_speed'] < 130],ball_vel[part]))
        poss = np.array(ball[part]['Ball Owning team'])
        frames += len(poss)
        
        for g,players in enumerate([players_home,players_away]):
            data = home_data if g == 0 else away_data
            for player in players[part].keys():
                if len(players[part][player]['mod_speed']) != len(ball[part]['mod_speed']): continue #Get rid of those players who did not played the whole game
                num = re.findall("\d+",player)[0]
                pos = data.Position.loc[data['ShirtNumber'] == str(num)].tolist()[0]
                
                #Speed and distances
                vel = players[part][player]['mod_speed']
                #Distance while attacking or defending
                
                dis_att = players[part][player]['Ball_distance_att']
                dis_def = players[part][player]['Ball_distance_def']
                #Total distance
                total_dis = dis_def.copy()
                total_dis[np.isnan(total_dis)] = dis_att
                
                #Remove na
                dis_att = dis_att[~np.isnan(dis_att)]
                dis_def = dis_def[~np.isnan(dis_def)]
                
                #Angles
                #Player pos 
                if g == 0: #Home
                    x = players[part][player]['Position X']
                    y = players[part][player]['Position Y']
                    
                    speed_x = players[part][player]['Speed_X']
                    speed_y = players[part][player]['Speed_Y']
                    
                    #Ball position
                    x_b = ball[part]['BPosition X']
                    y_b = ball[part]['BPosition Y']
                else:
                    x = -players[part][player]['Position X']
                    y = -players[part][player]['Position Y']
                    
                    speed_x = -players[part][player]['Speed_X']
                    speed_y = -players[part][player]['Speed_Y']
                    
                    x_b = -ball[part]['BPosition X']
                    y_b = ball[part]['BPosition Y']
                    
                
                pos_player = np.array([x,y])
                pos_ball = np.array([x_b ,y_b])
                
                
                
                angles = angles_between_vectors(pos_player,pos_ball)
                
                #attacking and defending speeds
                if g == 0:
                    idx_att = poss[vel.index] == 1
                    idx_def = poss[vel.index] == 2
                else:
                    idx_att = poss[vel.index] == 2
                    idx_def = poss[vel.index] == 1
                
                vel_att = vel[idx_att]
                vel_def = vel[idx_def]
                
                angles_att = np.array(angles)[idx_att]
                angles_def = np.array(angles)[idx_def]
                
                
            
                if pos == 'Substitute':
                    pos = data.SubPosition[data['ShirtNumber'] == str(num)].tolist()[0]
                if pos == 'Striker':
                    pos = 'Forward'
                #Discrete    
                vel_dist = dist_speed(total_dis, vel)
                vel_att_dist = dist_speed(dis_att, vel_att)
                vel_def_dist = dist_speed(dis_def, vel_def)
                #Cont
                vel_dist_cont = dist_speed_cont(total_dis, vel)
                vel_att_dist_cont = dist_speed_cont(dis_att, vel_att)
                vel_def_dist_cont = dist_speed_cont(dis_def, vel_def)
                
                #Angles ball - vel
                angles_cont = dist_speed_cont(total_dis, angles)
                angles_att_cont = dist_speed_cont(dis_att, angles_att)
                angles_def_cont = dist_speed_cont(dis_def, angles_def)
                
                
                
                
                velocities_dict = {'all':vel_dist,'attacking':vel_att_dist,'defending':vel_def_dist}
    
                variables = [velocities,velocities_player]
                player = re.sub(r'\(.*?\)', '', player) #Avoid confusions when players change the shirt number
                for attribute, vel_value in velocities_dict.items():
                    for variable in variables:
                        if np.array_equal(variable, variables[1]): #players dataset
                            if player not in variable[pos][part][attribute].keys():
                                variable[pos][part][attribute][player] = {'[0, 3)': np.array([]), '[3, 10)': np.array([]), '[10, inf)': np.array([])}
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][player][interval] = np.concatenate((variable[pos][part][attribute][player][interval], np.array(vel_interval)))
                        else:
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][interval] = np.concatenate((variable[pos][part][attribute][interval], np.array(vel_interval)))
                #Continuous case           
                velocities_dict_cont = {'all':vel_dist_cont,'attacking':vel_att_dist_cont,'defending':vel_def_dist_cont}
                variables_cont = [velocities_cont,velocities_cont_player]
                for attribute, vel_value in velocities_dict_cont.items():
                    for variable in variables_cont:
                        if np.array_equal(variable, variables_cont[1]): #players dataset
                            if player not in variable[pos][part][attribute].keys():
                                variable[pos][part][attribute][player] = {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] }
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][player][interval] = np.concatenate((variable[pos][part][attribute][player][interval], np.array(vel_interval)))
                        else:
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][interval] = np.concatenate((variable[pos][part][attribute][interval], np.array(vel_interval)))
        
                #Angles        
                velocities_dict_cont = {'all':angles_cont,'attacking':angles_att_cont,'defending':angles_def_cont}
                variables_cont = [angle_cont,angle_cont_player]
                for attribute, vel_value in velocities_dict_cont.items():
                    for variable in variables_cont:
                        if np.array_equal(variable, variables_cont[1]): #players dataset
                            if player not in variable[pos][part][attribute].keys():
                                variable[pos][part][attribute][player] = {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] }
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][player][interval] = np.concatenate((variable[pos][part][attribute][player][interval], np.array(vel_interval)))
                        else:
                            for interval,vel_interval in vel_value.items():
                                variable[pos][part][attribute][interval] = np.concatenate((variable[pos][part][attribute][interval], np.array(vel_interval)))
        

               
####################### FIGURES ##############################

def join(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    
    return joined_parts
def join_and_compute_average_speed(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed = {position: {attribute: {key: np.nanmean(joined_parts[position][attribute][key][joined_parts[position][attribute][key] < 27]) for key in joined_parts[position][attribute].keys()} for attribute in joined_parts[position].keys()} for position in joined_parts}
    return average_speed


def join_and_compute_average_speed_per_attribute(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed = {position: {attribute: np.nanmean(joined_parts[position][attribute][key][joined_parts[position][attribute][key] < 27], axis=0) for attribute in joined_parts[position].keys() for key in joined_parts[position][attribute].keys()} for position in joined_parts}
    return average_speed

def join_and_average(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed_per_position = {position: np.nanmean([np.nanmean(joined_parts[position]['all'][key][joined_parts[position]['all'][key] < 27], axis=0)  for key in joined_parts[position][attribute].keys()]) for position in joined_parts}
    return average_speed_per_position


def join_and_compute_average_speed_and_std(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed = {position: {attribute: {key: (np.nanmean(joined_parts[position][attribute][key][joined_parts[position][attribute][key] < 27]), np.nanstd(joined_parts[position][attribute][key][joined_parts[position][attribute][key] < 27])) for key in joined_parts[position][attribute].keys()} for attribute in joined_parts[position].keys()} for position in joined_parts}
    return average_speed


average_speed_cont = join_and_compute_average_speed(velocities_cont)
average_speed =  join_and_compute_average_speed(velocities)
average_speed_std =  join_and_compute_average_speed_and_std(velocities)

average_speed_attribute = join_and_compute_average_speed_per_attribute(velocities)
speed_mean = join_and_average(velocities)
join_speed = join(velocities)


######################### CONTINUOUS CASE ######################################
#Scatter 
figure_features(tex = False)
plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)','(C)','(D)']
for i,position in enumerate(positions):
    att_velocities = []
    def_velocities = []
    ax = plt.subplot(2, 2, i + 1)
      
    
    
    ax.text(0.85, 0.85, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    for interval in np.linspace(0, divisions ,divisions + 1):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_speed_cont[position]['attacking'][interval])
        def_velocities.append(average_speed_cont[position]['defending'][interval])
    sns.scatterplot(x = np.linspace(0, divisions ,divisions + 1), y = att_velocities,label = 'attacking',ax = ax)
    sns.scatterplot(x = np.linspace(0, divisions ,divisions + 1), y = def_velocities,label = 'defending',ax = ax)
    ax.axhline(y = speed_mean[position], color = 'black', linestyle = '--')
plt.legend()
    
#Barplot 
figure_features(tex = False)

plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.5)     
for p,position in enumerate(positions):
    ax = plt.subplot(2, 2, p + 1)
    
    my_df = pd.DataFrame(average_speed[position]['all'].items())
    sns.barplot(x=0, y=1, data=my_df,ax = ax)
    
    #Attacking and defending figures 
    #Scatter 
figure_features(tex = False)
plt.figure(figsize=(7, 5))
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)']
attributes = ['attacking', 'defending']
for i,attribute in enumerate(attributes):
    ax = plt.subplot(1, 2, i + 1)
    ax.text(0.20, 0.10, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    if i == 0:
        ax.set_ylabel('speed (km/h)')
        ax.set_ylim([2,13])
        ax.set_yticks([2,4,6,8,10,12])
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xlabel('distance (m)')
    if i == 1:
        ax.set_ylim([2,13.5])
        ax.set_yticks([2,4,6,8,10,12])
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xlabel('distance (m)')
    for position in positions:
        att_velocities = []
        att_time = []
        
       
        for interval in np.linspace(0, divisions ,divisions + 1):
            interval = "%g" % interval
            # Get the attacking and defending velocities for the current position
            att_velocities.append(average_speed_cont[position][attribute][interval])
            att_time.append(len(velocities_cont[position]['Part 1'][attribute][interval]) + len(velocities_cont[position]['Part 2'][attribute][interval]))
            
            
        # Calculate marker size based on the length of the represented array
        
        att_time = np.array(att_time) / np.sum(att_time) 

        
        
        att_sizes = np.array(att_time) * 2000
            
        sns.scatterplot(x = np.linspace(0, divisions ,divisions + 1), y = att_velocities,ax = ax, s=att_sizes, edgecolor='black')
        #ax.axhline(y = speed_mean[position], color = 'black', linestyle = '--')
fig = plt.gcf()
fig.savefig('scatter_ball_dist_att_def.png', format='png', dpi=1200)  

#all
plt.figure(figsize=(5, 5))
plt.subplots_adjust(hspace=0.2)
attributes = ['all']
for i, attribute in enumerate(attributes):
    ax = plt.subplot(1, 1, 1)
    
    ax.set_ylabel('speed (km/h)')
    ax.set_ylim([3,13.5])
    ax.set_xticks([0,10,20,30,40,50])

    ax.set_xlabel('distance (m)')
    for position in positions:
        velocities = []
        att_time = []
        for interval in np.linspace(0, divisions, divisions + 1):
            interval = "%g" % interval
            velocities.append(average_speed_cont[position][attribute][interval])
            att_time.append(len(velocities_cont[position]['Part 1'][attribute][interval]) + len(velocities_cont[position]['Part 2'][attribute][interval]))
            
            
        # Calculate marker size based on the length of the represented array
        
        att_time = np.array(att_time) / np.sum(att_time) 

        
        
        att_sizes = np.array(att_time) * 2000

            
        sns.scatterplot(x=np.linspace(0, divisions, divisions + 1), y=velocities, ax=ax,s = att_sizes, edgecolor='black')
fig = plt.gcf()
fig.savefig('scatter_all.png', format='png', dpi=1200)
    
########### Boxplot inset with scatter (and means for att and def) ##################


figure_features(tex = False)
plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.2,wspace =0 )
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
labels = ['(A)','(B)','(C)','(D)']
scatlims = [[0,7.5],[6,13],[7.5,13.5],[6,14]]
boxlims = [[3,6.5],[4.5,11.5],[4,11.5],[4.5,12.5]]
for i,position in enumerate(positions):
    att_velocities = []
    def_velocities = []
    att_time = []  # Array to store time spent at each distance
    def_time = []  # Array to store time spent at each distance
    ax = plt.subplot(2, 2, i + 1)
    if i != 0:
        ax.text(0.1, 0.10, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    else:
        ax.text(0.9, 0.90, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    for interval in np.linspace(0, interval,interval + 20):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_speed_cont[position]['attacking'][interval])
        def_velocities.append(average_speed_cont[position]['defending'][interval])
        # Get the time spent at each distance for the current position
        att_time.append(len(velocities_cont[position]['Part 1']['attacking'][interval]) + len(velocities_cont[position]['Part 2']['attacking'][interval]))
        def_time.append(len(velocities_cont[position]['Part 1']['defending'][interval]) + len(velocities_cont[position]['Part 2']['defending'][interval]))
        
    # Calculate marker size based on the length of the represented array
    
    att_time = np.array(att_time) / np.sum(att_time) 
    def_time = np.array(def_time) / np.sum(def_time) 
    
    
    att_sizes = np.array(att_time) * 2000
    def_sizes = np.array(def_time) * 2000
    
    #sns.scatterplot(x=np.linspace(0, 50, 51), y=att_velocities, ax=ax, color='blue', edgecolor='black')
    #sns.scatterplot(x=np.linspace(0, 50, 51), y=def_velocities, ax=ax, color='orange', edgecolor='black')
    #With sizes proportional to the time
    sns.scatterplot(x=np.linspace(0, 50, 51), y=att_velocities, ax=ax, color='blue', s=att_sizes, edgecolor='black')
    sns.scatterplot(x=np.linspace(0, 50, 51), y=def_velocities, ax=ax, color='orange', s=def_sizes, edgecolor='black')
    ax.set_xticks(np.linspace(0, 50,5))
    ax.axhline(y = average_speed_attribute[position]['attacking'], color = 'b', linestyle = '--')
    ax.axhline(y = average_speed_attribute[position]['defending'], color = 'orange', linestyle = '--')
    ax.set_ylim(scatlims[i][0],scatlims[i][1])
    

    #plt.legend()
    if i == 0:
        ax.set_ylabel('speed (km/h)', fontsize=16)
    
        
    elif i == 2:
        ax.set_xlabel('distance (m)', fontsize=16)
        ax.set_ylabel('speed (km/h)', fontsize=16)
    
     
        
    elif i == 3:
        ax.set_xlabel('distance (m)', fontsize=16)
    
    
    else:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # create the inset AxesSubplot 
    if i == 0:
        inset_ax = plt.axes([box.x0 + box.width * 0.12, box.y0 + box.height * 0.1, box.width * 0.35, box.height * 0.35])
    else:
        inset_ax = plt.axes([box.x0 + box.width * 0.42, box.y0 + box.height * 0.62, box.width * 0.35, box.height * 0.35])
        
    plot_data = average_speed_std[position]['all']
    all_data = []
    all_keys = []
    for key, value in plot_data.items():
        all_data.append(value)
        all_keys.append(key)
    sns.boxplot(data=all_data, width=.18, ax = inset_ax)
    #sns.swarmplot(data=all_data, size=6, edgecolor="black", linewidth=.9, ax = inset_ax)
    plt.xticks(plt.xticks()[0], all_keys,fontsize = 11)
    plt.yticks(np.linspace(boxlims[i][0],boxlims[i][1],3),fontsize = 11)
    plt.ylim(boxlims[i][0],boxlims[i][1])
fig = plt.gcf()
fig.savefig('Scatter_std_boxplot_twomeans.png', format='png', dpi=1200)    
plt.show()

########### errorbar inset with scatter ##################
plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)','(C)','(D)']
scatlims = [[0,6],[7,11],[7,11],[7,11]]
boxlims = [[0,9],[2,17],[2,17],[2,17]]
average_speed_std = join_and_compute_average_speed_and_std(velocities)
for i,position in enumerate(positions):
    att_velocities = []
    def_velocities = []
    ax = plt.subplot(2, 2, i + 1)
    
    ax.text(0.85, 0.90, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    for interval in np.linspace(0,  divisions,divisions + 20):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_speed_cont[position]['attacking'][interval])
        def_velocities.append(average_speed_cont[position]['defending'][interval])
    sns.scatterplot(x = np.linspace(0, 50,51), y = att_velocities,ax = ax)
    sns.scatterplot(x = np.linspace(0, 50,51), y = def_velocities,ax = ax)
    ax.axhline(y = speed_mean[position], color = 'black', linestyle = '--')
    ax.set_ylim(scatlims[i][0],scatlims[i][1])
    ax.set_xticks(np.linspace(0, 50,5))
    #plt.legend()
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # create the inset AxesSubplot
    if i == 0 or i == 2:
        inset_ax = plt.axes([box.x0 + box.width * 0.1, box.y0 + box.height * 0.1, box.width * 0.3, box.height * 0.3])
    else:
        inset_ax = plt.axes([box.x0 + box.width * 0.4, box.y0 + box.height * 0.5, box.width * 0.3, box.height * 0.3])
        
    plot_data = average_speed_std[position]['all']
    all_keys = []
    mean = []
    std = []
    for key, value in plot_data.items():
        all_keys.append(key)
        mean.append(value[0])
        std.append(value[1])
    plt.errorbar(x=all_keys, y=mean, yerr=std, fmt='o', color='black', capsize=5, elinewidth=2)
    plt.xticks(plt.xticks()[0], all_keys,fontsize = 9)
    plt.yticks(np.linspace(boxlims[i][0],boxlims[i][1],3),fontsize = 9)
    plt.ylim(boxlims[i][0],boxlims[i][1])
fig = plt.gcf()
fig.savefig('Scatter_std_inset.pdf', format='pdf', dpi=1200)    
plt.show()

################## Time spent at each interval ##########################


figure_features(tex=False)
labels = ['(A)', '(B)']
for i,attribute in enumerate(attributes[1:]):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.5)
    #fig.suptitle(attribute, fontsize=18, y=0.96)

    ax.text(0.1, 0.90, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)

    for position in positions:
        time = []
        for interval in np.linspace(0, 50, 51):
            interval = "%g" % interval
            time.append(
                len(velocities_cont[position]['Part 1'][attribute][interval])
                + len(velocities_cont[position]['Part 2'][attribute][interval])
            )

        # Normalize time values to create a PDF
        normalized_time = np.array(time) / np.sum(time) 

        # Calculate the width of each bar
        bar_width = 50 / len(time)

        # Plot the bar plot
        plt.bar(np.linspace(0, 50, 51), normalized_time, width=bar_width, label=position, alpha = 0.5)

    #ax.text(0.85, 0.50, '(A)', horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, fontsize=25)
    ax.set_ylabel('p.d.f.')
    ax.set_xlabel('distance (m)')
    ax.set_ylim([0,0.05])
    if i == 0:
        ax.legend()
    fig.savefig(f'General_speed_distributions_{attribute}.png', format='png', dpi=1200)


        



################## Speed in continuous case (per position, all) ########################
figure_features(tex = False)
plt.figure(figsize=(12, 10))
scatlims = [[3,6],[7,10],[7,10],[7,10]]
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)','(C)','(D)']
for i,position in enumerate(positions):
    print(position)
    att_velocities = []
    def_velocities = []
    ax = plt.subplot(2, 2, i + 1)
    
    ax.text(0.10, 0.10, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    for interval in np.linspace(0, divisions ,divisions + 20):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_speed_cont[position]['all'][interval])
    sns.scatterplot(x = np.linspace(0,  divisions,divisions + 20), y = att_velocities,ax = ax)
    ax.axhline(y = speed_mean[position], color = 'k', linestyle = '--')
    if i == 0:
        ax.set_ylabel('speed (km/h)')
    
        
    elif i == 2:
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('speed (km/h)')
    
     
        
    elif i == 3:
        ax.set_xlabel('distance (m)')
    
    
    else:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
    #ax.set_ylim(scatlims[i][0],scatlims[i][1])


############################## Angles #######################################

def join_and_compute_average_speed(velocities_cont):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed = {position: {attribute: {key: np.nanmean(joined_parts[position][attribute][key]) for key in joined_parts[position][attribute].keys()} for attribute in joined_parts[position].keys()} for position in joined_parts}
    return average_speed
def join_and_average(velocities_cont,attribute):
    joined_parts = {position: {attribute: {key: np.concatenate((velocities_cont[position]['Part 1'][attribute][key], velocities_cont[position]['Part 2'][attribute][key]), axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()} for attribute in velocities_cont[position]['Part 1']} for position in velocities_cont}
    average_speed_per_position = {position: np.nanmean([np.nanmean(joined_parts[position][attribute][key], axis=0)  for key in joined_parts[position][attribute].keys()]) for position in joined_parts}
    return average_speed_per_position




average_angle_cont = join_and_compute_average_speed(angle_cont)
average_angle_att = join_and_average(angle_cont,'attacking')
average_angle_def = join_and_average(angle_cont,'defending')
average_angle_all = join_and_average(angle_cont,'all')

#Scatter 
figure_features(tex = False)
plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)','(C)','(D)']
for i,position in enumerate(positions):
    att_velocities = []
    def_velocities = []
    att_time = []
    def_time = []
    ax = plt.subplot(2, 2, i + 1)
    
    ax.text(0.1, 0.85, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    for interval in np.linspace(0, divisions ,divisions + 1):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_angle_cont[position]['attacking'][interval])
        def_velocities.append(average_angle_cont[position]['defending'][interval])
        # Get the time spent at each distance for the current position
        att_time.append(len(velocities_cont[position]['Part 1']['attacking'][interval]) + len(velocities_cont[position]['Part 2']['attacking'][interval]))
        def_time.append(len(velocities_cont[position]['Part 1']['defending'][interval]) + len(velocities_cont[position]['Part 2']['defending'][interval]))
        
    # Calculate marker size based on the length of the represented array
    
    att_time = np.array(att_time) / np.sum(att_time) 
    def_time = np.array(def_time) / np.sum(def_time) 
    
    
    att_sizes = np.array(att_time) * 2000
    def_sizes = np.array(def_time) * 2000
    sns.scatterplot(x=np.linspace(0, 50, 51), y=att_velocities, ax=ax, color='blue', s=att_sizes, edgecolor='black')
    sns.scatterplot(x=np.linspace(0, 50, 51), y=def_velocities, ax=ax, color='orange', s=def_sizes, edgecolor='black')
    ax.axhline(y = average_angle_att[position], color = 'b', linestyle = '--')
    ax.set_ylim([40,120])
    ax.axhline(y = average_angle_def[position], color = 'orange', linestyle = '--')
    
    if i == 0:
        ax.set_ylabel('angle (º)')
    
        
    elif i == 2:
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('angle (º)')
    
     
        
    elif i == 3:
        ax.set_xlabel('distance (m)')
    
    
    else:
        ax.set_xlabel(None)
        ax.set_ylabel(None)

fig = plt.gcf()
fig.savefig('angles_att_def.png', format='png', dpi=1200)    

#Scatter 
figure_features(tex = False)
plt.figure(figsize=(12, 10))
scatlims = [[70,90],[50,80],[50,80],[50,80]]
plt.subplots_adjust(hspace=0.2)

for i,position in enumerate(positions):
    print(position)
    att_velocities = []
    def_velocities = []
    ax = plt.subplot(2, 2, 1)
    
    
    for interval in np.linspace(0, divisions ,divisions + 1):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        att_velocities.append(average_angle_cont[position]['all'][interval])
        
    
    
    
    ax.set_ylabel('angle (º)')
    ax.set_xlabel('distance (m)')
    
figure_features(tex = False)
plt.figure(figsize=(7, 5))
plt.subplots_adjust(hspace=0.2)
labels = ['(A)','(B)']
attributes = ['attacking', 'defending']
for i,attribute in enumerate(attributes):
    ax = plt.subplot(1, 2, i + 1)
    ax.text(0.20, 0.10, labels[i], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    if i == 0:
        ax.set_ylabel('angle (º)')
        ax.set_ylim([40,120])
        #ax.set_yticks([2,4,6,8,10,12])
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xlabel('distance (m)')
    if i == 1:
        ax.set_ylim([40,120])
        #ax.set_yticks([2,4,6,8,10,12])
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xlabel('distance (m)')
    for position in positions:
        att_velocities = []
        att_time = []
       
        for interval in np.linspace(0, divisions ,divisions + 1):
            interval = "%g" % interval
            # Get the attacking and defending velocities for the current position
            att_velocities.append(average_angle_cont[position][attribute][interval])
            att_time.append(len(velocities_cont[position]['Part 1'][attribute][interval]) + len(velocities_cont[position]['Part 2'][attribute][interval]))
            
            
        # Calculate marker size based on the length of the represented array
        
        att_time = np.array(att_time) / np.sum(att_time) 

        
        
        att_sizes = np.array(att_time) * 2000

            
        sns.scatterplot(x=np.linspace(0, divisions, divisions + 1), y=att_velocities, ax=ax, s=att_sizes, edgecolor='black')
        
        #ax.axhline(y = speed_mean[position], color = 'black', linestyle = '--')
fig = plt.gcf()
fig.savefig('scatter_angle_att_def.png', format='png', dpi=1200)  
    
        
############################## Angle - Velocity scatterplot ##############################################


from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

figure_features(tex=False)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
fig.subplots_adjust(hspace=0.2)
labels = ['(A)', '(B)', '(C)', '(D)']
vellims = [[3, 6], [7, 10], [7, 10], [7, 10]]

# Calculate the overall minimum and maximum velocity values
min_velocity = min([min(average_speed_cont[position]['all'].values()) for position in positions])
max_velocity = max([max(average_speed_cont[position]['all'].values()) for position in positions])
colors = sns.color_palette("viridis", 51)

for i, position in enumerate(positions):
    ax = axs[i // 2, i % 2]

    ax.text(0.1, 0.10, labels[i], horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=25)
    angle_df = pd.DataFrame.from_dict(average_angle_cont[position]['all'], orient='index', columns=['angle'])
    velocities_df = pd.DataFrame.from_dict(average_speed_cont[position]['all'], orient='index', columns=['velocities'])
    merged_df = pd.merge(angle_df, velocities_df, left_index=True, right_index=True)
    dict_keys = sorted(average_angle_cont[position]['all'].keys())
    
    norm = mpl.colors.Normalize(vmin=min_velocity, vmax=max_velocity)
    scatter = sns.scatterplot(data=merged_df, x=np.linspace(0, divisions, divisions + 1), y='angle',
                              c=merged_df.velocities, size=merged_df.velocities / max_velocity * 10,
                              palette="viridis", legend=False, ax=ax,vmin=min_velocity, vmax=max_velocity)
    ax.axhline(y=average_angle_all[position], color='k', linestyle='--')
    ax.set_ylim([60, 105])
    if i == 0:
        ax.set_ylabel('angle (º)')
        ax.set_xlabel(None)
    elif i == 2:
        ax.set_xlabel('distance (m)')
        ax.set_ylabel('angle (º)')
    elif i == 3:
        ax.set_xlabel('distance (m)')
        ax.set_ylabel(None)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = mpl.colors.Normalize(vmin=min_velocity, vmax=max_velocity)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
        cbar.ax.set_ylabel('speed (km/h)', rotation=90, labelpad=20)

    else:
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        norm = mpl.colors.Normalize(vmin=min_velocity, vmax=max_velocity)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cax)
        cbar.ax.set_ylabel('speed (km/h)', rotation=90, labelpad=20)
fig = plt.gcf()
fig.savefig('angles_speed.png', format='png', dpi=1200)
 
# Add colorbar





################################## Specific player analysis ##################################################

def vel_specific(position, player, attribute):
    
    figure_features(tex = False)
    plt.figure(figsize=(12, 5))
    
    # Define the scatter plot limits
    scatlims_dict = {
        'Goalkeeper': [3, 7],
        'Defender': [3.5, 13.5],
        'Midfielder': [3.5, 13.5],
        'Forward': [3.5, 14]
    }
    scatlims = scatlims_dict[position]
    labels = ['(A)','(B)']
    plt.subplots_adjust(hspace=0.2)
    reference = []
    wanted_player = []
    ax = plt.subplot(1, 2, 1)
    # check if player is present in both "Part 1" and "Part 2" of velocities_cont_player[position]
    if player in velocities_cont_player[position]['Part 1'][attribute] and player in velocities_cont_player[position]['Part 2'][attribute]:
        # concatenate velocity data for player across both parts
        joined_parts = {key: np.concatenate((velocities_cont_player[position]['Part 1'][attribute][player][key], velocities_cont_player[position]['Part 2'][attribute][player][key]), axis=0) for key in velocities_cont_player[position]['Part 1'][attribute][player].keys()}
    else:
        # take only the part where player is present
        if player in velocities_cont_player[position]['Part 1'][attribute]:
            joined_parts = velocities_cont_player[position]['Part 1'][attribute][player]
        elif player in velocities_cont_player[position]['Part 2'][attribute]:
            joined_parts = velocities_cont_player[position]['Part 2'][attribute][player]
        else:
            # handle case where player is not present in either part
            print('No data is avaliable for the player')
            return None
    att_time = []
    player_time = []
    for interval in np.linspace(0, divisions ,divisions + 1):
        interval = "%g" % interval
        # Get the attacking and defending velocities for the current position
        reference.append(average_speed_cont[position][attribute][interval])
        wanted_player.append(np.mean(joined_parts[interval][joined_parts[interval] < 27]))
        att_time.append(len(velocities_cont[position]['Part 1'][attribute][interval]) + len(velocities_cont[position]['Part 2'][attribute][interval]))
        player_time.append(len(joined_parts[interval][joined_parts[interval] < 27]))
        
    # Calculate marker size based on the length of the represented array
    
    att_time = np.array(att_time) / np.sum(att_time) 
    player_time = np.array(player_time) / np.sum(player_time)

    
    
    att_sizes = np.array(att_time) * 2500
    player_sizes = np.array(player_time) * 2500

        
    bar_width = 1.2
        
    sns.scatterplot(x = np.linspace(0, divisions ,divisions + 1), y = reference,ax = ax, alpha = 0.5, s=att_sizes, edgecolor='black',label = 'league average')
    sns.scatterplot(x = np.linspace(0, divisions ,divisions + 1), y = wanted_player,ax = ax,s=player_sizes, edgecolor='black',label = player)
    ax.set_ylim(scatlims)
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('speed (km/h)')
    ax.axhline(y = speed_mean[position], color = 'k', linestyle = '--')
    ax.text(0.10, 0.10, labels[0], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    ax = plt.subplot(1, 2, 2)
    ax.bar(np.linspace(0, 50, 51), att_time, width = bar_width , alpha = 0.5,label = 'league average' )
    ax.bar(np.linspace(0, 50, 51), player_time, width = bar_width , alpha = 0.5,label = player)
    ax.set_ylabel('p.d.f.')
    ax.set_xlabel('distance (m)')
    ax.set_ylim([0,0.05])
    ax.text(0.90, 0.90, labels[1], horizontalalignment='center', verticalalignment='center',transform = ax.transAxes,fontsize=25)
    ax.legend(loc = 'upper left')
    
    
    plt.tight_layout()  # Adjust spacing between subplots
    
    

    
#Examples proposed to be compared

vel_specific('Goalkeeper','Jan Oblak ','all')  
vel_specific('Defender','Daniel Carvajal ','all')  
vel_specific('Midfielder','Martin Ødegaard ','all')  
vel_specific('Forward','Lionel Messi ','all') 

pos = 'Midfielder'
for player in velocities_cont_player[pos]['Part 1']['all'].keys():
    vel_specific(pos,player,'all') 
    
def angle_specific(position, player, attribute):
    figure_features(tex=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)  # Use subplots to create two side-by-side plots

    reference = []
    wanted_player = []
    att_time = []
    player_time = []
    labels = ['(A)', '(B)']

    # Plotting for the first subplot
    ax = axs[0]
    joined_parts = {key: np.concatenate((angle_cont_player[position]['Part 1'][attribute][player][key],
                                         angle_cont_player[position]['Part 2'][attribute][player][key]),
                                        axis=0) for key in velocities_cont[position]['Part 1'][attribute].keys()}

    for interval in np.linspace(0, divisions, divisions + 1):
        interval = "%g" % interval
        reference.append(average_angle_cont[position][attribute][interval])
        wanted_player.append(np.nanmean(joined_parts[interval]))
        att_time.append(len(velocities_cont[position]['Part 1'][attribute][interval]) +
                        len(velocities_cont[position]['Part 2'][attribute][interval]))
        player_time.append(len(joined_parts[interval][joined_parts[interval] < 27]))

    att_time = np.array(att_time) / np.sum(att_time)
    player_time = np.array(player_time) / np.sum(player_time)

    att_sizes = np.array(att_time) * 2500
    player_sizes = np.array(player_time) * 2500

    sns.scatterplot(x=np.linspace(0, divisions, divisions + 1), y=reference, ax=ax, s=att_sizes, edgecolor='black',
                    alpha=0.5, label='league average')
    sns.scatterplot(x=np.linspace(0, divisions, divisions + 1), y=wanted_player, ax=ax, s=player_sizes,
                    edgecolor='black', label=player)
    ax.axhline(y=average_angle_all[position], color='k', linestyle='--')
    ax.set_xlabel('distance (m)')
    ax.set_ylabel('angle (º)')
    ax.set_ylim([40, 110])
    ax.text(0.10, 0.10, labels[0], horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=25)

    # Plotting for the second subplot
    ax = axs[1]
    ax.set_ylim([0, 0.05])
    ax.bar(np.linspace(0, 50, 51), att_time, width=bar_width, alpha=0.5, label='league average')
    ax.bar(np.linspace(0, 50, 51), player_time, width=bar_width, alpha=0.5, label=player)
    ax.set_ylabel('p.d.f.')
    ax.set_xlabel('distance (m)')
    
    ax.text(0.90, 0.90, labels[1], horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=25)
    ax.legend(loc='upper left')

    plt.tight_layout()  # Adjust spacing between subplots

    plt.show()

#Examples proposed to be compared
angle_specific('Goalkeeper','Jan Oblak ','all')  
angle_specific('Defender','Daniel Carvajal ','all')  
angle_specific('Midfielder','Martin Ødegaard ','all')  
angle_specific('Forward','Lionel Messi ','all')  
