# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:42:11 2023

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
positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward','All']
attributes = ['all', 'attacking', 'defending']
y_divisions = ['CED','CID','CC','CII','CEI']

ball_vel = {part:np.array([]) for part in parts}

ball_distance = {part: [] for part in parts}

velocities = {position: {part: {attribute: pd.DataFrame([{"[0, 3)":[], "[3, 10)":[], "[10, inf)":[]} for y in range(4)] for x in range(5))
 for attribute in attributes} for part in parts} for position in positions}
"""
velocities_cont = {position: {part: {attribute: {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] } for attribute in attributes} for part in parts} for position in positions}
velocities_cont_player = {position: {part: {attribute: {} for attribute in attributes} for part in parts} for position in positions}
angle_cont = {position: {part: {attribute: {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] } for attribute in attributes} for part in parts} for position in positions}
angle_cont_horiz = {position: {part: {attribute: {key:np.array([]) for key in ["%g" % x for x in np.linspace(0, divisions,divisions + 1)] } for attribute in attributes} for part in parts} for position in positions}
"""

 
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
def angles_to_horizontal(v1):
    def slope(x1, y1, x2, y2): # Line slope given two points:
        return (y2-y1)/(x2-x1)

    def angle(s1, s2): 
        return math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
    
    v1 = np.array(v1)
    angles = []
    for i in range(v1.shape[1] - 1):

        lineB = ((v1[0,i],v1[0,i+1]), (v1[1,i], v1[1,i+1]))
        
        slope1 = 0
        slope2 = slope(lineB[0][0], lineB[0][1], lineB[1][0], lineB[1][1])

        ang = angle(slope1, slope2)
        
        
        angles.append(abs(ang))
    # Append the last angle as the same as the penultimate angle
    angles.append(angles[-1])
    return angles


def bimodal_split(vel, dt=1/(5*3600)):
    # Find the mode of the data
    mode_val = np.median(vel)
    peak1_vel = [v for v in vel if v <= mode_val]
    peak2_vel = [v for v in vel if v > mode_val]
    
    # Compute the distance vector
    distances = vel * dt
    
    peak1_distances = distances[np.isin(vel, peak1_vel)]
    peak2_distances = distances[np.isin(vel, peak2_vel)]
    return peak1_distances, peak2_distances


def bimodal_split_mean(vel, dt=1/(5*3600)):
    # Find the mode of the data
    mode_val = np.median(vel)
    peak1_vel = [v for v in vel if v <= mode_val]
    peak2_vel = [v for v in vel if v > mode_val]
    
    # Compute the distance vector
    peak1_vel_mean = np.median(peak1_vel)
    peak2_vel_mean = np.median(peak2_vel)

    return peak1_vel_mean, peak2_vel_mean

def bimodal_split_dist(vel):
    # Find the mode of the data
    mode_val = np.median(vel)
    peak1_vel = [v for v in vel if v <= mode_val]
    peak2_vel = [v for v in vel if v > mode_val]

    return peak1_vel,peak2_vel           


missing_games = []


dir_path = 'C:/Users/alvaro/Documents/DATOS/tracking'

Position =  'All' #['Defender', 'Goalkeeper', 'Midfielder', 'Striker', 'Substitute','All']
 #Index of the data (0 -> 1 as in Matlab)
parts = ['Part 1', 'Part 2'] #  ['Part 1'], ['Part 2']

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
    
    xtree = et.parse(dir_path + '/' + pitch_info)
    root = xtree.getroot()
    
    b = []
    
    
    for person in root.iter('match'):
        b.append(person.attrib)
        
        
    pitchLengthX = b[0]['fPitchXSizeMeters']
    pitchLengthX = float(pitchLengthX.replace(',', '.'))
    pitchWidthY = b[0]['fPitchYSizeMeters']
    pitchWidthY = float(pitchWidthY.replace(',', '.'))
    
    
    if len(desired_file_name) != 2:
        missing_games.append(games)

        
        #Skip game if it is not complete
        continue

    
   
    players_home,players_away,ball,game_info,home_data,away_data = load_single_game(desired_game,dir_path,Team,Position,jump_gap,only_ball_alive, parts,5)
    
    (data_referee,_,_,ref_data) = load_single_game(desired_game,dir_path,'Referee',Position,jump_gap,only_ball_alive, parts,5)
    #Extract the main info about the match
    Team = 'Home'
    Team_name = game_info['Game'][game_info['Game'].Status == Team].Name.tolist()[0]

    

    rival_team = 'Away'
    rival_name = game_info['Game'][game_info['Game'].Status == rival_team].Name.tolist()[0]

    
    def dist_speed(dist, vel, pos_player):
        """
        Function to divide the velocities into 3 intervals based on distances and divide them into 5 regions based on x position
        Parameters:
        dist (list or numpy array): a list or numpy array of distances
        vel (list or numpy array): a list or numpy array of velocities, corresponding to each distance
        pos_player (numpy array): a 2D numpy array of player positions, where pos_player[0,:] contains x positions and pos_player[1,:] contains y positions for each distance
        Returns:
        intervals (dict): a dictionary with 3 key-value pairs, each key describes an interval and its corresponding value is a dictionary with 5 key-value pairs, each key describes a region and its corresponding value is the list of velocities in that region and interval
        """
        dist = np.array(dist)
        vel = np.array(vel)
        x = pos_player[0,:]
        y = pos_player[1,:]
        
        R = pd.arrays.IntervalArray.from_breaks([0, 3, 10, np.inf])
        X = pd.arrays.IntervalArray.from_breaks(np.linspace(-pitchLengthX/2,pitchLengthX/2,5))
        Y = pd.arrays.IntervalArray.from_breaks(np.linspace(-pitchWidthY/2,pitchWidthY/2,6))[::-1]
        
        # determine which interval each distance belongs to
        dist_idx = np.digitize(dist, R.right)
   
        
        # determine which region each x position belongs to
        X_idx = np.digitize(x, X.right) 
        if 4 in X_idx: #Do not let the player be outside the field
            X_idx[X_idx==4] = X_idx[X_idx==4] - 1

  
        Y_idx = np.digitize(y, Y.left)  
        if 5 in Y_idx:#Do not let the player be outside the field
            Y_idx[Y_idx==5] = Y_idx[Y_idx==5] - 1

            
        intervals = pd.DataFrame([{'[0, 3)':[],'[3, 10)': [],'[10, inf)': []} for x in range(4)] for y in range(5))
        for i,d in enumerate(dist_idx):
            key3 = list(intervals[X_idx[i]][Y_idx[i]].keys())[d]
            intervals[X_idx[i]][Y_idx[i]][key3].append(vel[i])
    
        return intervals






    def dist_speed_cont(dist, vel, y):
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
        dist_idx = []
        for x in dist:
            dist_idx.append([i for i, val in enumerate(R.contains(x)) if val])
        dist_idx = [item for sublist in dist_idx for item in sublist]
   
        # divide velocities into 3 intervals based on the distances
        labels = ["%g" % x for x in np.linspace(0, divisions ,divisions + 1)]
        
        intervals = {key:[] for key in labels}
        for i, idx in enumerate(dist_idx):
            intervals[list(intervals.keys())[idx]].append(vel[i])
        
        return intervals
    





            
    for part in parts:
        ball_vel[part] = np.concatenate((ball[part]['mod_speed'][ball[part]['mod_speed'] < 130],ball_vel[part]))
        poss = np.array(ball[part]['Ball Owning team'])
        for g,players in enumerate([players_home,players_away]): #Only home team in order to understand data
            data = home_data if g == 0 else away_data
            for player in players[part].keys():
                if len(players[part][player]['mod_speed']) != len(ball[part]['mod_speed']): continue #Get rid of those players who did not played the whole game
                num = re.findall("\d+",player)[0]
                pos = data.Position.loc[data['ShirtNumber'] == str(num)].tolist()[0]
                if pos == 'Substitute':
                    pos = data.SubPosition[data['ShirtNumber'] == str(num)].tolist()[0]
                if pos == 'Striker':
                    pos = 'Forward'
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
                
                
                #Player speed
                
                
                #v_vect = np.transpose(np.transpose(np.array([pd.DataFrame(players[part][player]['Position X']).diff().values,pd.DataFrame(players[part][player]['Position Y']).diff().values]))[0])
                v_vect = np.array([speed_x,speed_y])
                
                
                
                """
                angles = angles_between_vectors(pos_player,pos_ball)
                angles_horiz = angles_to_horizontal(pos_player)
                """
                #attacking and defending speeds
                if g == 0:
                    idx_att = poss[vel.index] == 1
                    idx_def = poss[vel.index] == 2
                else:
                    idx_att = poss[vel.index] == 2
                    idx_def = poss[vel.index] == 1
                
                vel_att = vel[idx_att]
                vel_def = vel[idx_def]
                
                pos_player_att = pos_player[:,idx_att]
                pos_player_def = pos_player[:,idx_def]
                """
                angles_att = np.array(angles)[idx_att]
                angles_def = np.array(angles)[idx_def]
                
                angles_horiz_att = np.array(angles_horiz)[idx_att]
                angles_horiz_def = np.array(angles_horiz)[idx_def]
                """
            
                
                #Discrete    
                vel_dist = dist_speed(total_dis, vel,pos_player)
                vel_att_dist = dist_speed(dis_att, vel_att,pos_player_att)
                vel_def_dist = dist_speed(dis_def, vel_def,pos_player_def)
                """
                #Cont
                vel_dist_cont = dist_speed_cont(total_dis, vel)
                vel_att_dist_cont = dist_speed_cont(dis_att, vel_att)
                vel_def_dist_cont = dist_speed_cont(dis_def, vel_def)
                
                #Angles ball - vel
                angles_cont = dist_speed_cont(total_dis, angles)
                angles_att_cont = dist_speed_cont(dis_att, angles_att)
                angles_def_cont = dist_speed_cont(dis_def, angles_def)
                
                
                #Angles vel - horiz
                angles_horiz_cont = dist_speed_cont(total_dis, angles_horiz)
                angles_horiz_att_cont = dist_speed_cont(dis_att, angles_horiz_att)
                angles_horiz_def_cont = dist_speed_cont(dis_def, angles_horiz_def)
                """
                velocities_dict = {'all':vel_dist,'attacking':vel_att_dist,'defending':vel_def_dist}
    
                variables = [velocities,velocities_player]
                player = re.sub(r'\(.*?\)', '', player) #Avoid confusions when players change the shirt number
                for attribute, vel_value in velocities_dict.items():
                    for variable in variables:
                        if np.array_equal(variable, variables[1]): #players dataset
                            if player not in variable[pos][part][attribute].keys():
                                variable[pos][part][attribute][player] = pd.DataFrame([{'[0, 3)':np.array([]),'[3, 10)': np.array([]),'[10, inf)': np.array([])} for y in range(4)] for x in range(5))
                                
                            # Concatenate the elements of the arrays inside the keys '[0, 3)', '[3, 10)', and '[10, inf)'
                            for y in range(5):
                                for x in range(4):
                                    for col in ['[0, 3)', '[3, 10)', '[10, inf)']:
                                        variable[pos][part][attribute][player].iloc[y,x][col] =  pd.concat([pd.DataFrame(variable[pos][part][attribute][player].iloc[y,x][col]).reset_index(drop=True), pd.DataFrame(vel_value.iloc[y,x][col]).reset_index(drop=True)], axis=0)
                        else:
                            # Concatenate the elements of the arrays inside the keys '[0, 3)', '[3, 10)', and '[10, inf)'
                            for y in range(5):
                                for x in range(4):
                                    for col in ['[0, 3)', '[3, 10)', '[10, inf)']:
                                        variable[pos][part][attribute].iloc[y,x][col] =  pd.concat([pd.DataFrame(variable[pos][part][attribute].iloc[y,x][col]).reset_index(drop=True), pd.DataFrame(vel_value.iloc[y,x][col]).reset_index(drop=True)], axis=0)
                                        variable['All'][part][attribute].iloc[y,x][col] =  pd.concat([pd.DataFrame(variable[pos][part][attribute].iloc[y,x][col]).reset_index(drop=True), pd.DataFrame(vel_value.iloc[y,x][col]).reset_index(drop=True)], axis=0)
    
    
                
               
####################### FIGURES ##############################


def compute_means(velocities):
    positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward','All']
    
    attributes = ['all', 'attacking', 'defending']
    
    # Initialize the resulting dataframes
    mean_dfs = {position:  {attribute:{
        "[0, 3)": pd.DataFrame([[None for y in range(4)] for x in range(5)]),
        "[3, 10)": pd.DataFrame([[None for y in range(4)] for x in range(5)]),
        "[10, inf)": pd.DataFrame([[None for y in range(4)] for x in range(5)])
    } for attribute in attributes}  for position in positions}
    
    pos_dfs = {position:  {attribute:{
        "[0, 3)": pd.DataFrame([[None for y in range(4)] for x in range(5)]),
        "[3, 10)": pd.DataFrame([[None for y in range(4)] for x in range(5)]),
        "[10, inf)": pd.DataFrame([[None for y in range(4)] for x in range(5)])
    }for attribute in attributes}  for position in positions}
    # Compute the means of each interval
    for position in positions:
        for attribute in attributes:
            for y in range(4):
                for x in range(5):
                    part1_df = velocities[position]['Part 1'][attribute].iloc[x, y]
                    part2_df = velocities[position]['Part 2'][attribute].iloc[x, y]
                    
                    for col in ['[0, 3)', '[3, 10)', '[10, inf)']:
                        part1_values = part1_df[col][part1_df[col] < 27]
                        part2_values = part2_df[col][part2_df[col] < 27]
                        mean = np.mean(part1_values.append(part2_values))
                        if len(mean) == 0 or (len(mean) == 1 and np.isnan(mean[0])):
                            mean = 0
                        else:
                            mean = mean.to_numpy()[0]
                    
                        mean_dfs[position][attribute][col].iloc[x, y] = mean
                        pos_dfs[position][attribute][col].iloc[x, y] = len(part1_values.append(part2_values))
    
    return mean_dfs,pos_dfs







from mpl_toolkits.axes_grid1 import make_axes_locatable

def fieldgram_visualization_all(position):
    mean_velocities, position_matrix = compute_means(velocities)
    vmax = 1
    vmin = 0
    
    from Field_paint import createPitch
    
    intervals = ['[0, 3)', '[3, 10)', '[10, inf)']
    attributes = ['all', 'attacking', 'defending']
    
    fig, ax = plt.subplots(nrows=len(attributes), ncols=len(intervals), figsize=(25, 20))
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    letter_idx = 0
    norms = []
    for i, attribute in enumerate(attributes):
        for j, interval in enumerate(intervals):
            norm = max(max(position_matrix[position][attribute][interval].to_numpy().tolist()))
            norms.append(norm)

    norm = max(norms)
    for i, attribute in enumerate(attributes):
        for j, interval in enumerate(intervals):
            cmap_eff = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#7EAF34','#FF0000'])
            createPitch(pitchLengthX, pitchWidthY, 'meters', 'black', fig=fig, ax=ax[i, j])

            pos = ax[i, j].imshow((position_matrix[position][attribute][interval].to_numpy()/max(max(position_matrix[position][attribute][interval].to_numpy().tolist()))).tolist(), extent=[0, pitchLengthX, 0, pitchWidthY], aspect='auto', cmap=cmap_eff, vmax=vmax, vmin=vmin)
            
            # Add a letter outside the plot space
            if j == 2:
                divider = make_axes_locatable(ax[i, j])
                cax = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(cax)
                cbar = fig.colorbar(pos, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                
                pos.set_clim(0, 1)
            
            letter = f'({letters[letter_idx]})'
            letter_idx += 1
            ax[i, j].text(-0.15, 0.9, letter, transform=ax[i, j].transAxes, fontsize=35, va='center')
            
            # Add text to the center of each cell
            nrows = len(mean_velocities[position][attribute][interval].to_numpy().tolist())
            ncols = len(mean_velocities[position][attribute][interval].to_numpy().tolist()[0])
            for k in range(nrows):
                for l in range(ncols):
                    value_rounded = np.round(mean_velocities[position][attribute][interval].to_numpy().tolist()[k][l], decimals=2)
                    x = (l + 0.5) * pitchLengthX / ncols
                    y = (nrows - 1 - k + 0.5) * pitchWidthY / nrows
                    ax[i, j].text(x, y, value_rounded, ha='center', va='center', color='black', fontsize = 25)


    
    
def fieldgram_visualization(position, visual):
    mean_velocities, position_matrix = compute_means(velocities)
    
    vmax = 1
    vmin = 0
    
    from Field_paint import createPitch
    
    intervals = ['[0, 3)', '[3, 10)', '[10, inf)']
    attributes = ['all', 'attacking', 'defending']
    
    fig, ax = plt.subplots(nrows=len(attributes), ncols=len(intervals), figsize=(25, 20))
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    letter_idx = 0
    norms = []
    def mat_visual(visual):
        if visual == 'col':
            def col_mean(matrix):
                rows = len(matrix)
                cols = len(matrix[0])
                for j in range(cols):
                    col_sum = 0
                    for i in range(rows):
                        col_sum += matrix[i][j]
                    mean = col_sum / rows
                    for i in range(rows):
                        matrix[i][j] = mean
                return matrix
    
            value = col_mean(mean_velocities[position][attribute][interval].to_numpy().tolist())
            posit = col_mean(position_matrix[position][attribute][interval].to_numpy().tolist())
    
        else:
    
            def row_mean(matrix):
                rows = len(matrix)
                cols = len(matrix[0])
                for j in range(rows):
                    row_sum = sum(matrix[j])
                    mean = row_sum / cols
                    for i in range(cols):
                        matrix[j][i] = mean
                return matrix
    
            value = row_mean(mean_velocities[position][attribute][interval].to_numpy().tolist())
            posit = row_mean(position_matrix[position][attribute][interval].to_numpy().tolist())
        return value, posit
    
    for i, attribute in enumerate(attributes):
        for j, interval in enumerate(intervals):
            value, posit = mat_visual(visual)
            norms.append(max(max(posit)))
    norm = max(norms)
    for i, attribute in enumerate(attributes):
        for j, interval in enumerate(intervals):
            cmap_eff = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#7EAF34','#FF0000'])
            createPitch(pitchLengthX, pitchWidthY, 'meters', 'black', fig=fig, ax=ax[i, j])
            value, posit = mat_visual(visual)
            pos = ax[i, j].imshow((np.array(posit)/max(max(posit))).tolist(), extent=[0, pitchLengthX, 0, pitchWidthY], aspect='auto', cmap=cmap_eff,vmax = vmax,vmin = vmin)
            # Add a letter outside the plot space
            if j == 2:
                divider = make_axes_locatable(ax[i, j])
                cax = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(cax)
                cbar = fig.colorbar(pos, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                
                pos.set_clim(0, 1)
            
            letter = f'({letters[letter_idx]})'
            letter_idx += 1
            ax[i, j].text(-0.15, 0.9, letter, transform=ax[i, j].transAxes, fontsize=35, va='center')

    

            
            # Add text to the center of each cell
            nrows = len(value)
            ncols = len(value[0])
            for k in range(ncols) if visual == 'col' else range(nrows):
                value_rounded =np.round(value[0][k], decimals=2) if visual == 'col' else np.round(value[k][0], decimals=2)
                if visual == 'col':
                    x = (k + 0.5) * pitchLengthX / ncols
                    y = (nrows - 1 - k) * pitchWidthY / nrows
                else:
                    x = pitchLengthX / 2 + 5
                    y = (nrows - 1 - k + 0.5) * pitchWidthY / nrows
                ax[i, j].text(x, y, value_rounded, ha='center', va='center', color='black', fontsize = 35)

fieldgram_visualization_all('Goalkeeper')  
fig = plt.gcf()
fig.savefig('Goalkeeper.png', format='png', dpi=1200) 
fieldgram_visualization_all('Defender')  
fig = plt.gcf()
fig.savefig('Defender.png', format='png', dpi=1200) 
fieldgram_visualization_all('Midfielder')
fig = plt.gcf()
fig.savefig('Midfielder.png', format='png', dpi=1200)   
fieldgram_visualization_all('Forward')  
fig = plt.gcf()
fig.savefig('Forward.png', format='png', dpi=1200) 
fieldgram_visualization_all('All')  

#Figures paper 

fieldgram_visualization('All','row') 

fieldgram_visualization('All','col')    

"""
fig = plt.gcf()
fig.savefig('Forward.pdf', format='pdf', dpi=1200)   
"""
        
################################## Specific player analysis ##################################################

def compute_mean_players(velocities,position,player,attribute):
    
    # Initialize the resulting dataframes
    mean_dfs = {position:  {attribute:{
        "[0, 3)": pd.DataFrame([[0 for y in range(4)] for x in range(5)]),
        "[3, 10)": pd.DataFrame([[0 for y in range(4)] for x in range(5)]),
        "[10, inf)": pd.DataFrame([[0 for y in range(4)] for x in range(5)])
    }}  }
    
    pos_dfs = {position:  {attribute:{
        "[0, 3)": pd.DataFrame([[0 for y in range(4)] for x in range(5)]),
        "[3, 10)": pd.DataFrame([[0 for y in range(4)] for x in range(5)]),
        "[10, inf)": pd.DataFrame([[0 for y in range(4)] for x in range(5)])
    }}  }
    # Compute the means of each interval
    total_len = 0
    for y in range(4):
        for x in range(5):
            part1_df = velocities[position]['Part 1'][attribute][player].iloc[x, y]
            part2_df = velocities[position]['Part 2'][attribute][player].iloc[x, y]
            
            for col in ['[0, 3)', '[3, 10)', '[10, inf)']:
                part1_values = part1_df[col][part1_df[col] < 27]
                part2_values = part2_df[col][part2_df[col] < 27]
                mean = np.mean(part1_values.append(part2_values))
                
                if len(mean) == 0 or (len(mean) == 1 and np.isnan(mean[0])):
                    mean = 0
                else:
                    mean = mean.to_numpy()[0]
            
                mean_dfs[position][attribute][col].iloc[x, y] = mean
                pos_dfs[position][attribute][col].iloc[x, y] = len(part1_values.append(part2_values))

    return mean_dfs,pos_dfs


def fieldgram_visualization_player(position, player):
    vmax = 1 
    vmin = 0
    from Field_paint import createPitch
    
    intervals = ['[0, 3)', '[3, 10)', '[10, inf)']
    attributes = ['all', 'attacking', 'defending']
    
    fig, ax = plt.subplots(nrows=len(attributes), ncols=len(intervals), figsize=(25, 20))
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    letter_idx = 0
    norms = []
    for i, attribute in enumerate(attributes):
        mean_velocities, position_matrix = compute_mean_players(velocities_player,position,player,attribute)
        for j, interval in enumerate(intervals):
            norm = max(max(position_matrix[position][attribute][interval].to_numpy().tolist()))
            norms.append(norm)
    
    for i, attribute in enumerate(attributes):
        mean_velocities, position_matrix = compute_mean_players(velocities_player,position,player,attribute)
        for j, interval in enumerate(intervals):
            cmap_eff = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#7EAF34','#FF0000'])
            createPitch(pitchLengthX, pitchWidthY, 'meters', 'black', fig=fig, ax=ax[i, j])

            pos = ax[i, j].imshow((position_matrix[position][attribute][interval].to_numpy()/max(max(position_matrix[position][attribute][interval].to_numpy().tolist()))).tolist(), extent=[0, pitchLengthX, 0, pitchWidthY], aspect='auto', cmap=cmap_eff, vmax=vmax, vmin=vmin)
            
            # Add a letter outside the plot space
            if j == 2:
                divider = make_axes_locatable(ax[i, j])
                cax = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(cax)
                cbar = fig.colorbar(pos, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                
                pos.set_clim(0, 1)
            
            letter = f'({letters[letter_idx]})'
            letter_idx += 1
            ax[i, j].text(-0.15, 0.9, letter, transform=ax[i, j].transAxes, fontsize=35, va='center')
            
            # Add text to the center of each cell
            nrows = len(mean_velocities[position][attribute][interval].to_numpy().tolist())
            ncols = len(mean_velocities[position][attribute][interval].to_numpy().tolist()[0])
            for k in range(nrows):
                for l in range(ncols):
                    value_rounded = np.round(mean_velocities[position][attribute][interval].to_numpy().tolist()[k][l], decimals=2)
                    x = (l + 0.5) * pitchLengthX / ncols
                    y = (nrows - 1 - k + 0.5) * pitchWidthY / nrows
                    ax[i, j].text(x, y, value_rounded, ha='center', va='center', color='black', fontsize = 25)
                    
    # Add a suptitle with the player name
    fig.suptitle(f"{player}", fontsize=80)
    
    return fig, ax



fieldgram_visualization_player('Goalkeeper','Jan Oblak ')  
fieldgram_visualization_player('Defender','Sergio Ramos ')  
fieldgram_visualization_player('Defender','Jesús Navas ')  
fieldgram_visualization_player('Midfielder','Luka Modric ')  
fieldgram_visualization_player('Midfielder','Toni Kroos ') 
fieldgram_visualization_player('Midfielder','Federico Valverde ') 
fieldgram_visualization_player('Forward','Lionel Messi ')  
fieldgram_visualization_player('Forward','Karim Benzema ')  
fieldgram_visualization_player('Forward','Joselu ') 
#Ejemplo Javier
fieldgram_visualization_player('Forward','Iñaki Williams ') 

#Faltan
fieldgram_visualization_player('Defender','Daniel Carvajal ') 
fig = plt.gcf()
fig.savefig('Daniel Carvajal.jpg', format='jpg', dpi=1200)  
     
fieldgram_visualization_player('Midfielder','Martin Ødegaard ') 
fig = plt.gcf()
fig.savefig('Martin Ødegaard.jpg', format='jpg', dpi=1200)  

     

def fieldgram_visualization_player_sector(position,player, visual):
    
    vmax = 1
    vmin = 0
    
    from Field_paint import createPitch
    
    intervals = ['[0, 3)', '[3, 10)', '[10, inf)']
    attributes = ['all', 'attacking', 'defending']
    
    fig, ax = plt.subplots(nrows=len(attributes), ncols=len(intervals), figsize=(25, 20))
    
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    letter_idx = 0
    norms = []
    def mat_visual(visual):
        if visual == 'col':
            def col_mean(matrix):
                rows = len(matrix)
                cols = len(matrix[0])
                for j in range(cols):
                    col_sum = 0
                    for i in range(rows):
                        col_sum += matrix[i][j]
                    mean = col_sum / rows
                    for i in range(rows):
                        matrix[i][j] = mean
                return matrix
    
            value = col_mean(mean_velocities[position][attribute][interval].to_numpy().tolist())
            posit = col_mean(position_matrix[position][attribute][interval].to_numpy().tolist())
    
        else:
    
            def row_mean(matrix):
                rows = len(matrix)
                cols = len(matrix[0])
                for j in range(rows):
                    row_sum = sum(matrix[j])
                    mean = row_sum / cols
                    for i in range(cols):
                        matrix[j][i] = mean
                return matrix
    
            value = row_mean(mean_velocities[position][attribute][interval].to_numpy().tolist())
            posit = row_mean(position_matrix[position][attribute][interval].to_numpy().tolist())
        return value, posit
    for i, attribute in enumerate(attributes):
        mean_velocities, position_matrix = compute_mean_players(velocities_player,position,player,attribute)
        for j, interval in enumerate(intervals):
            value, posit = mat_visual(visual)
            norms.append(max(max(posit)))
    norm = max(norms)
    
    for i, attribute in enumerate(attributes):
        mean_velocities, position_matrix = compute_mean_players(velocities_player,position,player,attribute)
        for j, interval in enumerate(intervals):

            cmap_eff = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#7EAF34','#FF0000'])
            createPitch(pitchLengthX, pitchWidthY, 'meters', 'black', fig=fig, ax=ax[i, j])
            value, posit = mat_visual(visual)
            pos = ax[i, j].imshow((np.array(posit)/norm).tolist(), extent=[0, pitchLengthX, 0, pitchWidthY], aspect='auto', cmap=cmap_eff,vmax = vmax,vmin = vmin)
            # Add a letter outside the plot space
            if j == 2:
                divider = make_axes_locatable(ax[i, j])
                cax = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(cax)
                cbar = fig.colorbar(pos, cax=cax)
                for t in cbar.ax.get_yticklabels():
                    t.set_fontsize(20)
                
                pos.set_clim(0, 1)
            
            letter = f'({letters[letter_idx]})'
            letter_idx += 1
            ax[i, j].text(-0.15, 0.9, letter, transform=ax[i, j].transAxes, fontsize=35, va='center')

    

            
            # Add text to the center of each cell
            nrows = len(value)
            ncols = len(value[0])
            for k in range(ncols) if visual == 'col' else range(nrows):
                value_rounded =np.round(value[0][k], decimals=2) if visual == 'col' else np.round(value[k][0], decimals=2)
                if visual == 'col':
                    x = (k + 0.5) * pitchLengthX / ncols
                    y =  pitchWidthY / 2
                else:
                    x = pitchLengthX / 2 + 5
                    y = (nrows - 1 - k + 0.5) * pitchWidthY / nrows
                ax[i, j].text(x, y, value_rounded, ha='center', va='center', color='black', fontsize = 35)

fieldgram_visualization_player_sector('Goalkeeper','Jan Oblak ','col') 
fieldgram_visualization_player_sector('Defender','Sergio Ramos ','col')  
fieldgram_visualization_player_sector('Defender','Jesús Navas ','row')  
fieldgram_visualization_player_sector('Midfielder','Luka Modric ','row')  
fieldgram_visualization_player_sector('Midfielder','Toni Kroos ','row') 
fieldgram_visualization_player_sector('Midfielder','Federico Valverde ','row') 
fieldgram_visualization_player_sector('Forward','Lionel Messi ','row')  
fieldgram_visualization_player_sector('Forward','Karim Benzema ','row')  
#Ejemplo Javier
fieldgram_visualization_player_sector('Forward','Iñaki Williams ','row') 
