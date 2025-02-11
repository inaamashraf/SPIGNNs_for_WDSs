"""
    Dataset Generator:
    Using partial code from 
    Vrachimis et al. https://github.com/KIOS-Research/BattLeDIM
"""
import pandas as pd
import numpy as np
import wntr
import pickle
import os
import argparse
import shutil
import time
from math import ceil
import warnings, copy
warnings.filterwarnings('ignore')
import scipy.io

def create_cli_parser():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser  = argparse.ArgumentParser()
    parser.add_argument('--wds',
                        default = 'anytown',
                        type    = str,
                        choices = ['anytown', 'hanoi', 'pescara', 'area_c', 'zhijiang', 'modena', 'pa1', 'balerma', 'area_a', 'l_town', 'kl'],
                        help    = "select the WDS; default is anytown. ")
    parser.add_argument('--sim_start_time',
                        default = '2018-01-01 00:00',
                        type    = str,
                        help    = "specify simulation start time; default is '2018-01-01 00:00'.")
    parser.add_argument('--sim_end_time',
                        default = '2018-01-01 02:30',
                        type    = str,
                        help    = "specify simulation end time; default is '2018-01-01 02:30'.")
    parser.add_argument('--start_scenario',
                        default = 1000,
                        type    = int,
                        help    = "specify start of the range of scenarios; default is 1000.")
    parser.add_argument('--end_scenario',
                        default = 1099,
                        type    = int,
                        help    = "specify end of the range of scenarios; default is 1999.")
    return parser


""" 
    Realistic demand generation method (not used in the paper) 
    adapted from Vrachimis et al. https://github.com/KIOS-Research/BattLeDIM
"""
def genDem():
    weekPat = scipy.io.loadmat('weekPat_30min.mat')
    Aw = weekPat['Aw']
    nw = weekPat['nw']
    yearOffset = scipy.io.loadmat('yearOffset_30min.mat')
    Ay = yearOffset['Ay']
    ny = yearOffset['ny']
     
    # Create yearly component
    days = 365
    
    T=(288/6)*days # one year period in five minute intervals
    w=2*np.pi/T
    k=np.arange(1, days*288/6+1 ,1) # number of time steps in time series
    n=ny[0][0] # number of fourier coefficients
    Hy=[1]*len(k)
    
    for i in range(1,n+1):
        Hy=np.column_stack((Hy, np.sin(i*w*k), np.cos(i*w*k)))
    
    Hy.shape # check size matrix
    uncY=0.1
    AyR = Ay*(1-uncY+2*uncY*np.random.rand(int(Ay.shape[0]),int(Ay.shape[1]))) # randomize fourier coefficients
    yearOffset = np.dot(Hy, AyR)
    
    # Create weekly component
    T=(288/6)*7 #one week period in five minute intervals
    w=2*np.pi/T
    k=np.arange(1, days*288/6+1 ,1) # number of time steps in time series
    n=nw[0][0] # number of fourier coefficients
    Hw=[1]*len(k)
    for i in range(1,n+1):
        Hw=np.column_stack((Hw, np.sin(i*w*k), np.cos(i*w*k)))
    
    uncW=0.1
    AwR = Aw*(1-uncW+2*uncW*np.random.rand(int(Aw.shape[0]),int(Aw.shape[1]))) # randomize fourier coefficients
    weekYearPat = np.dot(Hw, AwR)
    
    # Create random component
    uncR=0.05
    random = np.random.normal(0,(-uncR+2*uncR),(int(weekYearPat.shape[0]),int(weekYearPat.shape[1]))) #normally distributed random numbers
    
    # Create demand
    #blow=30
    #bhigh=35
    base =1#blow+np.random.rand()*(bhigh-blow)
    variation = 0.75+ np.random.normal(0,0.07) # from 0 to 1
    dem = base * (yearOffset+1) * (weekYearPat*variation+1) * (random+1)
    dem = dem.tolist()
    demFinal = []
    for d in dem:
        demFinal.append(d[0])
      
    return demFinal



class DatasetCreator:
    def __init__(self, scenario_folder, inp_file, sim_start_time, sim_end_time, 
                 scenario=1, qunc=np.arange(0, 0.25, 0.05), dem_multiplier=1., pattern_multiplier=1.,
                 pattern_gen=False, wdn_name=None, ignore_leakages=True):

        # Read input arguments from yalm file
        self.scenario_folder = scenario_folder
        self.results_folder = os.path.join(scenario_folder, "results")

        # demand-driven (DD) or pressure dependent demand (PDD)
        Mode_Simulation = 'DD'  # 'PDD'#'PDD'

        # Create Results folder
        self.create_folder(self.results_folder)

        # Load EPANET network file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.wn.options.hydraulic.demand_model = Mode_Simulation

        self.nodes = self.wn.get_graph().nodes()
        self.links = self.wn.link_name_list

        self.inp = os.path.basename(self.wn.name)[0:-4]

        # Get the name of input file
        self.net_name = f'{self.results_folder}{self.inp}'

        # Get time step
        self.time_step = round(self.wn.options.time.hydraulic_timestep)
        # Create time_stamp
        self.time_stamp = pd.date_range(sim_start_time, sim_end_time, freq=str(self.time_step / 60) + "min")

        # Simulation duration in steps
        self.wn.options.time.duration = (len(self.time_stamp) - 1) * self.time_step  # 5min step
        self.TIMESTEPS = int(self.wn.options.time.duration / self.wn.options.time.hydraulic_timestep)

        if pattern_gen:
            # For Demand Pattern Generation
            qunc_index = int(round(np.random.uniform(len(qunc)-1)))
            uncertainty_Length = qunc[qunc_index]
            
            qunc_index = int(round(np.random.uniform(len(qunc)-1)))
            uncertainty_Diameter = qunc[qunc_index]
            
            qunc_index = int(round(np.random.uniform(len(qunc)-1)))
            uncertainty_Roughness = qunc[qunc_index]
            
            qunc_index = int(round(np.random.uniform(len(qunc)-1)))
            uncertainty_base_demand = qunc[qunc_index]

            ###########################################################################  
            ## SET BASE DEMANDS AND PATTERNS      
            # Remove all patterns
            #        # Initial base demands SET ALL EQUAL 1
            #if "ky" in INP:
            self.wn._patterns= {}
            tempbase_demand = self.wn.query_node_attribute('base_demand')
            tempbase_demand = np.array([tempbase_demand[i] for i, line in enumerate(tempbase_demand)]) * dem_multiplier
            
            tmp = list(map(lambda x: x * uncertainty_base_demand, tempbase_demand))
            ql=tempbase_demand-tmp
            qu=tempbase_demand+tmp
            mtempbase_demand=len(tempbase_demand)
            qext_mtempbase_demand=ql+np.random.rand(mtempbase_demand)*(qu-ql)
            
            for w, junction in enumerate(self.wn.junction_name_list):
                self.wn.get_node(junction).demand_timeseries_list[0].base_value = qext_mtempbase_demand[w] #self.wn.query_node_attribute('base_demand')
                pattern_name = 'P_'+junction
                patts = genDem()
                patts = list(((np.array(patts) - 1) * pattern_multiplier) + 1)
                self.wn.add_pattern(pattern_name, patts)
                
                for patterns in self.wn.nodes._data[junction].demand_timeseries_list._list:
                    patterns.pattern_name = pattern_name
                    patterns.pattern.name = pattern_name
            
            ###########################################################################
            ## SET UNCERTAINTY PARAMETER
            # Uncertainty Length
            tempLengths = self.wn.query_link_attribute('length')
            tempLengths = np.array([tempLengths[i] for i, line in enumerate(tempLengths)])
            tmp = list(map(lambda x: x * uncertainty_Length, tempLengths))
            ql=tempLengths-tmp
            qu=tempLengths+tmp
            mlength=len(tempLengths)
            qext=ql+np.random.rand(mlength)*(qu-ql)
                
            # Uncertainty Diameter
            tempDiameters = self.wn.query_link_attribute('diameter')
            tempDiameters = np.array([tempDiameters[i] for i, line in enumerate(tempDiameters)])
            tmp = list(map(lambda x: x * uncertainty_Diameter, tempDiameters))
            ql=tempDiameters-tmp
            qu=tempDiameters+tmp
            dem_diameter=len(tempDiameters)
            diameters=ql+np.random.rand(dem_diameter)*(qu-ql)
                
            # Uncertainty Roughness
            tempRoughness = self.wn.query_link_attribute('roughness')
            tempRoughness = np.array([tempRoughness[i] for i, line in enumerate(tempRoughness)])
            tmp = list(map(lambda x: x * uncertainty_Roughness, tempRoughness))
            ql=tempRoughness-tmp
            qu=tempRoughness+tmp
            dem_roughness=len(tempRoughness)
            qextR=ql+np.random.rand(dem_roughness)*(qu-ql)
            for w, line1 in enumerate(qextR):
                self.wn.get_link(self.wn.link_name_list[w]).roughness=line1
                self.wn.get_link(self.wn.link_name_list[w]).length=qext[w]
                self.wn.get_link(self.wn.link_name_list[w]).diameter=diameters[w]

            filename = os.path.join(scenario_folder, wdn_name + ".inp")
            out_inp = wntr.epanet.io.InpFile()
            out_inp.write(
                filename, 
                wn = self.wn,
                units = None, 
                version = 2.2, 
                force_coordinates = False
                )
            inp_file = filename      


    def create_csv_file(self, values, time_stamp, columnname, pathname):

        file = pd.DataFrame(values)
        file['time_stamp'] = time_stamp
        file = file.set_index(['time_stamp'])
        file.columns.values[0] = columnname
        file.to_csv(pathname)
        del file, time_stamp, values

    def create_folder(self, _path_):

        try:
            if os.path.exists(_path_):
                shutil.rmtree(_path_)
            os.makedirs(_path_)
        except Exception as error:
            pass

    def dataset_generator(self, scenario_times=[], save_data=True):
        # Path of EPANET Input File
        print(f"Dataset Generator run...")
        
        # Save the water network model to a file before using it in a simulation
        with open(os.path.join(self.scenario_folder,'self.wn.pickle'), 'wb') as f:
            pickle.dump(self.wn, f)

        # Run wntr simulator
        scenario_start_time = time.time()
        sim = wntr.sim.WNTRSimulator(self.wn)
        results = sim.run_sim()
        scenario_end_time = time.time()
        scenario_time = scenario_end_time - scenario_start_time
        scenario_times.append(scenario_time)
        print('... simulation done! time taken: \t', scenario_time, ' seconds.')
        
        if results.node["pressure"].empty:
            print("Negative pressures.")
            return -1

        if results:
            decimal_size = 16
           
            # Create xlsx file with Measurements
            def export_measurements(pressure_sensors, flow_sensors, file_out="Measurements.xlsx", save_data=True):
                total_pressures = {'Timestamp': self.time_stamp}
                total_demands = {'Timestamp': self.time_stamp}
                total_flows = {'Timestamp': self.time_stamp}
                total_levels = {'Timestamp': self.time_stamp}
                total_heads = {'Timestamp': self.time_stamp}
                total_velocities = {'Timestamp': self.time_stamp}
                for j in range(0, self.wn.num_nodes):
                    node_id = self.wn.node_name_list[j]
                    pres = results.node['pressure'][node_id]
                    pres = pres[:len(self.time_stamp)]
                    pres = [round(elem, decimal_size) for elem in pres]
                    total_pressures[node_id] = pres

                    head = results.node['head'][node_id]
                    head = head[:len(self.time_stamp)]
                    head = [round(elem, decimal_size) for elem in head]
                    total_heads[node_id] = head

                    dem = results.node['demand'][node_id]
                    dem = dem[:len(self.time_stamp)]
                    dem = [round(elem, decimal_size) for elem in dem] #CMH / L/s
                    total_demands[node_id] = dem

                    level_pres = results.node['pressure'][node_id]
                    level_pres = level_pres[:len(self.time_stamp)]
                    level_pres = [round(elem, decimal_size) for elem in level_pres]
                    total_levels[node_id] = level_pres

                for j in range(0, self.wn.num_links):
                    link_id = self.wn.link_name_list[j]

                    if link_id not in flow_sensors:
                        continue
                    flows = results.link['flowrate'][link_id]
                    flows = [round(elem, decimal_size) for elem in flows]
                    flows = flows[:len(self.time_stamp)]
                    total_flows[link_id] = flows

                    velocities = results.link['velocity'][link_id]
                    velocities = [round(elem, decimal_size) for elem in velocities]
                    velocities = velocities[:len(self.time_stamp)]
                    total_velocities[link_id] = velocities

                dem_multiplier = self.wn.options.hydraulic.demand_multiplier
                n_timesteps = len(self.time_stamp)
                orig_demands = {'Timestamp': self.time_stamp}
                for node_id in self.wn.node_name_list:
                    node_dem = 0
                    if self.wn.nodes._data[node_id].node_type == 'Junction' and 'leak' not in node_id:
                        for patterns in self.wn.nodes._data[node_id].demand_timeseries_list._list:
                            if patterns.pattern is not None and patterns.pattern.multipliers is not None:
                                node_dem += (patterns.base_value * dem_multiplier) * patterns.pattern.multipliers[: n_timesteps]                                    
                            else:
                                node_dem += patterns.base_value * dem_multiplier

                    try:
                        repeat_idx = ceil(n_timesteps / len(node_dem))
                        node_dem_copy = copy.deepcopy(node_dem)
                        for i in range(1, repeat_idx):
                            node_dem = np.concatenate((node_dem, node_dem_copy))
                        orig_demands[node_id] = node_dem[: n_timesteps] #* 3600 * 1000
                    except:
                        orig_demands[node_id] = node_dem

                # Create a Pandas dataframe from the data.                
                df1 = pd.DataFrame(total_pressures)
                df2 = pd.DataFrame(total_demands)
                df3 = pd.DataFrame(total_flows)
                df4 = pd.DataFrame(total_levels)
                df5 = pd.DataFrame(total_heads)
                df6 = pd.DataFrame(orig_demands)
                df7 = pd.DataFrame(total_velocities)
                print(df1[df1 != 0].min(numeric_only=True).min(), df1.max(numeric_only=True).max())
                print(df5.min(numeric_only=True).min(), df5.max(numeric_only=True).max())

                if save_data:
                    # Create a Pandas Excel writer using XlsxWriter as the engine.
                    writer = pd.ExcelWriter(os.path.join(self.results_folder, file_out), engine='xlsxwriter')

                    # Convert the dataframe to an XlsxWriter Excel object.
                    # Pressures (m), Demands (m^3/s), Flows (m^3/s), Levels (m)
                    df1.to_excel(writer, sheet_name='Pressures (m)', index=False, float_format="%.12f")
                    df2.to_excel(writer, sheet_name='Demands (m3_s)', index=False, float_format="%.12f")
                    df3.to_excel(writer, sheet_name='Flows (m3_s)', index=False, float_format="%.12f")
                    df4.to_excel(writer, sheet_name='Levels (m)', index=False, float_format="%.12f")
                    df5.to_excel(writer, sheet_name='Heads (m)', index=False, float_format="%.12f")
                    df6.to_excel(writer, sheet_name='Orig_Demands (m)', index=False, float_format="%.12f")
                    df7.to_excel(writer, sheet_name='Velocities (m_s)', index=False, float_format="%.12f")

                    # Close the Pandas Excel writer and output the Excel file.
                    writer._save()

                return df5, df3, df6

            heads_df, flows_df, demands_df = \
                export_measurements(self.nodes, self.links, "Measurements_All.xlsx", save_data=save_data)

            # Clean up
            os.remove(os.path.join(os.getcwd(), self.scenario_folder,'self.wn.pickle'))
        else:
            print('Results empty.')
            return -1

        return scenario_times, heads_df, flows_df, demands_df

def run_data_gen(wds = 'pescara', start_scenario=1000, end_scenario=1099, scenario_times=[], 
        save_dir='', sim_start_time='2018-01-01 00:00', sim_end_time = '2018-01-01 23:30',
        pattern_gen=False, qunc=np.arange(0, 0.25, 0.05), pattern_multiplier=1.0, reservoir_multiplier=1.0,
        sigma_d=1/30, mu_dem = 1., sigma_dem = .1, dem_multiplier=1.0, dem_addition=0.0, in_seed=None,
        save_data=True):
    

    for s in range(start_scenario, end_scenario + 1):
        inp_file = os.path.join(save_dir, wds+'.inp')
        scenario = 's' + str(s)
        scenario_dir = os.path.join(save_dir, scenario)
        if not os.path.isdir(scenario_dir):
            os.system('mkdir ' + scenario_dir)

        if in_seed is None:
            _seed = s
        else:
            _seed = in_seed
        np.random.seed(_seed)
        print(_seed, np.random.get_state()[1][0])

        t = time.time()

        if not pattern_gen:

            wn = wntr.network.WaterNetworkModel(inp_file)

            wn.options.hydraulic.demand_multiplier = dem_multiplier

            """ Generating demand patterns by sampling from normal distributions. """
            _len = 48*7*2
            for node_id in wn.node_name_list:
                if wn.nodes._data[node_id].node_type == 'Junction' and 'leak' not in node_id:
                    for patterns in wn.nodes._data[node_id].demand_timeseries_list._list:
                        pattern = np.round(np.random.normal(mu_dem, sigma_dem, size = _len), 6).clip(0) 
                        pattern_offset = np.round(np.random.normal(mu_dem, sigma_dem, size = _len), 6).clip(0)  
                        wn.add_pattern(
                            name = "random_week_"+str(node_id), 
                            pattern = pattern + pattern_offset
                            )
                        patterns.pattern_name = "random_week_"+str(node_id)
                        patterns.pattern.name = "random_week_"+str(node_id)
                        patterns.pattern.multipliers = pattern + pattern_offset

                        patterns.base_value += dem_addition
            
            """ Adding noise to the diameters sampled from a normal distribution. """
            # _seed = s
            np.random.seed(_seed)
            print(_seed, np.random.get_state()[1][0])
            _min, _max = np.min(wn.query_link_attribute('diameter')), 1.
            for key, value in wn.links._data.items():
                if wn.links._data[key].link_type == 'Pipe':
                    wn.links._data[key].diameter = \
                        (wn.links._data[key].diameter * ( 1 + np.random.normal(0, sigma_d, size=1)[0] )).clip(min=_min, max=_max) 
            
            """ Saving the new network configuration file. """
            filename = os.path.join(scenario_dir, wds+'.inp')
            out_inp = wntr.epanet.io.InpFile()
            out_inp.write(
                filename, 
                wn = wn,
                units = None, 
                version = 2.2, 
                force_coordinates = False
                )
            inp_file = filename

        # Call dataset creator        
        L = DatasetCreator(scenario_dir, inp_file, sim_start_time, sim_end_time, scenario, 
                            pattern_gen=pattern_gen, qunc=qunc, wdn_name=wds, 
                            dem_multiplier=dem_multiplier, pattern_multiplier=pattern_multiplier)
        scenario_times, heads_df, flows_df, demands_df = L.dataset_generator(scenario_times, save_data=save_data)

        print('\nScenario ' + scenario + ' generated. Total Elapsed time is ' + str(time.time() - t) + ' seconds.\n')

    if save_data:
        return scenario_times
    else:
        return scenario_times, heads_df, flows_df, demands_df


if __name__ == '__main__':

    parser = create_cli_parser()

    args = parser.parse_args()    

    dm_dict = {
                "anytown":        1.0,
                "hanoi":          0.5,
                "pescara":        0.3,
                "area_c" :        2.0,
                "zhijiang":       0.05,
                "modena":         0.3,
                "balerma":        0.5,
                "pa1":            1.0,
                "marchirural":    3.0,
                "kl":             0.3,
                "area_a":         2.0,
                "c_town":         0.2,
                "l_town":         2.0,
                "pa2":            0.2,
                }

    dem_multiplier = dm_dict[args.wds]
    pattern_multiplier = 1.

    sigma_d = .01
    mu_dem, sigma_dem = 1., .1
    in_seed = None 
    dem_addition = 0    
    reservoir_multiplier = 1       

    save_dir = os.path.join(os.getcwd(), "wds", args.wds, "toy")

    scenario_times = []
    scenario_times = run_data_gen(args.wds, args.start_scenario, args.end_scenario, scenario_times, 
                         save_dir, args.sim_start_time, args.sim_end_time,
                         pattern_gen=False, pattern_multiplier=pattern_multiplier,
                         sigma_d=sigma_d, mu_dem=mu_dem, sigma_dem=sigma_dem,
                         dem_multiplier=dem_multiplier, dem_addition=dem_addition,
                         reservoir_multiplier=reservoir_multiplier,
                         in_seed=in_seed)    