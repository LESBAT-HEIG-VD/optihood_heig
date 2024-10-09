# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:32:17 2022

@author: stefano.pauletta
"""
import xlwt
import xlrd
import pvlib
import pandas as pd
import pyarrow.feather as feather
import numpy as np
import geopandas
from shapely.geometry import Point
import os
import scipy
from scipy.cluster import vq
from xlutils.copy import copy
from optihood.calpinage import Calpinage_light as cp
import math


class weather:
    """
    Class to download Tipical Meteorogical Year data from PVGIS for a given location.
    Data are saved in the target file while location is assigned from the source excel file ("scenario.xls" by default)
    Soil probe temperature is derived from surface soil temperature at the nearest MeteoSchweiz station, the 0°C limit and a sinusoidal profile along the year.
    A file target is written in a csv.
    """

    def __init__(self,
                 source="scenario.xls",
                 cluster=True,
                 n_clusters=20,
                 soil_param=True,
                 T_soil_min=0,
                 Day_soil_min=155,
                 T_soil_max=10,
                 clustering_vars=[],
                 save_file=False,
                 load_file=False,
                 set_scenario=True,
                 single_scenario=True
                 ):
        """
        Class constructor. Geographic info are taken from the source file
        Methods are applied at the end in sequence to create weather file for optimease
        """
        self.f_plot=False
        self.single_scenario = single_scenario
        self.get_scenario(source)
        self.tilt = 0
        self.az = 0
        self.coordinates = [self.lat, self.long, self.tilt, self.az]
        self.w = 1
        self.l = 2
        self.time_y = "time.yy"
        self.time_m = "time.mm"
        self.time_d = "time.dd"
        self.time_h = "time.hh"
        self.lb_air = "tre200h0"          # tre200s0
        self.lb_ghi = "gls"               # gre000z0
        self.lb_dhi = "str.diffus"        # ods000z0
        self.lb_gT = "ground_temp"
        self.lb_RH = "Rel_Hum"            # ure200s0
        self.lb_wnd = "wind_speed"        # fkl010z0
        self.lb_p = "pressure"            # prestas0
        self.lb_wnd_d = "wind_direction"  # dkl010z0
        self.lb_IR = "IR"                 # oli000z0
        self.n_cluster = n_clusters

        if soil_param == True:
            self.T_soil_min = T_soil_min
            self.Day_soil_min = Day_soil_min
            self.T_soil_max = T_soil_max
        else:
            self.T_soil_min = 0
            self.Day_soil_min = 58
            self.T_soil_max = 999
        self.load_file = load_file
        self.get_TMY_formatted()
        if self.load_file == False:
            self.get_soil()
        if set_scenario:
            self.set_solar_scenarios(source)
        if cluster == True:
            self.do_clustering(clustering_vars)
        if save_file != False:
            self.save_DB(self.target)

        return None
    
    def load_table_values(self):
        self.En_HP=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'En_HP'].values[0]
        self.En_HC=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'En_HC'].values[0]
        self.Tr_r_HP=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_r_HP'].values[0]
        self.Tr_r_HC=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_r_HC'].values[0]
        self.Tr_r_abo=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_r_abo'].values[0]
        self.Tr_r_Pw=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_r_Pw'].values[0]
        self.Tr_N_HP=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_N_HP'].values[0]
        self.Tr_N_HC=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_N_HC'].values[0]            
        self.Tr_N_Pw=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Tr_N_Pw'].values[0]
        self.Taxes=self.tarfElec_table.loc[ self.tarfElec_table.Profil==self.profile,'Taxes'].values[0]
        return None
        
    
    
    def elec(self,profile_elec="Tarif"):
        try: 
            self.COP_HP=float(self.df_transf.loc[(self.df_transf.building==1) & (self.df_transf.label=='HP'),'efficiency'].values[0])
        except:
            self.COP_HP=3    

        self.demand_elec_tot=self.agg_demand.electricityDemand+ \
            (self.agg_demand.spaceHeatingDemand+
            self.agg_demand.domesticHotWaterDemand)/self.COP_HP
        # self.demand_elec_tot.timestamp = self.year_index
        self.demand_elec_tot=self.demand_elec_tot.to_frame(name='elecCons')
        # self.demand_elec_tot.timestamp =pd.to_datetime(self.year_index)
        self.demand_elec_tot=self.demand_elec_tot.sort_index()
        self.demand_elec_tot['cost']=0
        self.summer_end=str(self.demand_elec_tot.index[0].year)+"-10-01"
        self.summer_start=str(self.demand_elec_tot.index[0].year)+"-04-01"
        self.HP_start=6
        self.HP_end=22
        self.el_prof_list=['Double',
                           'Pro_BT_L',
                           'Pro_BT_H',	
                           'Pro_MT_L_SUMMER'	,
                           'Pro_MT_L_Winter'	,
                           'Pro_MT_H_SUMMER'	,
                           'Pro_MT_H_Winter']
        self.tarfElec_table = pd.read_csv(r'..\excels\clustering\TarifElec.csv', sep=";")
        self.tarfElec_table.dropna(inplace=True)
        self.pic_elec_demand=self.demand_elec_tot.elecCons.max()
        if profile_elec=="Tarif":
            if self.demand_elec_tot['elecCons'].sum()<50000:
                self.profile='Double'
            elif self.demand_elec_tot['elecCons'].sum()>1500000 and \
                self.pic_elec_demand>500:
                if self.demand_elec_tot['elecCons'].sum()/self.pic_elec_demand<3000:
                    self.profile='Pro_MT_L_Winter'
                else:
                    self.profile='Pro_MT_H_Winter'
            else:
                if self.demand_elec_tot['elecCons'].sum()/self.pic_elec_demand<3000:
                    self.profile='Pro_BT_L'
                else:
                    self.profile='Pro_BT_H'
            self.load_table_values()        
            self.demand_elec_tot['cost']=(self.En_HC+ \
                                               self.Tr_r_HC+ \
                                                   self.Tr_r_abo*12/365/24*100+ \
                                            self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Tr_N_HC+ \
                                            self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Taxes)/100
            self.demand_elec_tot.loc[(self.demand_elec_tot.index.hour>self.HP_start) &
                                     (self.demand_elec_tot.index.hour<self.HP_end),'cost']= \
                                                    (self.En_HP+ \
                                                    self.Tr_r_HP+ \
                                                    self.Tr_r_abo*12/365/24*100+ \
                                                    self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Tr_N_HP+ \
                                                    self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Taxes)/100
            if 'MT' in self.profile:
                self.profile=self.profile.split(sep='_W')[0]+"_SUMMER"
                self.load_table_values()
                self.demand_elec_tot.loc[ self.summer_start: self.summer_end,
                                     'cost']=(self.En_HC+ \
                                                self.Tr_r_HC+ \
                                                self.Tr_r_abo*12/365/24*100+ \
                                                self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                                self.Tr_N_HC+ \
                                                self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                                self.Taxes)/100
                self.demand_elec_tot.loc[(self.demand_elec_tot.index.hour>self.HP_start) & 
                                         (self.demand_elec_tot.index.hour<self.HP_end),'cost']= \
                                                       (self.En_HP+ \
                                                        self.Tr_r_HP+ \
                                                        self.Tr_r_abo*12/365/24*100+ \
                                                        self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                                        self.Tr_N_HP+ \
                                                        self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                                        self.Taxes)/100
        elif profile_elec=="Spot":
            # if elec profile is based on spot price then the transport componenet of the
            #electricity price is based on MT fares, L or H depending on the DUP
            if self.demand_elec_tot['elecCons'].sum()/self.pic_elec_demand<3000:
                self.profile='Pro_MT_L_Winter'
            else:
                self.profile='Pro_MT_H_Winter'
            self.load_table_values() 
            self.spot = pd.read_csv(r'..\excels\clustering\electricity_spot.csv', sep=";")
            self.spot.index=self.year_index
            self.spot.drop(columns='timestamp')
            
            # self.demand_elec_tot['cost']=self.spot.cost.values
            self.demand_elec_tot['cost']=(self.Tr_r_HC+ \
                                                   self.Tr_r_abo*12/365/24*100+ \
                                            self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Tr_N_HC+ \
                                            self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Taxes)/100
            self.demand_elec_tot.loc[(self.demand_elec_tot.index.hour>self.HP_start) &
                                     (self.demand_elec_tot.index.hour<self.HP_end),'cost']= \
                                                    (self.Tr_r_HP+ \
                                                    self.Tr_r_abo*12/365/24*100+ \
                                                    self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Tr_N_HP+ \
                                                    self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Taxes)/100
            self.profile=self.profile.split(sep='_W')[0]+"_SUMMER"
            self.load_table_values()
            self.demand_elec_tot.loc[ self.summer_start: self.summer_end,
                                 'cost']=(self.Tr_r_HC+ \
                                            self.Tr_r_abo*12/365/24*100+ \
                                            self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Tr_N_HC+ \
                                            self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                            self.Taxes)/100
            self.demand_elec_tot.loc[(self.demand_elec_tot.index.hour>self.HP_start) & 
                                     (self.demand_elec_tot.index.hour<self.HP_end),'cost']= \
                                                   (self.Tr_r_HP+ \
                                                    self.Tr_r_abo*12/365/24*100+ \
                                                    self.Tr_r_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Tr_N_HP+ \
                                                    self.Tr_N_Pw*self.pic_elec_demand/365/24*100+ \
                                                    self.Taxes)/100
            self.demand_elec_tot.cost=self.demand_elec_tot['cost']+self.spot.cost.values
        try:
            self.demand_elec_tot.cost.to_csv(self.df_cmSRC.loc[self.df_cmSRC.label=='electricityResource','variable costs'].values[0],sep=";")    
        except:
            self.demand_elec_tot.cost.to_csv(r'..\excels\clustering\electricity_cost.csv',sep=';')
            # self.demand_elec_tot['elecPrice'].loc[]
                
        return None

    def get_scenario(self, source):
        """
        Method to access the source file and locate data concerning the solar simulations

        """
        workbook = xlrd.open_workbook(source)

        worksheet = workbook.sheet_by_name('profiles')
        self.solar_f_adr = worksheet.cell(3, worksheet.row_values(
            0, start_colx=0, end_colx=None).index('path')).value
        self.demand_f_adr = worksheet.cell(2, worksheet.row_values(
            0, start_colx=0, end_colx=None).index('path')).value
        if self.solar_f_adr != '' or self.solar_f_adr != 'PVGIS':
            self.solar_file = True
        else:
            self.solar_file = False
        self.target = worksheet.cell(1, worksheet.row_values(
            0, start_colx=0, end_colx=None).index('path')).value

        """
        Load new sheet, compute centroid of building portfolio
        and use it for solar radiation data calculations
        """
        worksheet = workbook.sheet_by_name('hood')
        df_columns = worksheet.row_values(0, start_colx=0, end_colx=None)
        self.df_hood = pd.DataFrame(columns=df_columns)
        i = 0
        for i in range(len(df_columns)):
            self.df_hood[df_columns[i]] = worksheet.col_values(
                i, start_rowx=1, end_rowx=None)

        geometry = geopandas.GeoSeries(geopandas.points_from_xy(self.df_hood.loc[:, 'longitude'],
                                                                self.df_hood.loc[:, 'latitude']))
        self.lat = geometry.unary_union.centroid.y
        self.long = geometry.unary_union.centroid.x
        self.W_S_D = -23.45
        self.W_S_A = np.abs(self.lat+self.W_S_D)
        # self.set_solar_scenarios()
        self.load_agg_demand()

        worksheet = workbook.sheet_by_name('solar_technology')
        tecno_df_columns = worksheet.row_values(0, start_colx=0, end_colx=None)
        self.df_tecno = pd.DataFrame(columns=tecno_df_columns)
        i = 0
        for i in range(len(tecno_df_columns)):
            self.df_tecno[tecno_df_columns[i]] = worksheet.col_values(
                i, start_rowx=1, end_rowx=None)
        self.tecno_list = worksheet.col_values(0, start_rowx=1, end_rowx=None)
        worksheet = workbook.sheet_by_name('solar_technology')
        tecno_df_columns = worksheet.row_values(0, start_colx=0, end_colx=None)
        
        worksheet = workbook.sheet_by_name('commodity_sources')
        cmSRC_df_columns = worksheet.row_values(
            0, start_colx=0, end_colx=None)
        self.df_cmSRC = pd.DataFrame(columns=cmSRC_df_columns)
        i = 0
        for i in range(len(cmSRC_df_columns)):
            self.df_cmSRC[cmSRC_df_columns[i]] = worksheet.col_values(
                i, start_rowx=1, end_rowx=None)
        
        
        worksheet = workbook.sheet_by_name('transformers')
        transf_df_columns = worksheet.row_values(
            0, start_colx=0, end_colx=None)
        self.df_transf = pd.DataFrame(columns=transf_df_columns)
        i = 0
        for i in range(len(transf_df_columns)):
            self.df_transf[transf_df_columns[i]] = worksheet.col_values(
                i, start_rowx=1, end_rowx=None)
        # self.PV_eff = float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('pv'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('efficiency')).value)
        # self.ST_eta0= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('solarCollector'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('eta_0')).value)
        # self.ST_a1= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('solarCollector'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('a_1')).value)
        # self.ST_a2= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('solarCollector'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('a_2')).value)
        # self.PVT_eff=float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('pv'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('efficiency')).value)
        # self.PVT_eta0= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('pvt'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('eta_0')).value)
        # self.PVT_a1= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('pvt'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('a_1')).value)
        # self.PVT_a2= float(worksheet.cell(worksheet.col_values(0, start_rowx=0, end_rowx=None).index('pvt'),
        #                                    worksheet.row_values(0, start_colx=0, end_colx=None).index('a_2')).value)
        return None

    def load_agg_demand(self):
        self.agg_demand = pd.DataFrame()

        for bld in self.df_hood.building:
            if bld == self.df_hood.building[0]:
                self.agg_demand = self.load_bld_demand(bld)
                try:
                    self.year_index = self.agg_demand.timestamp.copy()
                except:
                    self.year_index = self.agg_demand.index
            else:
                try:
                    source = self.demand_f_adr + \
                        r"\Building2"+str(int(bld))+".csv"
                    demand = pd.read_csv(source, sep=";")
                    self.agg_demand = self.agg_demand+demand
                except:
                    source = self.demand_f_adr + \
                        r"\Building"+str(int(bld))+".csv"
                    demand = pd.read_csv(source, sep=";")                   
                    demand.set_index('timestamp', drop=True, inplace=True)
                    demand.index=pd.to_datetime(demand.index,format='%d.%m.%Y %H:%M')
                    for col in self.agg_demand.columns:
                        self.agg_demand[col] = self.agg_demand[col]+demand[col]
        # self.agg_demand.timestamp = self.year_index
        return None

    def load_bld_demand(self, bld_name):
        try:
            source = self.demand_f_adr + \
                r"\Building2"+str(int(bld_name))+".csv"
            demand = pd.read_csv(source, sep=";")
        except:
            source = self.demand_f_adr + r"\Building"+str(int(bld_name))+".csv"
            demand = pd.read_csv(source, sep=";")
            demand.set_index('timestamp', inplace=True)
            demand.index=pd.to_datetime(demand.index,format='%d.%m.%Y %H:%M')

        return demand

    def minimum_distance_tilt(self):
        def func(tilt, w, dist, W_S_A):
            value = self.w*math.sin(math.radians(self.tilt)) * \
                1/np.tan(math.radians(self.W_S_A))
            return value
        return None

    def set_solar_scenarios(self, source):
        """
        Method to set the solar scenarios with optimized azimuth and tilt 
        """

        '''
            we subtract the busy area as a section of the long side
            '''
        self.df_hood.loc[:, 'long_side'] = self.df_hood.loc[:, 'long_side'] - \
            self.df_hood.loc[:, 'busy_area']/self.df_hood.loc[:, 'short_side']
        self.df_hood.loc[:, 'roof_area'] = self.df_hood.loc[:,
                                                            'long_side']*self.df_hood.loc[:, 'short_side']
        """
            once the building geometry is defined, the class Calpinage can 
            be called to compute the roof coverage ratio, the number of solar panels
            and the total receiving surface for each optimization scenario for 
            orientation: same of building on short side
                        same of buidlgin on long side
                        full south (180° according to used convention)
            and structure:  portait, with optimal tilt for each case as computed by PVlib 
                            East West (EW) with fixed tilt of 20°
            """

        self.solar_cases = pd.DataFrame(
            columns=['bld_name', 'Techno', 'latitude', 'longitude',
                     'bld_azimut', 'cll_azimut', 'tilt',
                     'row_dist', 'N_panel', 'ratio',
                     'cll_layout',
                     'cll_alignment',
                     'Optimal tilt or max4dist',
                     'el cover ratio',
                     'el coverage', 'el production', 'el demand',
                     'th cover ratio',
                     'th coverage', 'th production', 'th demand',
                     'spec_prod_e', 'spec_prod_th',
                     'ICPe', 'ICPth']
        )
        self.tecno_list = ['pv', 'solarCollector', 'pvt']
        self.layout_list = ['east-west', 'portrait']
        self.arrangement_list = ['south', 'long', 'short', ]
        self.tecno_db = pd.DataFrame(columns=self.tecno_list)
        optimality_list = ['']
        for bld in self.df_hood.building:
            # method to acquire buiding hourly thermal and electricity demand
            # gets a DF with columns=['elec_demand','thermal_demad']
            # and values normalized on annual sum

            if self.df_hood.loc[self.df_hood.building==bld,'Roof_tilt'].iloc[0]>0:
                self.flat_roof=False
                self.roof_ridge=self.df_hood.loc[self.df_hood.building==bld,'Roof_Ridge'].iloc[0]
            else:
                self.flat_roof=True
                self.roof_ridge='flat'
            # method
            tot_demand = self.load_bld_demand(bld)
            heat_demand = tot_demand.loc[
                :, 'spaceHeatingDemand']+tot_demand.loc[:, 'domesticHotWaterDemand']
            try: 
                self.COP_HP=float(self.df_transf.loc[(self.df_transf.building==bld) & (self.df_transf.label=='HP'),'efficiency'].values[0])
            except:
                self.COP_HP=2.6    
            try: 
                self.COP_GWHP=float(self.df_transf.loc[(self.df_transf.building==bld) & (self.df_transf.label=='GWHP'),'efficiency'].values[0])
            except:
                self.COP_GWHP=4.5
            # if self.df_transf.loc[(self.df_transf['building']==bld) & (self.df_transf['label']=='HP'),'active'].any()==1:
            #     elec_demand = tot_demand.loc[:, 'electricityDemand']+ heat_demand / COP_HP
            # elif self.df_transf.loc[(self.df_transf['building']==bld) & (self.df_transf['label']=='GWHP'),'active'].any()==1:
            #     elec_demand = tot_demand.loc[:, 'electricityDemand']+ heat_demand / COP_GWHP
            # else :
            elec_demand = tot_demand.loc[:, 'electricityDemand']
            tecno_PV = self.df_hood.loc[self.df_hood.building ==
                                        bld, 'pv'].iloc[0]
            tecno_ST = self.df_hood.loc[self.df_hood.building ==
                                        bld, 'solarCollector'].iloc[0]
            tecno_PVT = self.df_hood.loc[self.df_hood.building ==
                                         bld, 'pvt'].iloc[0]
            self.tecno_flag = [tecno_PV, tecno_ST, tecno_PVT]
            self.tecno_db.loc[len(self.tecno_db)] = self.tecno_flag
#if the roof is flat, we analyze the case for East-West structure
            # or portrait on flat roof
            if self.flat_roof==True:
                for lay in self.layout_list:
                    if lay == "portrait":
                        f_EW = False
                        opt_type = ['tilt', 'max4dist',
                                    'tilt+10', 'tilt+20']  # ,'tilt+30']
                        # opt_type=['max4dist','tilt+10','tilt+20']
                        # 'tilt+10','tilt+15','tilt+20',
                        # 'tilt+25','tilt+30','tilt+35']
                    elif lay=='east-west':
                        f_EW = True
                        opt_type = ['tilt']
                    
                    for arr in self.arrangement_list:
                        self.opt_tilt = 0
                        for opt in opt_type:
                            for i in range(3):
                                # if self.tecno_flag[i]==1:
                                if self.tecno_db.iloc[int(bld-1), i] == 1:
                                    # case 1: portait parallel to building on short side
                                    #         PVGIS optimal tilt
                                    #         minimum interdistance is 0.6m
                                    roof_short_opt = cp(orientation=float(self.df_hood.loc[self.df_hood.building == bld, 'bld_orientation'].iloc[0]),
                                                        lat=float(
                                                            self.df_hood.loc[self.df_hood.building == bld, 'latitude'].iloc[0]),
                                                        long=float(
                                                            self.df_hood.loc[self.df_hood.building == bld, 'longitude'].iloc[0]),
    
                                                        W=float(
                                                            self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0]),
                                                        L=float(
                                        self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0]),
                                        tilt_EW=20, f_EW=f_EW,f_flat_roof=self.flat_roof,
                                        f_plot=self.f_plot,
                                        d_rows=0.6,
                                        parallel=arr,
                                        optimal=opt,
                                        opt_tilt=self.opt_tilt,
                                        # demand=demand.iloc[:,1],#"electricityDemand","spaceHeatingDemand","domesticHotWaterDemand"
                                        elec_demand=elec_demand,
                                        heat_demand=heat_demand,
                                        tecno=self.tecno_list[i],
                                        irradiance=[
                                        self.irr_TMY.ghi, self.irr_TMY.dhi, self.irr_TMY.dni, self.irr_TMY.temp_air],
                                        tecno_df=self.df_tecno,
                                        COP_HP=self.COP_HP,
                                        COP_GWHP=self.COP_GWHP)
                                    # print('end')
                                    if opt == 'tilt':
                                        self.opt_tilt = roof_short_opt.roof.tilt[0]
                                    self.solar_cases.loc[
                                        self.solar_cases.index.size] = [bld,
                                                                        self.tecno_list[i],
                                                                        float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'latitude'].iloc[0]),
                                                                        float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'longitude'].iloc[0]),
                                                                        float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'bld_orientation'].iloc[0]),
                                                                        float(
                                                                            roof_short_opt.roof.loc[0, 'cll_azimut'])+180,
                                                                        float(
                                                                            roof_short_opt.roof.loc[0, 'tilt']),
                                                                        float(
                                                                            roof_short_opt.roof.loc[0, 'row_dist']),
                                                                        float(
                                                                            roof_short_opt.roof.loc[0, 'N_panel']),
                                                                        float(
                                                                            roof_short_opt.roof.loc[0, 'ratio']),
                                                                        lay,
                                                                        arr,
                                                                        opt,
                                                                        float(
                                                                            roof_short_opt.cover_ratio_el),
                                                                        float(
                                                                            roof_short_opt.annual_cov_el),
                                                                        float(
                                                                            roof_short_opt.annual_prod_el),
                                                                        float(
                                                                            roof_short_opt.elec_demand.sum()),
                                                                        float(
                                                                            roof_short_opt.cover_ratio_th),
                                                                        float(
                                                                            roof_short_opt.annual_cov_th),
                                                                        float(
                                                                            roof_short_opt.annual_prod_th),
                                                                        float(
                                                                            roof_short_opt.heat_demand.sum()),
    
                                                                        float(roof_short_opt.annual_prod_el)/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0])/float(
                                                                            roof_short_opt.roof.loc[0, 'ratio']),
    
                                                                        float(roof_short_opt.annual_prod_th)/(float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])*float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0])*float(
                                                                            roof_short_opt.roof.loc[0, 'ratio'])),
    
                                                                        float(roof_short_opt.annual_cov_el)/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0]),
    
                                                                        float(roof_short_opt.annual_cov_th)/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])/float(
                                                                            self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0]),]
            else:
                # if the roof is not flat
                lay = "superposed"
                f_EW = False
                opt_type = ['none']
                self.opt_tilt = self.df_hood.loc[self.df_hood.building==bld,'Roof_tilt'].iloc[0]
                building_ridge=self.df_hood.loc[self.df_hood.building==bld,'Roof_Ridge'].iloc[0]
                bld_orientation=self.df_hood.loc[self.df_hood.building==bld,'bld_orientation'].iloc[0]
                if building_ridge != 'L' and building_ridge != 'S':
                    print(r'No valid direction for roof_ridge; default to "S"')
                    building_ridge=='S'
                    
                #☻each side of a slanted roof is considered a separate field
                # orientation of field is the tilt direction of a roof side (perpendicular to the roof ridge)
                if building_ridge=='S':
                    field_orientation=[bld_orientation-90,bld_orientation+90]
                    f_W=float(self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0])/math.cos(math.radians(self.opt_tilt))/2
                    f_L=float(self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])
                elif building_ridge=='L':
                    field_orientation=[bld_orientation,bld_orientation+180]
                    f_W=float(self.df_hood.loc[self.df_hood.building == bld, 'short_side'].iloc[0])/math.cos(math.radians(self.opt_tilt))/2
                    f_L=float(self.df_hood.loc[self.df_hood.building == bld, 'long_side'].iloc[0])                       
                        
                opt='none'
                slanted_arrangement_list = ['long' ]
                for arr in slanted_arrangement_list:
                    for f_or in field_orientation:
                        for i in range(3):
                            if self.tecno_db.iloc[int(bld-1), i] == 1:
                                roof_short_opt = cp(orientation=f_or,
                                                    lat=float(
                                                        self.df_hood.loc[self.df_hood.building == bld, 'latitude'].iloc[0]),
                                                    long=float(
                                                        self.df_hood.loc[self.df_hood.building == bld, 'longitude'].iloc[0]),        
                                                    W=f_W,
                                                    L=f_L,
                                                    tilt_EW=20, f_EW=f_EW,f_flat_roof=self.flat_roof,
                                                    f_plot=self.f_plot,
                                                    d_rows=0.1,
                                                    parallel=arr,
                                                    optimal=opt,
                                                    opt_tilt=self.opt_tilt,
                                    # demand=demand.iloc[:,1],#"electricityDemand","spaceHeatingDemand","domesticHotWaterDemand"
                                                    elec_demand=elec_demand,
                                                    heat_demand=heat_demand,
                                                    tecno=self.tecno_list[i],
                                                    irradiance=[self.irr_TMY.ghi, 
                                                                self.irr_TMY.dhi, 
                                                                self.irr_TMY.dni, 
                                                                self.irr_TMY.temp_air],
                                                    tecno_df=self.df_tecno,
                                                    COP_HP=self.COP_HP,
                                                    COP_GWHP=self.COP_GWHP)
                                # print('end')
                                """RIPRENDI DA QUI DOPO CALPINAGE
                                """
                                self.solar_cases.loc[
                                    self.solar_cases.index.size] = [bld,
                                                                    self.tecno_list[i],
                                                                    float(
                                                                        self.df_hood.loc[self.df_hood.building == bld, 'latitude'].iloc[0]),
                                                                    float(
                                                                        self.df_hood.loc[self.df_hood.building == bld, 'longitude'].iloc[0]),
                                                                    float(
                                                                        self.df_hood.loc[self.df_hood.building == bld, 'bld_orientation'].iloc[0]),
                                                                    float(roof_short_opt.roof.loc[0, 'cll_azimut'])+180,
                                                                    float(roof_short_opt.roof.loc[0, 'tilt']),
                                                                    float(roof_short_opt.roof.loc[0, 'row_dist']),
                                                                    float(
                                                                        roof_short_opt.roof.loc[0, 'N_panel']),
                                                                    float(
                                                                        roof_short_opt.roof.loc[0, 'ratio']),
                                                                    lay,
                                                                    arr,
                                                                    opt,
                                                                    float(
                                                                        roof_short_opt.cover_ratio_el),
                                                                    float(
                                                                        roof_short_opt.annual_cov_el),
                                                                    float(
                                                                        roof_short_opt.annual_prod_el),
                                                                    float(
                                                                        roof_short_opt.elec_demand.sum()),
                                                                    float(
                                                                        roof_short_opt.cover_ratio_th),
                                                                    float(
                                                                        roof_short_opt.annual_cov_th),
                                                                    float(
                                                                        roof_short_opt.annual_prod_th),
                                                                    float(
                                                                        roof_short_opt.heat_demand.sum()),
        
                                                                    float(roof_short_opt.annual_prod_el)/f_L/f_W/float(
                                                                        roof_short_opt.roof.loc[0, 'ratio']),
        
                                                                    float(roof_short_opt.annual_prod_th)/(f_L*f_W*float(
                                                                        roof_short_opt.roof.loc[0, 'ratio'])),
        
                                                                    float(roof_short_opt.annual_cov_el)/f_L/f_W,
        
                                                                    float(roof_short_opt.annual_cov_th)/f_L/f_W,
                                                        ]
            print('Change of building')
        self.solar_cases.to_csv('solar_cases.csv', sep=";")
        if self.single_scenario:
            self.single_case()
        else:
            self.multi_case()

        self.solar_cases_select.to_csv('solar_cases_select.csv', sep=";")
        self.write_cases(source)

        return None

    def multi_case(self):
        select_prov=np.empty_like(self.solar_cases.iloc[0,:])
        for bld in self.df_hood.building:
            for lay in self.solar_cases.cll_layout.unique():
                for i in range(3):
                    if self.tecno_db.iloc[int(bld-1), i] == 1:
                        tec = self.tecno_db.columns[i]
                        
                        select = self.solar_cases.loc[
                            self.solar_cases.bld_name == bld, :].loc[
                                self.solar_cases.cll_layout == lay, :].loc[
                                self.solar_cases.Techno == tec, :]
                        if select.empty==False:
                            if lay!="superposed":
                           
                                if tec == 'pv' or tec == 'pvt':
                                    select_prov=np.vstack([select_prov,select.loc[select['ICPe'].idxmax(), :]])
                                    # self.solar_cases_select = pd.concat([self.solar_cases_select,
                                    #                                      select.loc[select['ICPe'].idxmax(), :]], axis=1)
                                    
                                else:
                                    # self.solar_cases_select = pd.concat([self.solar_cases_select,
                                    #                                      select.loc[select['ICPth'].idxmax(), :]], axis=1)
                                    select_prov=np.vstack([select_prov,select.loc[select['ICPth'].idxmax(), :]])
                                    # self.solar_cases_select = self.solar_cases_select.append(
                                    #     select.loc[select['ICPth'].idxmax(), :])
                            else:
                                # self.solar_cases_select = pd.concat([self.solar_cases_select,
                                #                                      select], axis=1)
                                select_prov=np.vstack([select_prov,select])
                        
            if select_prov[0,1]==None:                        
                select_prov=np.delete(select_prov,0,0)
            self.solar_cases_select = pd.DataFrame(select_prov,columns=self.solar_cases.columns)
            
            
            pv_counter = 0
            pvt_counter = 0
            st_counter = 0
            for i in range(self.solar_cases_select.index.size):
                if self.solar_cases_select.bld_name.iloc[i] == bld:
                    if self.solar_cases_select.Techno.iloc[i] == 'pvt':
                        pvt_counter = pvt_counter+1
                        self.solar_cases_select.loc[
                            self.solar_cases_select.index[i], 'Techno'] = 'pvt_'+str(pvt_counter)
                    if self.solar_cases_select.Techno.iloc[i] == 'pv':
                        pv_counter = pv_counter+1
                        self.solar_cases_select.loc[
                            self.solar_cases_select.index[i], 'Techno'] = 'pv_'+str(pv_counter)
                    if self.solar_cases_select.Techno.iloc[i] == 'solarCollector':
                        st_counter = st_counter+1
                        self.solar_cases_select.loc[
                            self.solar_cases_select.index[i], 'Techno'] = 'solarCollector_'+str(st_counter)
        # self.solar_cases_select.columns=self.solar_cases.columns
        return None

    def single_case(self):
        select_prov=np.empty_like(self.solar_cases.iloc[0,:])
        for bld in self.df_hood.building:
            
            for i in range(3):
                if self.tecno_db.iloc[int(bld-1), i] == 1:
                    tec = self.tecno_db.columns[i]
                    
                    select = self.solar_cases.loc[
                        self.solar_cases.bld_name == bld, :].loc[
                            self.solar_cases.Techno == tec, :]
                    if select.empty==False:
                        
                   
                        if tec == 'pv' or tec == 'pvt':
                            # select_prov=np.vstack([select_prov,select.loc[select['ICPe'].idxmax(), :]])spec_prod_e
                            select_prov=np.vstack([select_prov,select.loc[select['spec_prod_e'].idxmax(), :]])
                            # self.solar_cases_select = pd.concat([self.solar_cases_select,
                            #                                      select.loc[select['ICPe'].idxmax(), :]], axis=1)
                            
                        else:
                            # self.solar_cases_select = pd.concat([self.solar_cases_select,
                            #                                      select.loc[select['ICPth'].idxmax(), :]], axis=1)
                            select_prov=np.vstack([select_prov,select.loc[select['spec_prod_th'].idxmax(), :]])
                            # self.solar_cases_select = self.solar_cases_select.append(
                            #     select.loc[select['ICPth'].idxmax(), :])
                        
                        
            if select_prov[0,1]==None:                        
                select_prov=np.delete(select_prov,0,0)
            self.solar_cases_select = pd.DataFrame(select_prov,columns=self.solar_cases.columns)
            # pv_counter = 0
            # pvt_counter = 0
            # st_counter = 0
            # for i in range(self.solar_cases_select.index.size):
            #     if self.solar_cases_select.bld_name.iloc[i] == bld:
            #         if self.solar_cases_select.Techno.iloc[i] == 'pvt':
            #             pvt_counter = pvt_counter+1
            #             self.solar_cases_select.loc[
            #                 self.solar_cases_select.index[i], 'Techno'] = 'pvt_'+str(pvt_counter)
            #         if self.solar_cases_select.Techno.iloc[i] == 'pv':
            #             pv_counter = pv_counter+1
            #             self.solar_cases_select.loc[
            #                 self.solar_cases_select.index[i], 'Techno'] = 'pv_'+str(pv_counter)
            #         if self.solar_cases_select.Techno.iloc[i] == 'solarCollector':
            #             st_counter = st_counter+1
            #             self.solar_cases_select.loc[
            #                 self.solar_cases_select.index[i], 'Techno'] = 'solarCollector_'+str(st_counter)
        return None

    def write_cases(self, source):
        workbook = xlrd.open_workbook(source)

        worksheet = workbook.sheet_by_name('solar')
        wb = copy(workbook)
        w_sheet = wb.get_sheet('solar')
        for i in range(1, 100):
            for j in range(50):
                w_sheet.write(i, j, label=None)
        header = ['label',
                  'bld_name',
                  'from',
                  'to',
                  'connect',
                  'electrical_consumption',
                  'peripheral_losses',
                  'latitude',
                  'longitude',
                  'tilt',
                  'cll_azimut',
                  'eta_0',
                  'a_1',
                  'a_2',
                  'temp_collector_inlet',
                  'delta_temp_n',
                  'capacity_max',
                  'capacity_min',
                  'lifetime',
                  'maintenance',
                  'installation',
                  'planification',
                  'invest_base',
                  'invest_cap',
                  'heat_impact',
                  'elec_impact',
                  'impact_cap',
                  'active',
                  'zenith_angle',
                  'roof_area',
                  'efficiency',
                  'layout',
                  'space',
                  'cll_alignment']
        # self.solar_cases_select.sort_values(['bld_name','Techno'],axis=0,inplace=True)
        self.solar_cases_select.reset_index(drop=True, inplace=True)
        self.solar_cases_select = self.solar_cases_select.rename(
            columns={'Techno': 'label'})

        pv_counter = 0
        pvt_counter = 0
        st_counter = 0

        for casos in range(self.solar_cases_select.index.size):
            # for bld in self.df_hood.building:
            for j in [0, 1, 7, 8, 9]:
                w_sheet.write(
                    casos+1, j, self.solar_cases_select.loc[casos, header[j]])
            for j in [10]:
                w_sheet.write(
                        casos+1, j, self.solar_cases_select.loc[casos, 'cll_azimut'])
                
            for j in [2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30]:
                w_sheet.write(casos+1, j, self.df_tecno.loc[self.df_tecno.label ==
                                                            self.solar_cases_select.loc[casos, 'label'].split('_')[0], header[j]].iloc[0])
            # for j in [1 ,7,8,9,10]:
            #     w_sheet.write(casos+1,j , self.solar_cases_select.loc[casos,header[j]])

            w_sheet.write(
                casos+1, 31, self.solar_cases_select.loc[casos, 'cll_layout'])
            w_sheet.write(casos+1, 29, self.df_hood.loc[
                self.df_hood.building == self.solar_cases_select.loc[casos, 'bld_name'], 'roof_area'].iloc[0])
            w_sheet.write(casos+1, 28, 19.05)
            w_sheet.write(casos+1, 27, 1)
            w_sheet.write(
                casos+1, 32, self.solar_cases_select.loc[casos, 'ratio'])
            w_sheet.write(
                casos+1, 33, self.solar_cases_select.loc[casos, 'cll_alignment'])
            # if self.solar_cases_select.loc[casos,'Techno'].split('_')[0]=='pv':
            #     pv_counter=pv_counter+1
            #     w_sheet.write(casos+1,0,'pv_'+str(pv_counter))
            # if self.solar_cases_select.loc[casos,'Techno'].split('_')[0]=='pvt':
            #     pvt_counter=pvt_counter+1
            #     w_sheet.write(casos+1,0,'pvt_'+str(pvt_counter))
            # if self.solar_cases_select.loc[casos,'Techno'].split('_')[0]=='solarCollector':
            #     st_counter=st_counter+1
            #     w_sheet.write(casos+1,0,'solarCollector_'+str(st_counter))

        wb.save(source)
        print('end of solar optimization')

        return None

    def get_TMY(self):
        """
        Method to request TMY data from PVGIS or load data from file 
        """
        if self.load_file == True:
            self.irr_TMY = pd.read_csv(self.target, sep=";")
            dummy = []
            for tag in self.irr_TMY.columns:
                if 'time' in tag.lower():
                    dummy.append('time.'+tag[-2:])
                else:
                    dummy.append(tag)
            self.irr_TMY.columns = dummy
            self.irr_TMY.index = pd.to_datetime(self.irr_TMY['time.yy'].astype(str) + "-"
                                                + self.irr_TMY['time.mm'].astype(str) + "-" +
                                                self.irr_TMY['time.dd'].astype(str) + " " + self.irr_TMY["time.hh"].astype(str) + ":00:00")
            self.irr_TMY = self.irr_TMY.drop(columns=["time.yy",
                                                      "time.mm",
                                                      "time.dd",
                                                      "time.hh"])
            self.MaxTMYear = 2021 #self.irr_TMY.index.year.max()
            self.irr_TMY.index = self.irr_TMY.index.map(
                lambda t: t.replace(year=self.MaxTMYear))
            self.months = [{'month': 1, 'year': self.MaxTMYear},
                           {'month': 2, 'year': self.MaxTMYear},
                           {'month': 3, 'year': self.MaxTMYear},
                           {'month': 4, 'year': self.MaxTMYear},
                           {'month': 5, 'year': self.MaxTMYear},
                           {'month': 6, 'year': self.MaxTMYear},
                           {'month': 7, 'year': self.MaxTMYear},
                           {'month': 8, 'year': self.MaxTMYear},
                           {'month': 9, 'year': self.MaxTMYear},
                           {'month': 10, 'year': self.MaxTMYear},
                           {'month': 11, 'year': self.MaxTMYear},
                           {'month': 12, 'year': self.MaxTMYear}]

        elif self.solar_file == True:
            path = self.solar_f_adr
            meteo_data = pd.read_csv(path, sep='\t')
            # meteo_data.index = pd.to_datetime(meteo_data["time.yy"].astype(str) + "-" + meteo_data["time.mm"].astype(str) + "-" +
            #                                   meteo_data["time.dd"].astype(str) + " " + meteo_data["time.hh"].astype(str) + ":00:00")
            
            meteo_data.index = pd.to_datetime("2021-" + meteo_data["time.mm"].astype(str) + "-" +
                                              meteo_data["time.dd"].astype(str) + " " + meteo_data["time.hh"].astype(str) + ":00:00")
            
            meteo_data = meteo_data.drop(columns=["stn",
                                                  "time.yy",
                                                  "time.mm",
                                                  "time.dd",
                                                  "time.hh",
                                                  "rre150h0",
                                                  "fkl010h1",
                                                  "tso100hs",
                                                  "nto000sw",
                                                  "str.vert.E",
                                                  "str.vert.S",
                                                  "str.vert.W",
                                                  "str.vert.N",
                                                  "bodenalbedo",
                                                  "ir.vertikal.S",
                                                  "bodenemissivitaet",
                                                  "dewpt",
                                                  "enthalpy",
                                                  "mixratio",
                                                  "wetbulb"])
            meteo_data.columns = (["temp_air",
                                   "pressure",
                                  "relative_humidity",
                                   "wind_speed",
                                   "wind_direction",
                                   "ghi",
                                   "dhi",
                                   "dni",
                                   "IR(h)"])

            self.irr_TMY = meteo_data
            self.MaxTMYear = self.irr_TMY.index.year.max()
            self.irr_TMY.index = self.irr_TMY.index.map(
                lambda t: t.replace(year=self.MaxTMYear))
            self.months = [{'month': 1, 'year': self.MaxTMYear},
                           {'month': 2, 'year': self.MaxTMYear},
                           {'month': 3, 'year': self.MaxTMYear},
                           {'month': 4, 'year': self.MaxTMYear},
                           {'month': 5, 'year': self.MaxTMYear},
                           {'month': 6, 'year': self.MaxTMYear},
                           {'month': 7, 'year': self.MaxTMYear},
                           {'month': 8, 'year': self.MaxTMYear},
                           {'month': 9, 'year': self.MaxTMYear},
                           {'month': 10, 'year': self.MaxTMYear},
                           {'month': 11, 'year': self.MaxTMYear},
                           {'month': 12, 'year': self.MaxTMYear}]
        else:
            PVGIS_output = pvlib.iotools.get_pvgis_tmy(
                self.lat, self.long, map_variables=True)
            self.irr_TMY = PVGIS_output[0]
            self.MaxTMYear = self.irr_TMY.index.year.max()
            self.irr_TMY.index = self.irr_TMY.index.map(
                lambda t: t.replace(year=self.MaxTMYear))
            self.months = PVGIS_output[1]
        if hasattr(self, 'DB_soil'):
            self.align_DB()
        return None

    def get_hourly(self, st_date="2005", end_date="2005", opt=True):
        """
        Method to request historical data between 2005 and 2016 from PVGIS
        """
        self.opt = opt
        self.start_h = pd.to_datetime(st_date, format="%Y-%m-%d", exact=True)
        self.end_h = pd.to_datetime(end_date, format="%Y-%m-%d", exact=True)
        output = pvlib.iotools.get_pvgis_hourly(self.lat, self.long, start=self.start_h, end=self.end_h,
                                                map_variables=True, surface_tilt=self.tilt, optimal_surface_tilt=self.opt, optimalangles=False)
        self.irr_h = output[0]
        self.months_bis = output[1]
        if hasattr(self, 'DB_soil'):
            self.align_DBh()
        return None

    def get_TMY_formatted(self):
        """
        Method to request TMY data from PVGIS and format the output dataframe as wished
        """
        if ~hasattr(self, 'irr_TMY'):
            self.get_TMY()
        self.irr_TMYf = self.irr_TMY.copy()
        self.irr_TMYf.index.name = "utc_time"
        self.irr_TMYf[self.time_y] = self.MaxTMYear  # irr_TMY.index.year
        self.irr_TMYf[self.time_m] = self.irr_TMYf.index.month
        self.irr_TMYf[self.time_d] = self.irr_TMYf.index.day
        self.irr_TMYf[self.time_h] = self.irr_TMYf.index.hour

        if self.load_file == False:
            self.irr_TMYf.rename(columns={"temp_air": self.lb_air, "ghi": self.lb_ghi, "dhi": self.lb_dhi,
                                          "relative_humidity": self.lb_RH, "wind_speed": self.lb_wnd, "pressure": self.lb_p}, inplace=True)
            self.good_list = [self.time_y, self.time_m, self.time_d, self.time_h,
                              self.lb_air, self.lb_ghi, self.lb_dhi, self.lb_RH, self.lb_wnd, self.lb_p]
            self.bad_list = ['IR(h)', 'wind_direction']
            self.irr_TMYf.drop(self.bad_list, axis='columns', inplace=True)
            self.irr_TMYf = self.irr_TMYf.loc[:, self.good_list]

        if hasattr(self, 'DB_soil'):
            self.align_DB()
        return None

    def get_soil(self, start_date="2021-01-01", end_date="2021-12-31"):
        """
        Method to compute soil probe temperature hourlyprofile

        Parameters
        ----------
        start_date : TYPE, optional
            DESCRIPTION. The default is "2021-01-01".
        end_date : TYPE, optional
            DESCRIPTION. The default is "2021-12-31".

        Returns
        -------
        None.

        """
        self.get_stn()
        head_tail = os.path.split(os.path.abspath(__file__))

        # dati=feather.read_feather(os.path.join(head_tail[0],'DB_soil.fea'))
        dati = feather.read_feather('DB_soil.fea')
        self.Src_soil = dati.loc[dati['stn'] ==
                                 self.stn, ['time', 'Soil_temperature']].copy()
        self.Src_soil.set_index('time', inplace=True)
        self.soil_year = self.Src_soil.index.year.max()
        self.soil_mean = pd.to_numeric(self.Src_soil.iloc[:, 0]).mean()
        self.T_soil_max = self.soil_mean
        self.compute_soil()
        if hasattr(self, 'irr_TMY'):
            self.align_DB()
        if hasattr(self, 'irr_TMYf'):
            self.align_DB()
        if hasattr(self, 'irr_h'):
            self.align_DBh()
        return None

    def align_DB(self):
        """
        Method to align time index of TMY dataframe and soil temperature
        """
        dummy_index = pd.date_range(
            '2021-01-01 00:00:00', periods=8760, tz='UTC', freq='h')
        self.irr_TMY.index = dummy_index
        self.DB_soil.index = self.irr_TMY.index
        self.irr_TMY[self.lb_gT] = self.DB_soil['Soil_temperature'].copy()
        if hasattr(self, 'irr_TMYf'):
            self.irr_TMYf.index = dummy_index
            self.irr_TMYf[self.lb_gT] = self.DB_soil['Soil_temperature'].copy()
        return None

    def align_DBh(self):
        """
        Method to align time index of historical dataframe and soil temperature
        """
        self.DB_soilh = self.DB_soil.copy()
        self.DB_soilh.index = self.irr_h.index
        self.irr_h[self.lb_gT] = self.DB_soilh['Soil_temperature'].copy()
        return None

    def save_DB(self, target):
        """
        Method to save to target
        """
        self.irr_TMYf.to_csv(target, sep=';', index=False)
        return None

    def compute_soil(self):
        """
        Method to compute the geothermal probe temerpature during the year as a function of the annual
        soil average temperature in the nearest Meteoshweiz station
        """

        # start=-np.pi*(1-self.Day_soil_min*24/8760) #1400 is index for 3 March
        # end=2*np.pi+start
        start = 1
        end = 8760
        ind = np.linspace(start, end, 8760)
        mean = (self.T_soil_min+self.T_soil_max) / \
            2  # minimum probe temperature is 0°C
        # maximum probe temperature is the yearly average of surface soil temperature
        amp = (self.T_soil_max-self.T_soil_min)/2
        ground_temp = mean+amp * \
            np.sin(3/2*np.pi+(ind-self.Day_soil_min*24)/8760*2*np.pi)
        self.DB_soil = self.Src_soil.copy()
        self.DB_soil['Soil_temperature'] = ground_temp.tolist()
        return None

    def get_stn(self):
        """
        Method to locate the nearest soil temperature station
        (it can be improved...centroid of triangle of 3 nearest stations?)
        """
        # print(os.path.abspath(__file__))
        head_tail = os.path.split(os.path.abspath(__file__))
        # DB_geo= pd.read_csv(os.path.join(head_tail[0],"Soil_stn.csv"),encoding='cp1252',delimiter=';',
        #                                 skip_blank_lines=True,
        #                                 )
        DB_geo = pd.read_csv("Soil_stn.csv", encoding='cp1252', delimiter=';',
                             skip_blank_lines=True,
                             )
        DB_geo = DB_geo.loc[DB_geo['Data source'] == 'MeteoSchweiz']
        target = geopandas.GeoSeries([Point(self.long, self.lat)])
        target_arr = geopandas.GeoSeries()
        gdf = geopandas.GeoDataFrame(
            DB_geo, geometry=geopandas.points_from_xy(DB_geo.Longitude, DB_geo.Latitude))
        for i in range(0, gdf.index.size):
            target_arr = pd.concat([target_arr, target], axis="index")
        target_arr.index = gdf.index
        dist = gdf.distance(target_arr)
        station = dist.loc[dist == dist.min()].index[0]
        self.stn = DB_geo.loc[station, 'stn']
        return None

    def do_clustering(self, clustering_vars):
        self.meteo_daily = self.irr_TMYf.resample('D').agg({'tre200h0': 'mean',
                                                            'gls': 'sum',
                                                            'str.diffus': 'sum',
                                                            'ground_temp': 'mean',
                                                            'pressure': 'mean'})
        self.meteo_daily['week_end'] = [
            1000 if d.weekday() >= 5 else 0 for d in self.meteo_daily.index]
        # self.agg_demand.set_index('timestamp', drop=True, inplace=True)
        # try:
        #     self.agg_demand.index = pd.to_datetime(
        #         self.agg_demand.index,format="%m/%d/%Y %I:%M:%S %p").tz_localize('UTC')
        # except:
        #     self.agg_demand.index = pd.to_datetime(
        #             self.agg_demand.index,format="%d.%m.%Y %H:%M").tz_localize('UTC')
        self.agg_demand_daily = self.agg_demand.resample('D').sum()
        try:
            self.cluster_DB = pd.merge(
                self.meteo_daily.tz_localize(None), self.agg_demand_daily, how='inner', left_index=True, right_index=True)
        except:
            self.cluster_DB = pd.merge(
                self.meteo_daily, self.agg_demand_daily, how='inner', left_index=True, right_index=True)
            
        if clustering_vars == []:
            """
            we cluster on meteo variables and on aggregated demand
            """
            clustering_vars = ['tre200h0', 'gls', 'str.diffus', 'ground_temp', 'pressure', 'week_end',
                               'electricityDemand', 'spaceHeatingDemand', 'domesticHotWaterDemand']  # columns to use for clustering
        clustering_input = self.cluster_DB.loc[:, clustering_vars]

        """Normalize input data and perform clustering
        """
        clustering_input_norm = vq.whiten(clustering_input)
        self.meteo_cluster, self.code_BK = vq.kmeans2(
            clustering_input_norm, self.n_cluster, iter=100, thresh=1e-5, minit="++")

        """locate nearest days to clusters and compute bin
        """
        labels = []
        lab_indx = []
        lab_d = []
        for i in range(self.n_cluster):
            cl, d = vq.vq(clustering_input_norm, [self.meteo_cluster[i]])
            labels.append(d.argmin())
            lab_indx.append(i)
            lab_d.append(d.min())

        """create clustering result table
        """
        self.results = pd.DataFrame(index=self.meteo_daily.index[labels])
        self.results['labels'] = lab_indx
        self.results['count'] = pd.Series(
            self.code_BK).value_counts().loc[lab_indx].values
        self.results['distances'] = lab_d

        return None


if __name__ == "__main__":

    meteo = weather(cluster=True,
                    n_clusters=20,
                    source=r"..\data\excels\clustering\scenario_Annual_4_costs_100%_SH35_cluster.xls",
                    load_file=True,
                    save_file=False)
