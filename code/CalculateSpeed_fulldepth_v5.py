print('Getting started...')
from glob import glob
import xarray as xr
import numpy as np


###################################################################
### Set path information - needs to be checked before execution ###
print('Set path information...')

datapath = '/storage/shared/oceanparcels/input_data/NEMO4p2_CMCC/'
datapath_nc = datapath + 'MedMFS24_IMMERSE-NEMOv4p2_uncoupled_fulldepth/'
datapath_c = datapath + 'MedMFS24_IMMERSE-NEMOv4p2_coupled_fulldepth/'

outpath_data = '/nethome/ruhs0001/IMMERSE_waves/develop-lorenz/data/CalculateSpeed_fulldepth_v5/'

###################################################################


### 1. Open data
print('Opening input velocity data...')

list_ugrid_nc = sorted(glob(datapath_nc + 'MED24_OBC_1d*_grid_U.nc'))
list_vgrid_nc = sorted(glob(datapath_nc + 'MED24_OBC_1d*_grid_V.nc'))
list_ugrid_c = sorted(glob(datapath_c + 'MED24_OBC_1d*_grid_U.nc'))
list_vgrid_c = sorted(glob(datapath_c + 'MED24_OBC_1d*_grid_V.nc'))

# uncoupled experiment contains data from 2018 to 2020, coupled only 2019 to 2020
# -> need to restrict uncoupled one to period of coupled one
uvars_ugrid_nc = xr.open_mfdataset(list_ugrid_nc[365:], combine='by_coords')
vvars_vgrid_nc = xr.open_mfdataset(list_vgrid_nc[365:], combine='by_coords')
uvars_ugrid_c = xr.open_mfdataset(list_ugrid_c, combine='by_coords')
vvars_vgrid_c = xr.open_mfdataset(list_vgrid_c, combine='by_coords')


print('Performing calculation for each depth level individually...')
for i in range(0,141,1):

    zlev = i + 1
    print(f'Depth level {zlev}:')

    ### 2. Get data on same grid
    print('Interpolating velocities onto same grid...')

    ### 2.1 Rename depth coordinate/dimension (it is the same for both variables)
    uvars_ugrid_depth_nc = uvars_ugrid_nc.isel(depthu=i).rename({'depthu':'z'})
    vvars_vgrid_depth_nc = vvars_vgrid_nc.isel(depthv=i).rename({'depthv':'z'})
    uvars_ugrid_depth_c = uvars_ugrid_c.isel(depthu=i).rename({'depthu':'z'})
    vvars_vgrid_depth_c = vvars_vgrid_c.isel(depthv=i).rename({'depthv':'z'})

    ### 2.2 Interpolate
    vvars_c = vvars_vgrid_depth_c.interp_like(uvars_ugrid_depth_c, method='linear')
    vvars_nc = vvars_vgrid_depth_nc.interp_like(uvars_ugrid_depth_nc, method='linear')
    uvars_c = uvars_ugrid_depth_c
    uvars_nc = uvars_ugrid_depth_nc


    ### 3. Calculate speed
    print('Calculating speed...')

    def calc_velspeed(u,v):
        speed = (u**2 + v**2)**(1/2)
        return speed

    u_Enc = uvars_nc.vozocrtx
    v_Enc = vvars_nc.vomecrty
    speed_Enc = calc_velspeed(u_Enc,v_Enc)

    u_Ec = uvars_c.vozocrtx
    v_Ec = vvars_c.vomecrty
    speed_Ec = calc_velspeed(u_Ec,v_Ec)
        
    u_Sc = uvars_c.usd
    v_Sc = vvars_c.vsd
    speed_Sc = calc_velspeed(u_Sc,v_Sc)

    u_EcSc = u_Ec + u_Sc
    v_EcSc = v_Ec + v_Sc
    speed_EcSc = calc_velspeed(u_EcSc,v_EcSc)

    u_EncSc = u_Enc + u_Sc
    v_EncSc = v_Enc + v_Sc
    speed_EncSc = calc_velspeed(u_EncSc,v_EncSc)

    speed_ScProjEcSc = ((u_EcSc * u_Sc) + (v_EcSc * v_Sc)) / speed_EcSc
    
    speed_EcProjEcSc = ((u_EcSc * u_Ec) + (v_EcSc * v_Ec)) / speed_EcSc


    ### 4. Calculate temporal averages of speed
    print('Calculating temporal averages...')

    speed_Enc_tave = speed_Enc.mean(dim='time_counter').compute()
    speed_Ec_tave = speed_Ec.mean(dim='time_counter').compute()
    speed_Sc_tave = speed_Sc.mean(dim='time_counter').compute()
    speed_EcSc_tave = speed_EcSc.mean(dim='time_counter').compute()
    speed_EncSc_tave = speed_EncSc.mean(dim='time_counter').compute()
    speed_ScProjEcSc_tave = speed_ScProjEcSc.mean(dim='time_counter').compute()
    speed_EcProjEcSc_tave = speed_EcProjEcSc.mean(dim='time_counter').compute()
    print('Annual averages done.')

    speed_Enc_tsave = speed_Enc.groupby('time_counter.season').mean(dim='time_counter')
    speed_Ec_tsave = speed_Ec.groupby('time_counter.season').mean(dim='time_counter')
    speed_Sc_tsave = speed_Sc.groupby('time_counter.season').mean(dim='time_counter')
    speed_EcSc_tsave = speed_EcSc.groupby('time_counter.season').mean(dim='time_counter')
    speed_EncSc_tsave = speed_EncSc.groupby('time_counter.season').mean(dim='time_counter') 
    speed_ScProjEcSc_tsave = speed_ScProjEcSc.groupby('time_counter.season').mean(dim='time_counter')
    speed_EcProjEcSc_tsave = speed_EcProjEcSc.groupby('time_counter.season').mean(dim='time_counter')
    print('Seasonal averages done.')


    ### 5. Saving data
    print('Saving data ...')

    nav_lon = u_EcSc.nav_lon.values
    nav_lat = u_EcSc.nav_lat.values

    speed_tave_ds = xr.Dataset(data_vars = {'Enc': speed_Enc_tave.where(speed_Enc_tave !=0),
                                            'Ec': speed_Ec_tave.where(speed_Ec_tave !=0),
                                            'Sc': speed_Sc_tave.where(speed_Sc_tave !=0),
                                            'EcSc': speed_EcSc_tave.where(speed_EcSc_tave !=0),
                                            'EncSc': speed_EncSc_tave.where(speed_EncSc_tave !=0),
                                            'ScProjEcSc': speed_ScProjEcSc_tave.where(speed_ScProjEcSc_tave != 0),
                                            'EcProjEcSc': speed_EcProjEcSc_tave.where(speed_EcProjEcSc_tave != 0)},
                                coords = {'nav_lon': (('y','x'),nav_lon),
                                        'nav_lat': (('y','x'),nav_lat)})
    
    if zlev < 10:
        speed_tave_ds.to_netcdf(f'{outpath_data}Speed-tave-fulldepth-z00{zlev}.nc')
    if (zlev >= 10) and (zlev < 100):
        speed_tave_ds.to_netcdf(f'{outpath_data}Speed-tave-fulldepth-z0{zlev}.nc')
    if (zlev >= 100):
        speed_tave_ds.to_netcdf(f'{outpath_data}Speed-tave-fulldepth-z{zlev}.nc')
    print('Annual averages saved.')

    speed_tsave_ds = xr.Dataset(data_vars = {'Enc': speed_Enc_tsave.where(speed_Enc_tsave !=0),
                                            'Ec': speed_Ec_tsave.where(speed_Ec_tsave !=0),
                                            'Sc': speed_Sc_tsave.where(speed_Sc_tsave !=0),
                                            'EcSc': speed_EcSc_tsave.where(speed_EcSc_tsave !=0),
                                            'EncSc': speed_EncSc_tsave.where(speed_EncSc_tsave !=0),
                                            'ScProjEcSc': speed_ScProjEcSc_tsave.where(speed_ScProjEcSc_tsave != 0),
                                            'EcProjEcSc': speed_EcProjEcSc_tsave.where(speed_EcProjEcSc_tsave != 0)},
                                coords = {'nav_lon': (('y','x'),nav_lon),
                                        'nav_lat': (('y','x'),nav_lat)})
    if zlev < 10:
        speed_tsave_ds.to_netcdf(f'{outpath_data}Speed-tseasonave-fulldepth-z00{zlev}.nc')
    if (zlev >= 10) and (zlev < 100):
        speed_tsave_ds.to_netcdf(f'{outpath_data}Speed-tseasonave-fulldepth-z0{zlev}.nc')
    if (zlev >= 100):
        speed_tsave_ds.to_netcdf(f'{outpath_data}Speed-tseasonave-fulldepth-z{zlev}.nc')
    print('Seasonal averages saved.')

print('All done.')
    

