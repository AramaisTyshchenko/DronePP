import numpy as np
import pandas
import pvlib
from pvlib import pvsystem

from helpers import middle_node, calculate_tilt_and_azimuth
from nodes import Node


class PVSystemSimulator:
    def __init__(self):
        self.cec_mod_db = pvsystem.retrieve_sam('CECmod')
        self.invdb = pvsystem.retrieve_sam('CECInverter')
        # Accessing the characteristics of one of the modules randomly
        self.inverter_data = self.invdb.iloc[:, np.random.randint(0, high=len(self.invdb))]
        # Define the PV Module and the Inverter from the CEC databases (For example, the first entry of the databases)
        self.module_data = self.cec_mod_db['Aavid_Solar_ASMS_165P']
        # Define Temperature Paremeters
        self.temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
            'open_rack_glass_glass']

    def power_output(self, df_weather: pandas.DataFrame, node1: Node, node2: Node):
        node3 = middle_node(node1, node2)
        # Calculate the tilt and azimuth for node2 using its time
        tilt, azimuth = calculate_tilt_and_azimuth(node1, node2)
        # Calculate the difference
        time_delta = node2.time - node1.time
        hours_diff = time_delta.total_seconds() / 3600
        location = pvlib.location.Location(node3.lat, node3.lon, altitude=node3.alt)

        # Define the basics of the class PVSystem
        system = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=azimuth,
                                         module_parameters=self.module_data,
                                         inverter_parameters=self.inverter_data,
                                         temperature_model_parameters=self.temperature_model_parameters)

        # Creation of the ModelChain object
        """ The example does not consider AOI losses nor irradiance spectral losses"""
        mc = pvlib.modelchain.ModelChain(system, location,
                                         aoi_model='no_loss',
                                         spectral_model='no_loss',
                                         name='AssessingSolar_PV')

        # Pass the weather data to the model
        """ 
        The weather DataFrame must include the irradiance components with the names 'dni', 'ghi', and 'dhi'. 
        The air temperature named 'temp_air' in degree Celsius and wind speed 'wind_speed' in m/s are optional.
        """
        mc.run_model(df_weather)
        return mc.results.dc['p_mp'].sum() * hours_diff
