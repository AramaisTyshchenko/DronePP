import pandas as pd
import requests

from helpers import round_time
from nodes import Node


class WeatherData:
    def __init__(self):
        self.weather_data = None
        self.table_date = None

    def get_weather(self, service_type: str, node: Node):
        # Round to hours and convert datetime to string
        # time_from = round_time(node1.time, 60).strftime('%Y-%m-%d %H:%M:%S')
        # time_to = round_time(node2.time, 60).strftime('%Y-%m-%d %H:%M:%S')

        if service_type == 'open_meteo':
            return self.get_open_meteo_weather(node)
        elif service_type == 'nrel':
            df = self.get_nrel_weather(node)
            raise NotImplementedError
        # return self.weather_data.loc[time_from:time_to]

    def get_weather_for_edge(self, node1: Node, node2: Node):
        # # If the weather data for the required date is already stored, just return the required slice
        # if self.weather_data is not None and self.table_date == node1.time.date():
        #     return self.weather_data.loc[time_from:time_to]
        # Round to hours and convert datetime to string
        time_from = round_time(node1.time, 60).strftime('%Y-%m-%d %H:%M:%S')
        time_to = round_time(node2.time, 60).strftime('%Y-%m-%d %H:%M:%S')

        if not node1.weather.columns.equals(node2.weather.columns):
            raise ValueError('Weather columns are not equal')
        weather_avg = node1.weather.add(node2.weather, fill_value=0) / 2
        return weather_avg.loc[time_from:time_to]

    def get_open_meteo_weather(self, node: Node):
        # Otherwise, fetch the weather data for the whole day
        url = "https://archive-api.open-meteo.com/v1/archive"
        # Format the datetime object as a string in the 'YYYY-MM-DD' format
        start_date = node.time.strftime('%Y-%m-%d')
        end_date = start_date

        # parameters
        params = {
            "latitude": node.lat,
            "longitude": node.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m,shortwave_radiation,direct_radiation,diffuse_radiation,direct_normal_irradiance",
            "windspeed_unit": "ms"
        }

        # make the request
        response = requests.get(url, params=params)
        # convert the response to json
        data = response.json()
        # create a DataFrame from the 'hourly' data
        df = pd.DataFrame(data['hourly'])
        # convert the 'time' column into datetime
        df['time'] = pd.to_datetime(df['time'])

        # rename the columns to match your requirements
        df.rename(columns={
            'temperature_2m': 'temp_air',
            'windspeed_100m': 'wind_speed',
            'shortwave_radiation': 'ghi',
            'direct_normal_irradiance': 'dni',
            'diffuse_radiation': 'dhi'
        }, inplace=True)

        # set the 'time' as the index
        df.set_index('time', inplace=True)

        # Store the fetched data and the date for which it was fetched
        self.weather_data = df
        self.table_date = node.time.date()
        return df

    def get_nrel_weather(self, node: Node, year=2020, interval=30):
        """

        :param lat:
        :param lon:
        :param year:
        :param interval:
        :return:
        """
        api_key = 'kmV4pNdF3xCMhFHGPtiUJcBiZEl6DXR8IBE0tXQj'
        attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'
        leap_year = 'false'
        utc = 'false'
        your_name = 'Aramais+Tyshchenko'
        reason_for_use = 'beta+testing'
        your_affiliation = 'Warwick+Uni'
        your_email = 'Aramais.Tyshchenko@warwick.ac.uk'
        mailing_list = 'false'

        # Return all but first 2 lines of csv to get data:
        df = pd.read_csv(
            'https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(
                year=year, lat=node.lat, lon=node.lon, leap=leap_year, interval=interval, utc=utc, name=your_name,
                email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use,
                api=api_key, attr=attributes), skiprows=2)

        # Set the time index in the pandas dataframe:
        df = df.set_index(
            pd.date_range('1/1/{yr}'.format(yr=year), freq=str(interval) + 'Min', periods=525600 / int(interval)))

        df = df.rename(columns={'Temperature': 'temp_air', 'Wind Speed': 'wind_speed'})
        df.columns = df.columns.str.lower()
        return df
