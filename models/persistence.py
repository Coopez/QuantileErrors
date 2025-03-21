"""
Code recieved from IFE.


"""


import pvlib
import numpy as np
import pandas as pd
import datetime as dt


class PVForecast:

    def __init__(self, measured_signal: pd.Series, binning: str,
                 f_h: str, f_i: str, s_h: str, location: pvlib.location.Location, max_sza=86.3) -> None:
        """

        :param measured_signal:
        :param binning:
        :param f_h: horizon
        :param f_i: forecast issuing frequency
        :param s_h:
        :param location:
        :param max_sza:
        """

        self.measured_signal = measured_signal
        #       self.freq = pd.infer_freq(self.measured_signal.index)
        self.binning = binning # what is this?
        self.f_h = f_h
        self.f_i = f_i
        self.s_h = s_h
        self.max_sza = max_sza
        self.location = location
        self.f_h_periods = self.get_periods(self.f_h, self.f_i)
        #       self.f_i_periods = self.get_periods(self.f_i, self.f_i)
        self.s_h_periods = self.get_periods(self.s_h, self.f_i)

        self.ensure_identical_timezone()
        self.remove_night_timestamps()

    def ensure_identical_timezone(self) -> None:
        """
        Ensure identical time-zone (tz) for measured signal and location.
        :return: None
        """

        assert str(self.measured_signal.index.tz) == str(self.location.tz), \
            r'Incompatible time-zone (tz) between measured signal (tz={}) and location (tz={})'.format(
                self.measured_signal.index.tz, self.location.tz)

    def remove_night_timestamps(self) -> None:
        """
        Set timestamps to np.nan when the solar zenith angle (SZA) is larger than max_SZA (defined as night).
        :return: None
        """
        # Get solar zenith angles (SZA):
        sza = self.location.get_solarposition(self.measured_signal.index)['zenith']

        # Set night timestamps to np.nan:
        self.measured_signal[(sza > self.max_sza)] = np.nan

    def get_clearness(self, G_0=1360) -> pd.Series:
        """
        Calculate clearness.

        :param G_0: float
            the extra terrestrial radiation received by the Earth from the Sun
        :return: G_clear : pandas.Series

        """

        # Get solar zenith angle (SZA):
        sza = self.location.get_solarposition(self.measured_signal.index)['zenith']

        # Calculate cos(SZA):
        cos_zenith = pvlib.tools.cosd(sza)

        # Calculate clearness:
        G_clear = G_0 * np.maximum(cos_zenith, np.cos(np.radians(self.max_sza)))

        return G_clear

    def get_clearsky_index(self, G_clear, max_clearness_index=2.0) -> pd.Series:
        """
        Calculate clearness index.

        :param G_clear: pandas.Series
            Clearsky Global Horizontal Irradiance (GHI) in W/m2
        :param max_clearness_index: float
            maximum value for clearness index
        :return: kt : pandas.Series

        """

        # Calculate cleaness index:
        kt = self.measured_signal.div(G_clear, axis=0)

        # Apply thresholds:
        kt = np.maximum(kt, 0)
        kt = np.minimum(kt, max_clearness_index)

        return (kt)

    def day_ahead_smart_persistance_forecast(self, bidding_limit,
                                             G_0=1360, max_clearness_index=2.0, model='clearness') -> pd.Series:
        """
        Get smart persistance day-ahead forecast.

        :param G_0: float
            the extra terrestrial radiation received by the Earth from the Sun
        :param max_clearness_index: float
            maximum value for clearness index
        :param model: str
            clearness or clear-sky model (ineichen, haurwitz, simplified_solis)
        :return: G_forecast : pandas.Series

        """

        # Read raw-data time-zone:
        raw_tz = str(self.measured_signal.index.tz)

        try:
            if model == 'clearness':
                # Get clearness:
                G_clear = self.get_clearness(G_0)

                # Calculate clearness index:
                kt = self.get_clearsky_index(G_clear, max_clearness_index)

            else:
                # Calculate clear-sky global horizontal irradiance (GHI):
                G_clear = self.location.get_clearsky(self.measured_signal.index, model=model)
                G_clear = G_clear['ghi']

                # Calculate clear-sky index:
                kt = self.get_clearsky_index(G_clear, max_clearness_index)

        except ValueError:
            raise

        # Convert to bidding time-zone:
        G_clear = G_clear.tz_convert(bidding_limit.tzinfo)
        kt = kt.tz_convert(bidding_limit.tzinfo)
        measured_signal = self.measured_signal.copy().tz_convert(bidding_limit.tzinfo)

        # Account for binning of data:
        G_clear = G_clear.resample(self.f_i, closed='left', label=self.binning).median()
        kt = kt.resample(self.f_i, closed='left', label=self.binning).median()
        measured_signal = measured_signal.resample(self.f_i, closed='left', label=self.binning).median()

        # Get solar position:
        solar_position = self.location.get_solarposition(self.measured_signal.index)

        # Get solar noon:
        solar_noon_filter = solar_position.groupby(pd.Grouper(freq='D'))['zenith'].idxmin()
        solar_noon_hour = int(np.median(solar_position['zenith'].loc[solar_noon_filter].index.hour))
        solar_noon_minute = int(np.median(solar_position['zenith'].loc[solar_noon_filter].index.minute))

        # Calculate daily mid-day median of clear-sky index:
        kt_night = kt.loc[(kt.index.time >= dt.time(solar_noon_hour - 2, solar_noon_minute)) &
                          (kt.index.time <= dt.time(solar_noon_hour + 2, solar_noon_minute))].resample('d').median()

        # Shift daily mid-day median of clear-sky index for night time-stamps:
        kt_night = kt_night.asfreq(self.f_i).ffill().shift(periods=2, freq='d')

        # Generate forecast using daily mid-day median of clear-sky index for night time-stamps:
        G_da_forecast_night = kt_night.mul(G_clear, axis=0)

        # Generate persistance forecast:
        morning = measured_signal.shift(periods=1, freq='d')
        afternoon = measured_signal.shift(periods=2, freq='d')
        G_da_forecast = morning.copy()

        # Ensure no duplicates. NB! This approach is not entirely correct the day where clocks are adjusted one hour ahead!
        # 11:00 is taken from 2 days ahead, but it should be taken from one day ahead...
        G_da_forecast[G_da_forecast.index.time > bidding_limit] = np.nan
        afternoon = afternoon.loc[G_da_forecast.index[0]:G_da_forecast.index[-1]][G_da_forecast.isna()]
        G_da_forecast[G_da_forecast.isna()] = afternoon

        # Ensure same timeframe:
        G_da_forecast = G_da_forecast.loc[self.measured_signal.index[0]:self.measured_signal.index[-1]]
        G_da_forecast_night = G_da_forecast_night.loc[self.measured_signal.index[0]:self.measured_signal.index[-1]]

        # Set missing values to nan. I.e., this assumes that all periods of missing data is scheduled!
        G_da_forecast[kt.isna()] = np.nan

        # Convert back to original time-zone:
        G_da_forecast = G_da_forecast.tz_convert(raw_tz)

        # Identify periods with measurements, but with missing values in forecast:
        missing_forecast = (self.measured_signal.resample(self.f_i, closed='left',
                                                          label=self.binning).median().notna()) & (G_da_forecast.isna())

        # Use daily mid-day median for day before to forecast periods after periods of missing data (night, etc.):
        G_da_forecast[missing_forecast] = G_da_forecast_night[missing_forecast]

        return G_da_forecast

    def smart_persistance(self, G_0=1360, max_clearness_index=2.0, model='clearness') -> pd.Series:
        """
        Get smart persistance forecast.

        :param G_0: float
            the extra terrestrial radiation received by the Earth from the Sun
        :param max_clearness_index: float
            maximum value for clearness index
        :param model: str
            clearness or clear-sky model (ineichen, haurwitz, simplified_solis)
        :return: G_forecast : pandas.Series

        """

        try:
            if model == 'clearness':
                # Get clearness:
                G_clear = self.get_clearness(G_0)

                # Calculate clearness index:
                kt = self.get_clearsky_index(G_clear, max_clearness_index)

            else:
                # Calculate clear-sky global horizontal irradiance (GHI):
                G_clear = self.location.get_clearsky(self.measured_signal.index, model=model)
                G_clear = G_clear['ghi']

                # Calculate clear-sky index:
                kt = self.get_clearsky_index(G_clear, max_clearness_index)

        except ValueError:
            raise

        # Account for binning of data:
        G_clear = G_clear.resample(self.f_i, closed='left', label=self.binning).median()
        kt = kt.resample(self.f_i, closed='left', label=self.binning).median()

        # Get solar position:
        solar_position = self.location.get_solarposition(self.measured_signal.index)

        # Get solar noon:
        solar_noon_filter = solar_position.groupby(pd.Grouper(freq='D'))['zenith'].idxmin()
        solar_noon_hour = int(np.median(solar_position['zenith'].loc[solar_noon_filter].index.hour))
        solar_noon_minute = int(np.median(solar_position['zenith'].loc[solar_noon_filter].index.minute))

        # Calculate daily mid-day median of clear-sky index:
        kt_night = kt.loc[(kt.index.time >= dt.time(solar_noon_hour - 2, solar_noon_minute)) &
                          (kt.index.time <= dt.time(solar_noon_hour + 2, solar_noon_minute))].resample('d').median()

        # Shift daily mid-day median of clear-sky index for night time-stamps:
        kt_night = kt_night.asfreq(self.f_i).ffill().shift(periods=1, freq='d')[:-1]  # TODO: Improve this

        # Generate forecast using daily mid-day median of clear-sky index for night time-stamps:
        G_forecast_night = kt_night.mul(G_clear,
                                        axis=0)  # .shift(self.s_h_periods + self.f_h_periods).mul(G_clear, axis=0)

        # Get smart persistance forecast:
        G_forecast = kt.shift(self.s_h_periods + self.f_h_periods).mul(G_clear, axis=0)

        # Set missing values to nan. I.e., this assumes that all periods of missing data is scheduled!
        G_forecast[kt.isna()] = np.nan

        # As a consequence, no forecast can be issued for the first timestamps after a scheduled down-period.
        G_forecast[kt.shift(self.s_h_periods + self.f_h_periods).isna()] = np.nan

        # Identify periods with measurements, but with missing values in forecast:
        missing_forecast = (self.measured_signal.resample(self.f_i, closed='left',
                                                          label=self.binning).median().notna()) & (G_forecast.isna())

        # Use daily mid-day median for day before to forecast periods after periods of missing data (night, etc.):
        G_forecast[missing_forecast] = G_forecast_night[missing_forecast]

        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        # G_clear.plot(ax=ax1, marker='o', linestyle='None', markersize=2)
        ##self.measured_signal.plot(ax=ax1, marker='o', linestyle='None', markersize=2)
        # G_forecast.plot(ax=ax1, marker='o', linestyle='None', markersize=2)
        # kt.plot(ax=ax2, marker='o', linestyle='None', markersize=2)
        # plt.subplots_adjust(hspace=0.05)
        # plt.show()

        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        # G_clear.plot(ax=ax1, marker='o', linestyle='None', markersize=2, label='Clear')
        # self.measured_signal.resample(self.f_i, closed='left', label=self.binning).median().plot(ax=ax1, marker='o', linestyle='None', markersize=2, label='Measured')
        ##G_forecast.plot(ax=ax1, marker='o', linestyle='None', markersize=2)
        # kt.plot(ax=ax2, marker='o', linestyle='None', markersize=2)
        # ax1.set_ylabel(r'$G$ [W/m$^2$]', fontsize=15)
        # ax1.legend(loc='upper center', ncol=2, fontsize=10)
        ##ax2.set_xlim(pd.to_datetime('2020-06-15'), pd.to_datetime('2020-06-16 23:00:00+00:00'))
        # ax2.set_ylabel(r'$k_t$ [$\varnothing$]', fontsize=15)
        # plt.subplots_adjust(hspace=0.05)
        # plt.show()

        return G_forecast

    def get_vector_smart_persistance(self, max_horizon, horizon_freq):
        """
        getting a smart persistence for multiple horizons at once

        UNTESTED

        Parameters
        ----------
        max_horizon : int : max horizon in Seconds
        horizon_freq : int . horizon frequency in Seconds

        Returns
        -------

        """
        n_horizons = int(max_horizon / horizon_freq)

        # first_horizon
        self.f_h = f'{horizon_freq}S'
        self.f_i = f'{horizon_freq}S'
        sp = self.smart_persistance()

        # Rest of horizons
        for step in range(2, n_horizons):
            self.f_h = f'{step * horizon_freq}S'
            sp_i = self.smart_persistance()
            sp = pd.concat([sp, sp_i], axis=1)

        sp.columns = range(n_horizons)

        return sp



    @staticmethod
    def get_periods(f_h: str, f_i: str) -> int:
        """
        Convert forecast horizon f_h to number of periods in units of the forecast issuing frequency f_i.

        :param f_h: str
        :param f_i: str
            Forecast horizon, e.g., '1T'
        :return: periods for forecast : int
        """

        # Get timedelta of frequency of data:
        # freq = pd.infer_freq(measured_signal.index)
        # freq = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))

        # Get timedelta of forecast issuing frequency:
        f_i = pd.Timedelta(f_i)

        # Get timedelta of forecast horizon:
        f_h = pd.Timedelta(f_h)

        # TODO: Assert that f_h.total_seconds() / freq.total_seconds() is an integer!

        return int(f_h.total_seconds() / f_i.total_seconds())


if __name__ == '__main__':
    ghi = pd.Series(index=pd.date_range(start='2020-05-01', end='2020-05-02', freq='1 s'),
                    data=1)
    freq = pd.infer_freq(ghi.index)
    print(PVForecast.get_periods(ghi, '1T'))

    # Generate smp id forecasts:
    no_1_smp_id = PVForecast(ghi_df['NO1'].copy().shift(periods=450, freq='1S'), binning='left',
                             f_h='60T', f_i='15T', s_h='60T', location=locations['NO1']).smart_persistance(
        model='ineichen')



### own stuff
import torch
class Persistence_Model():
    def __init__(self,normalizer,params):

        self.data = pd.read_pickle('models/persistence.pkl')
        self.min = normalizer.min_target
        self.max = normalizer.max_target
        self.data_normalized = (self.data-self.min)/(self.max-self.min)
        self.params = params

        self.target_summary = params["target_summary"]
    
    def forecast_raw(self,time):
        
        persistence = np.zeros((time.shape[0],self.params['horizon_size']))
        for idx,instance in enumerate(time):
            temp_pers =self.data.loc[instance].rolling(window=self.target_summary, min_periods=1).mean()[self.target_summary-1::self.target_summary]
            persistence[idx] = temp_pers.values

        return persistence#self.data.loc[time]
    
    def forecast(self,time):
        persistence = np.zeros((time.shape[0],self.params['horizon_size']))
        for idx,instance in enumerate(time):
            temp_pers = self.data_normalized.loc[instance].rolling(window=self.target_summary, min_periods=1).mean()[self.target_summary-1::self.target_summary]
            persistence[idx] = temp_pers.values
        return persistence
    
    def evaluate(self,forecast,actual):
        mae =  np.mean(np.abs(forecast-actual))
        tau = 0.5
        fake_pinball = np.mean(np.maximum(tau*(actual-forecast), (tau-1)*(actual-forecast)))
        return mae,fake_pinball