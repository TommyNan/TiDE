import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import EasterMonday
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.holiday import Holiday
from pandas.tseries.holiday import SU
from pandas.tseries.holiday import TH
from pandas.tseries.holiday import USColumbusDay
from pandas.tseries.holiday import USLaborDay
from pandas.tseries.holiday import USMartinLutherKingJr
from pandas.tseries.holiday import USMemorialDay
from pandas.tseries.holiday import USPresidentsDay
from pandas.tseries.holiday import USThanksgivingDay
from pandas.tseries.offsets import DateOffset
from pandas.tseries.offsets import Day
from pandas.tseries.offsets import Easter
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second Of Minute encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of Week encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # return (index.isocalendar().week - 1) / 52.0 - 0.5
        return (index.week - 1) / 52.0 - 0.5

class DistanceToHoliday(TimeFeature):
    """Distance"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        index = index.to_series()
        hol_variates = np.vstack([index.apply(_distance_to_holiday(h)).values for h in tqdm(HOLIDAYS)])
        return StandardScaler().fit_transform(hol_variates)

def _distance_to_holiday(holiday):
  """Return distance to given holiday."""

  def _distance_to_day(index):
    holiday_date = holiday.dates(
        index - pd.Timedelta(days=200),
        index + pd.Timedelta(days=200),
    )
    assert (len(holiday_date) != 0), f"No closest holiday for the date index {index} found."
    # It sometimes returns two dates if it is exactly half a year after the
    # holiday. In this case, the smaller distance (182 days) is returned.
    return (index - holiday_date[0]).days
  return _distance_to_day


EasterSunday = Holiday(
    "Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)]
)
NewYearsDay = Holiday("New Years Day", month=1, day=1)
SuperBowl = Holiday(
    "Superbowl", month=2, day=1, offset=DateOffset(weekday=SU(1))
)
MothersDay = Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=SU(2))
)
IndependenceDay = Holiday("Independence Day", month=7, day=4)
ChristmasEve = Holiday("Christmas", month=12, day=24)
ChristmasDay = Holiday("Christmas", month=12, day=25)
NewYearsEve = Holiday("New Years Eve", month=12, day=31)
BlackFriday = Holiday(
    "Black Friday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(1)],
)
CyberMonday = Holiday(
    "Cyber Monday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(4)],
)

HOLIDAYS = [
    EasterMonday,
    GoodFriday,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    EasterSunday,
    NewYearsDay,
    SuperBowl,
    MothersDay,
    IndependenceDay,
    ChristmasEve,
    ChristmasDay,
    NewYearsEve,
    BlackFriday,
    CyberMonday,
]


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        # offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        # offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [MinuteOfHour, HourOfDay, DayOfMonth, DayOfWeek, DayOfYear, MonthOfYear, WeekOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        # offsets.Second: [
        #     SecondOfMinute,
        #     MinuteOfHour,
        #     HourOfDay,
        #     DayOfWeek,
        #     DayOfMonth,
        #     DayOfYear,
        # ],
        offsets.Second: [MinuteOfHour, HourOfDay, DayOfMonth, DayOfWeek, DayOfYear, MonthOfYear, WeekOfYear, DistanceToHoliday],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])


class Dataset_ETT_hour(Dataset):
    def __init__(self, dataset_filename, mode='train', in_label_out_lens=None, task_type='M2M', target='OT',
                 scale=True, timeenc=0, freq='h'):
        # in_label_out_lens [in_lens, label_lens, out_lens]
        self.in_lens, self.label_lens, self.out_lens = in_label_out_lens[0], in_label_out_lens[1], in_label_out_lens[2]

        assert mode in ['train', 'val', 'test']
        mode_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = mode_map[mode]

        self.task_type = task_type
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.dataset_filename = dataset_filename
        self.__read_data__()

    def __read_data__(self):
        # shape [num_examples, timestamp+features]
        df_raw = pd.read_csv(self.dataset_filename)
        # num_samples = df_raw.shape[0]
        # num_train, num_test = int(num_samples * 0.6), int(num_samples * 0.2)
        # num_valid = num_samples - num_train - num_test
        # train_val_test_lower_index = [0, num_train - self.in_lens, num_samples - num_test - self.in_lens]
        train_val_test_lower_index = [0, 12 * 30 * 24 - self.in_lens, 12 * 30 * 24 + 4 * 30 * 24 - self.in_lens]
        train_val_test_upper_index = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        data_lower_index = train_val_test_lower_index[self.set_type]
        data_upper_index = train_val_test_upper_index[self.set_type]

        if self.task_type == 'M2M' or self.task_type == 'M2S':
            cols_name = df_raw.columns[1:]
            df_data = df_raw[cols_name]
        elif self.task_type == 'S2S':
            df_data = df_raw[[self.target]]

        # data - Standardization return: df_data_std
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[train_val_test_lower_index[0]:train_val_test_upper_index[0]]
            self.scaler.fit(train_data.values)
            df_data_std = self.scaler.transform(df_data.values)
        else:
            df_data_std = df_data.values

        # Timestamp generation return: df_data_timestamp
        df_timestamp = df_raw[['date']][data_lower_index:data_upper_index]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)

        if self.timeenc == 0:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month, 1)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day, 1)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday(), 1)
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
            df_data_timestamp = df_timestamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            df_data_timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            df_data_timestamp = df_data_timestamp.transpose(1, 0)

        self.data_x = df_data_std[data_lower_index:data_upper_index]
        self.data_y = df_data_std[data_lower_index:data_upper_index]
        self.data_timestamp = df_data_timestamp

    def __getitem__(self, index):
        encoder_begin = index
        encoder_end = encoder_begin + self.in_lens
        decoder_begin = encoder_end - self.label_lens
        decoder_end = decoder_begin + self.label_lens + self.out_lens

        encoder_x = self.data_x[encoder_begin:encoder_end]
        decoder_y = self.data_y[decoder_begin:decoder_end]
        encoder_x_timestamp = self.data_timestamp[encoder_begin:encoder_end]
        decoder_y_timestamp = self.data_timestamp[decoder_begin:decoder_end]

        return encoder_x, decoder_y, encoder_x_timestamp, decoder_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.in_lens - self.out_lens + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, dataset_filename, mode='train', in_label_out_lens=None, task_type='M2M', target='OT',
                 scale=True, timeenc=0, freq='t'):
        # in_label_out_lens [in_lens, label_lens, out_lens]
        self.in_lens, self.label_lens, self.out_lens = in_label_out_lens[0], in_label_out_lens[1], in_label_out_lens[2]

        assert mode in ['train', 'val', 'test']
        mode_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = mode_map[mode]

        self.task_type = task_type
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.dataset_filename = dataset_filename
        self.__read_data__()

    def __read_data__(self):
        # shape [num_examples, timestamp+features]
        df_raw = pd.read_csv(self.dataset_filename)

        # num_samples = df_raw.shape[0]
        # num_train, num_test = int(num_samples * 0.6), int(num_samples * 0.2)
        # num_valid = num_samples - num_train - num_test
        # train_val_test_lower_index = [0, num_train - self.in_lens, num_samples - num_test - self.in_lens]
        # train_val_test_upper_index = [num_train, num_train + num_valid, num_samples]
        train_val_test_lower_index = [0, 12 * 30 * 24 * 4 - self.in_lens, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.in_lens]
        train_val_test_upper_index = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        data_lower_index = train_val_test_lower_index[self.set_type]
        data_upper_index = train_val_test_upper_index[self.set_type]

        if self.task_type == 'M2M' or self.task_type == 'M2S':
            cols_name = df_raw.columns[1:]
            df_data = df_raw[cols_name]
        elif self.task_type == 'S2S':
            df_data = df_raw[[self.target]]

        # data - Standardization return: df_data_std
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[train_val_test_lower_index[0]:train_val_test_upper_index[0]]
            self.scaler.fit(train_data.values)
            df_data_std = self.scaler.transform(df_data.values)
        else:
            df_data_std = df_data.values

        # Timestamp generation return: df_data_timestamp
        df_timestamp = df_raw[['date']][data_lower_index:data_upper_index]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)

        if self.timeenc == 0:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month, 1)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day, 1)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday(), 1)
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
            df_timestamp['minute'] = df_timestamp.date.apply(lambda row: row.minute, 1)
            df_timestamp['minute'] = df_timestamp.minute.map(lambda x: x // 15)
            df_data_timestamp = df_timestamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # shape [num_features, num_examples] -> [num_examples, num_features]
            df_data_timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            df_data_timestamp = df_data_timestamp.transpose(1, 0)

        self.data_x = df_data_std[data_lower_index:data_upper_index]
        self.data_y = df_data_std[data_lower_index:data_upper_index]
        self.data_timestamp = df_data_timestamp

    def __getitem__(self, index):
        encoder_begin = index
        encoder_end = encoder_begin + self.in_lens
        decoder_begin = encoder_end - self.label_lens
        decoder_end = decoder_begin + self.label_lens + self.out_lens

        encoder_x = self.data_x[encoder_begin:encoder_end]
        decoder_y = self.data_y[decoder_begin:decoder_end]
        encoder_x_timestamp = self.data_timestamp[encoder_begin:encoder_end]
        decoder_y_timestamp = self.data_timestamp[decoder_begin:decoder_end]

        return encoder_x, decoder_y, encoder_x_timestamp, decoder_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.in_lens - self.out_lens + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, dataset_filename, mode='train', in_label_out_lens=None, task_type='M2M', target='OT',
                 scale=True, timeenc=0, freq='h'):
        # in_label_out_lens [in_lens, label_lens, out_lens]
        self.in_lens, self.label_lens, self.out_lens = in_label_out_lens[0], in_label_out_lens[1], in_label_out_lens[2]

        assert mode in ['train', 'val', 'test']
        mode_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = mode_map[mode]

        self.task_type = task_type
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.dataset_filename = dataset_filename
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # shape [num_examples, timestamp+features]
        df_raw = pd.read_csv(self.dataset_filename)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_samples = df_raw.shape[0]
        num_train, num_test = int(num_samples * 0.7), int(num_samples * 0.2)
        num_valid = num_samples - num_train - num_test
        train_val_test_lower_index = [0, num_train - self.in_lens, num_samples - num_test - self.in_lens]
        train_val_test_upper_index = [num_train, num_train + num_valid, num_samples]
        data_lower_index = train_val_test_lower_index[self.set_type]
        data_upper_index = train_val_test_upper_index[self.set_type]

        if self.task_type == 'M2M' or self.task_type == 'M2S':
            cols_name = df_raw.columns[1:]
            df_data = df_raw[cols_name]
        elif self.task_type == 'S2S':
            df_data = df_raw[[self.target]]

        # data - Standardization return: df_data_std
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[train_val_test_lower_index[0]:train_val_test_upper_index[0]]
            self.scaler.fit(train_data.values)
            df_data_std = self.scaler.transform(df_data.values)
        else:
            df_data_std = df_data.values

        # Timestamp generation return: df_data_timestamp
        df_timestamp = df_raw[['date']][data_lower_index:data_upper_index]
        df_timestamp['date'] = pd.to_datetime(df_timestamp.date)

        if self.timeenc == 0:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month, 1)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day, 1)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday(), 1)
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
            df_data_timestamp = df_timestamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            # df_data_timestamp.shape [num_features, in_lens]->[in_lens, num_features]
            df_data_timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            df_data_timestamp = df_data_timestamp.transpose(1, 0)

        self.data_x = df_data_std[data_lower_index:data_upper_index]
        self.data_y = df_data_std[data_lower_index:data_upper_index]
        self.data_timestamp = df_data_timestamp

    def __getitem__(self, index):
        encoder_begin = index
        encoder_end = encoder_begin + self.in_lens
        decoder_begin = encoder_end - self.label_lens
        decoder_end = decoder_begin + self.label_lens + self.out_lens

        encoder_x = self.data_x[encoder_begin:encoder_end]
        decoder_y = self.data_y[decoder_begin:decoder_end]
        encoder_x_timestamp = self.data_timestamp[encoder_begin:encoder_end]
        decoder_y_timestamp = self.data_timestamp[decoder_begin:decoder_end]

        return encoder_x, decoder_y, encoder_x_timestamp, decoder_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.in_lens - self.out_lens + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, dataset_filename, mode='pred', in_label_out_lens=None, task_type='S2S', target='OT',
                 scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # in_label_out_lens [in_lens, label_lens, out_lens]
        self.in_lens, self.label_lens, self.out_lens = in_label_out_lens[0], in_label_out_lens[1], in_label_out_lens[2]

        assert mode in ['pred']

        self.task_type = task_type
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.dataset_filename = dataset_filename
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # shape [num_examples, timestamp+features]
        df_raw = pd.read_csv(self.dataset_filename)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')

        df_raw = df_raw[['date'] + cols + [self.target]]

        num_samples = df_raw.shape[0]
        data_lower_index = num_samples - self.in_lens
        data_upper_index = num_samples

        if self.task_type == 'M2M' or self.task_type == 'M2S':
            cols_name = df_raw.columns[1:]
            df_data = df_raw[cols_name]
        elif self.task_type == 'S2S':
            df_data = df_raw[[self.target]]

        # data - Standardization return: df_data_std
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            df_data_std = self.scaler.transform(df_data.values)
        else:
            df_data_std = df_data.values

        # Timestamp generation return: df_data_timestamp
        tmp_timestamp = df_raw[['date']][data_lower_index:data_upper_index]
        tmp_timestamp['date'] = pd.to_datetime(tmp_timestamp.date)
        pred_dates = pd.date_range(tmp_timestamp.date.values[-1], periods=self.out_lens + 1, freq=self.freq)

        df_timestamp = pd.DataFrame(columns=['date'])
        df_timestamp.date = list(tmp_timestamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])

        if self.timeenc == 0:
            df_timestamp['month'] = df_timestamp.date.apply(lambda row: row.month, 1)
            df_timestamp['day'] = df_timestamp.date.apply(lambda row: row.day, 1)
            df_timestamp['weekday'] = df_timestamp.date.apply(lambda row: row.weekday(), 1)
            df_timestamp['hour'] = df_timestamp.date.apply(lambda row: row.hour, 1)
            df_timestamp['minute'] = df_timestamp.date.apply(lambda row: row.minute, 1)
            df_timestamp['minute'] = df_timestamp.minute.map(lambda x: x // 15)
            df_data_timestamp = df_timestamp.drop(['date'], axis=1).values

        elif self.timeenc == 1:
            df_data_timestamp = time_features(pd.to_datetime(df_timestamp['date'].values), freq=self.freq)
            df_data_timestamp = df_data_timestamp.transpose(1, 0)

        self.data_x = df_data_std[data_lower_index:data_upper_index]
        if self.inverse:
            self.data_y = df_data.values[data_lower_index:data_upper_index]
        else:
            self.data_y = df_data_std[data_lower_index:data_upper_index]
        self.data_timestamp = df_data_timestamp

    def __getitem__(self, index):
        encoder_begin = index
        encoder_end = encoder_begin + self.in_lens
        decoder_begin = encoder_end - self.label_lens
        decoder_end = decoder_begin + self.label_lens + self.out_lens

        encoder_x = self.data_x[encoder_begin:encoder_end]
        if self.inverse:
            decoder_y = self.data_x[decoder_begin:decoder_begin + self.label_lens]
        else:
            decoder_y = self.data_y[decoder_begin:decoder_begin + self.label_lens]
        encoder_x_timestamp = self.data_timestamp[encoder_begin:encoder_end]
        decoder_y_timestamp = self.data_timestamp[decoder_begin:decoder_end]

        return encoder_x, decoder_y, encoder_x_timestamp, decoder_y_timestamp

    def __len__(self):
        return len(self.data_x) - self.in_lens + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

