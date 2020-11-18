import numpy as np
import pandas as pd
from labml import monit, lab, logger

from labml_samples.pytorch.stocks import CandleIdx


def parse(df: pd.DataFrame):
    time = np.zeros((len(df)), dtype=int)
    date = []
    for i in monit.iterate("Calculate time", len(df)):
        hour = int(df['Time'][i][0:2])
        mint = int(df['Time'][i][3:5])
        time[i] = hour * 60 + mint
        mon = df['Date'][i][0:2]
        day = df['Date'][i][3:5]
        year = df['Date'][i][6:10]
        date.append(f"{year}-{mon}-{day}")

    time = time - 570
    df['Minute'] = time
    df['Date'] = date

    return df


def filter_premarket(df: pd.DataFrame):
    df = df[0 <= df['Minute']]
    df = df[df['Minute'] < 390]
    df = df.reset_index()
    df = df[['Date', 'Time', 'Minute', 'Open', 'High', 'Low', 'Close', 'Volume']]

    return df


def to_daily_packets(df: pd.DataFrame):
    volume = np.array(df['Volume'], dtype=float)
    time = np.array(df["Minute"])
    candles = np.zeros((len(df), 6), dtype=float)
    candles[:, CandleIdx.high] = np.array(df['High'])
    candles[:, CandleIdx.low] = np.array(df['Low'])
    candles[:, CandleIdx.open] = np.array(df['Open'])
    candles[:, CandleIdx.close] = np.array(df['Close'])
    candles[:, CandleIdx.volume] = volume
    candles[:, 5] = time

    dates = []
    packets = []
    current_day = None
    packet = None
    for i in monit.iterate("To daily packets", len(df)):
        d = df['Date'][i]
        if d != current_day:
            if current_day is not None:
                dates.append(current_day)
                packets.append(packet)
            current_day = d
            packet = np.zeros((390, 5), dtype=float)
        t = time[i]
        if 0 <= t < 390:
            packet[t, :] = candles[i, 0:5]
    if current_day is not None:
        dates.append(current_day)
        packets.append(packet)

    return np.array(dates), np.array(packets)


def fill_empty_minutes_in_packet(packet: np.ndarray):
    last_idx = None
    for i in range(390):
        if np.sum(packet[i] == 0) > 0:
            if last_idx is not None:
                packet[i, 0:4] = packet[last_idx, 0:4]
        if np.sum(packet[i] > 0) == len(packet[i]):
            if last_idx is None:
                for j in range(i):
                    packet[j, 0:4] = packet[i, 0:4]
            last_idx = i


def fill_empty_minutes_in_packets(packets: np.ndarray):
    for i in monit.iterate("Fill empty minutes", packets.shape[0]):
        fill_empty_minutes_in_packet(packets[i])


def to_numpy(df: pd.DataFrame):
    dates, packets = to_daily_packets(df)
    empty_mins = np.sum(packets[:, :, 4] == 0)
    filled_mins = np.sum(packets[:, :, 4] > 0)
    empty_mins_high_activity = np.sum(packets[:, :, 4] == 0)
    fill_empty_minutes_in_packets(packets)
    zero_price = np.sum(packets[:, :, 0:4] == 0)
    zero_volume = np.sum(packets[:, :, 4] == 0)
    packets[:, :, 4] = np.maximum(packets[:, :, 4], 1)

    logger.inspect(empty_mins=empty_mins,
                   filled_mins=filled_mins,
                   empty_mins_high_activity=empty_mins_high_activity,
                   zero_price=zero_price,
                   zero_volume=zero_volume)

    return dates, packets


def build_cache():
    data_path = lab.get_data_path()

    with monit.section("Read data"):
        df = pd.read_csv(str(data_path / "IBM_unadjusted.txt"))
    df = parse(df)
    with monit.section("Filter pre-market data"):
        df = filter_premarket(df)

    with monit.section("To numpy"):
        dates, packets = to_numpy(df)

    with monit.section("Save"):
        np.save(str(data_path / "packets.npy"), packets)
        np.save(str(data_path / "dates.npy"), dates)


if __name__ == '__main__':
    build_cache()
