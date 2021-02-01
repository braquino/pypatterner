import datetime
from time import time
from typing import Tuple
from random import randrange, random, choice, shuffle
import pandas as pd

def random_date(patter: str) -> Tuple[str, str]:
    """
    :param patter: a date pattern, example: 'YYYY-MM-DD HH:mm:ss'
    :return: a tuple random date converted to string and your pattern
    """
    ms_random = randrange(0, 1_700_000_000_000) / 1000
    date = datetime.datetime.fromtimestamp(ms_random)
    try:
        formatted_date = date.strftime(patter)
    except Exception as e:
        raise Exception(str(e) + patter)
    return formatted_date, patter

def random_pattern(p_hour_min=0.3, p_sec=0.5, p_ms=0.3, p_month_str=0.2, p_year_4=0.8, p_zero_pad=0.85,
                   sep_day=("/", "-", ", ", " ", ""),
                   sep_time=(" ", ", ", "T", ":")):
    sep_day_choice: str = choice(sep_day)
    sep_time_choice: str = choice(sep_time)
    month_str = "%b" if random() < p_month_str else ("%m" if random() < p_zero_pad else "%#m")
    year_str = "%Y" if random() < p_year_4 else "%y"
    day_str = "%d" if random() < p_zero_pad else "%#d"
    y_m_d = [year_str, month_str, day_str]
    shuffle(y_m_d)
    full_patter: str = sep_day_choice.join(y_m_d)
    if random() < p_hour_min:
        H_M = ["%H", "%M"] if random() < p_zero_pad else ["%#H", "%#M"]
        full_patter += (sep_time_choice + ":".join(H_M))
        if random() < p_sec:
            S = "%S" if random() < p_zero_pad else "%#S"
            full_patter += (":" + S)
            if random() < p_ms:
                full_patter += (":" + "%f")
    return full_patter

def create_train_file(num_patterns=2000, num_dates_per_pattern=200):
    train_data_path = "../../train_data/random_train_{}.csv".format(int(time()))
    train_list = []
    for i in range(num_patterns):
        date_pattern = random_pattern()
        for j in range(num_dates_per_pattern):
            train_list.append(random_date(date_pattern))
    pd.DataFrame(train_list, columns=["date", "pattern"]).to_csv(train_data_path, index=False)
    return train_data_path

if __name__ == "__main__":
    print(create_train_file())
