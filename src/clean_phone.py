import re
import numpy as np
import pandas as pd
from operator import itemgetter

from typing import Any, Union
##import dask.dataframe as dd

NULL_VALUES = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    None,
}

NEARBYKEYS = {
    "a": ["q", "w", "s", "x", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "s", "d", "r"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "j", "k", "o"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k", "l"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "k", "l", "p"],
    "p": ["o", "l"],
    "q": ["w", "a", "s"],
    "r": ["e", "d", "f", "t"],
    "s": ["w", "e", "d", "x", "z", "a"],
    "t": ["r", "f", "g", "y"],
    "u": ["y", "h", "j", "i"],
    "v": ["c", "f", "g", "v", "b"],
    "w": ["q", "a", "s", "e"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "g", "h", "u"],
    "z": ["a", "s", "x"],
    " ": ["c", "v", "b", "n", "m"],
}


PHONE_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[+(]?(?P<country>(9[976]\d|8[987530]\d|6[987]\d|5[90]\d|42\d|3[875]\d|
                        2[98654321]\d|9[8543210]|8[6421]|6[6543210]|5[87654321]|
                        4[987654310]|3[9643210]|2[70]|7|1))[)\/]?)?
    [-. (]*
    (?P<area>\d{3})?
    [-. )\/]*
    (?:(?P<office>\d{3})
    [-. \/]*
    (?P<station>\d{4})|
    (?P<letters>[0-9A-Z-. \/]{7,13}?))
    (?:[ \t]*(?:\#|x[.:]?|[Ee]xt[.:]?|[Ee]xtension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)

US_PHONE_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[+(]?(?P<country>1)[)\/]?)?
    [-. (]*
    (?P<area>\d{3})?
    [-. )\/]*
    (?:(?P<office>\d{3})
    [-. \/]*
    (?P<station>\d{4})|
    (?P<letters>[0-9A-Z-. \/]{7,13}?))
    (?:[ \t]*(?:\#|x[.:]?|[Ee]xt[.:]?|[Ee]xtension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)

ALPHA_NUM_MAP = {
    "A": "2",
    "B": "2",
    "C": "2",
    "D": "3",
    "E": "3",
    "F": "3",
    "G": "4",
    "H": "4",
    "I": "4",
    "J": "5",
    "K": "5",
    "L": "5",
    "M": "6",
    "N": "6",
    "O": "6",
    "P": "7",
    "Q": "7",
    "R": "7",
    "S": "7",
    "T": "8",
    "U": "8",
    "V": "8",
    "W": "9",
    "X": "9",
    "Y": "9",
    "Z": "9",
}
def split_phone(mch, clean) -> Any:
    if mch.group("letters"):
        # Check that there are 7 alphanumeric characters present
        letters = re.sub(r"\W+", "", mch.group("letters"))
        if len(letters) != 7:
            return (None,) * 5 + ("unknown",) if clean else False
        # Convert letters to numbers
        numlist = [ALPHA_NUM_MAP[char] if char.isalpha() else char for char in letters]
        numbers = "".join(numlist)
    # Components for phone number
    country_code = mch.group("country")
    area_code = mch.group("area")
    office_code = numbers[:3] if mch.group("letters") else mch.group("office")
    station_code = numbers[3:] if mch.group("letters") else mch.group("station")
    ext_num = mch.group("ext")

    return (
        (country_code, area_code, office_code, station_code, ext_num, "success") if clean else True
    )
def _check_phone(phone: Any, clean: bool) -> Any:
    if phone in NULL_VALUES:
        return (None,) * 5 + ("null",) if clean else False

    mch = re.match(US_PHONE_PATTERN, re.sub(r"''", r'"', str(phone)))
    # Check if the value was able to be parsed

    if not mch:
        mch = re.match(PHONE_PATTERN, re.sub(r"''", r'"', str(phone)))
        if not mch:
            return (None,) * 5 + ("unknown",) if clean else False
        else:
            return split_phone(mch, clean)
    if mch.group("country") and not mch.group("area"):
        return (None,) * 5 + ("unknown",) if clean else False

    return split_phone(mch, clean)


def validate_phone(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    if isinstance(x, pd.Series):
        return x.apply(_check_phone, clean=False)
    return _check_phone(x, False)

def _format_phone(
    phone: Any, output_format: str, fix_missing: str, split: bool, errors: str
) -> Any:
    country_code, area_code, office_code, station_code, ext_num, status = _check_phone(phone, True)
    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0) if split else (np.nan, 0)

    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {phone}")
        result = phone if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, np.nan, 1) if split else (result, 1)

    if split:
        missing_code = "1" if fix_missing == "auto" and area_code else np.nan
        country_code = country_code if country_code else missing_code
        area_code = area_code if area_code else np.nan
        ext_num = ext_num if ext_num else np.nan
        return country_code, area_code, office_code, station_code, ext_num, 2

    if output_format == "nanp":  # NPA-NXX-XXXX
        area_code = f"{area_code}-" if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"
    elif output_format == "e164":  # +NPANXXXXXX
        print(country_code)
        if country_code is None and area_code:
            country_code = "+1"
        else:
            country_code = "+" + country_code if area_code else ""
        area_code = area_code if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{country_code}{area_code}{office_code}{station_code}{ext_num}"
    elif output_format == "national":  # (NPA) NXX-XXXX
        area_code = f"({area_code}) " if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"

    return result, 2 if phone != result else 3

def clean_phone(
    df: Union[pd.DataFrame, pd.DataFrame],
    column: str,
    output_format: str = "nanp",
    fix_missing: str = "empty",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    if output_format not in {"nanp", "e164", "national"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "nanp", "e164" or "national"'
        )

    if fix_missing not in {"auto", "empty"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "auto" or "empty"'
        )

    # convert to dask
    #df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].apply(
        lambda x: _format_phone(x, output_format, fix_missing, split, errors)
    )

    if split:
        # For some reason the meta data for the last 3 components needs to be
        # set. I think this is a dask bug
        df = df.assign(
            country_code=df["clean_code_tup"].map(itemgetter(0)),
            area_code=df["clean_code_tup"].map(itemgetter(1)),
            office_code=df["clean_code_tup"].map(itemgetter(2)),
            station_code=df["clean_code_tup"].map(itemgetter(3)),
            ext_num=df["clean_code_tup"].map(itemgetter(4)),
            _code_=df["clean_code_tup"].map(itemgetter(5)),
        )
    else:
        df = df.assign(
            _temp_=df["clean_code_tup"].map(itemgetter(0)),
            _code_=df["clean_code_tup"].map(itemgetter(1)),
        )
        df = df.rename(columns={"_temp_": f"{column}_clean"})

    # counts of codes indicating how values were changed
    stats = df["_code_"].value_counts(sort=False)
    df = df.drop(columns=["clean_code_tup", "_code_"])

    if inplace:
        df = df.drop(columns=column)

    # output a report describing the result of clean_phone
    #if report:
     #   create_report_new("Phone Number", stats, errors)

    return df