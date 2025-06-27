"""Custom configuration data class to allow for dictionary style and dot notation calling of
attributes.
"""

import re
import tomllib
from pathlib import Path


class Mapping(dict):
    """Dict-like class that allows for the use of dictionary style attribute calls on class
    attributes.
    """

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)


class Configuration(Mapping):
    """Configuration class for reading and converting nested dictionaries to allow for both
    namespace style and dot notation when collecting attributes.

    Customizations of the input data:
      - All fields containing "DIR" will be converted to a ``pathlib.Path`` object.
      - All nested data will be able to be called with dot notation and dictionary-style calls.
      - The `rev.turbine_class_dict` is converted to float data automatically.
      - All data in the `[sql]` section will get converted to proper constructor strings with the
        associated username and password data autopopulated with the match ``{USER}`` and
        ``PASSWORD`` fields in the same configuration section.
    """

    def __init__(self, config: str | Path | dict, *, initial: bool = True):
        """Create a hybrid dictionary and name space object for a given :py:attr:`config` and
        where all keys (including nested) are acessible with dictionary-style and dot notation.

        Args:
            config (str | Path | dict): A configuration dictionary or filename to the dictionary
                to read and convert. If passing a filename, it must be a TOML file.
            initial (bool, optional): Option to disable post-processing of configuration data.
        """
        if isinstance(config, str | Path):
            config = Path(config).resolve()
            with config.open("rb") as f:
                config = tomllib.load(f)

        for key, value in config.items():
            if isinstance(value, dict):
                self.__setattr__(key, Configuration(value, initial=False))
            else:
                if "DIR" in key:
                    self.__setattr__(key, Path(value).resolve())
                else:
                    self.__setattr__(key, value)

        if initial:
            self._convert_sql()
            self._convert_rev()

    def _convert_sql(self):
        """Replaces the "{USER}" and "{PASSWORD} portions of the sql constructor strings with
        the actual user and password information for ease of configuration reuse between users.
        """
        if "sql" in self:
            for key, value in self.sql.items():
                if key.startswith(("USER", "PASSWORD")):
                    continue
                for target in re.findall(r"\{(.*?)\}", value):
                    value = value.replace(target, self.sql[target])
                value = re.sub("[{}]", "", value)
                self.sql[key] = value

    def _convert_rev(self):
        if "rev" in self:
            self.rev.turbine_class_dict = {
                float(k): v for k, v in self.rev.turbine_class_dict.items()
            }
