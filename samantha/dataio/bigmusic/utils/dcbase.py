from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Type, TypeVar, Union, get_args, get_origin

T = TypeVar("T", bound="DCBase")


@dataclass
class DCBase:
    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Creates an instance of the dataclass from a dictionary, initializing nested dataclasses if needed."""
        if data is None:
            return None

        field_types = {
            f.name: f.type for f in fields(cls)
        }  # Get actual dataclass fields and their types
        field_names = set(field_types.keys())

        filtered_data = {}
        for key, value in data.items():
            if key in field_names:
                field_type = field_types[key]
                origin_type = get_origin(
                    field_type
                )  # Get origin type for generic types like Dict, List, etc.
                args = get_args(field_type)

                # Handle Optional types
                if origin_type is Union and type(None) in args:
                    # Check if the value is already None
                    if value is None:
                        filtered_data[key] = None
                        continue
                    # Extract the actual type from Optional
                    actual_types = [arg for arg in args if arg is not type(None)]
                    if len(actual_types) == 1:
                        actual_type = actual_types[0]
                        origin_type = get_origin(actual_type)
                        args = get_args(actual_type)
                    else:
                        # Complex Union type - just use the value as is
                        filtered_data[key] = value
                        continue
                # If the field type is a subclass of _DCBase, recursively initialize it
                if (
                    isinstance(value, dict)
                    and isinstance(field_type, type)
                    and issubclass(field_type, DCBase)
                ):
                    filtered_data[key] = field_type.from_dict(value)
                # Handle generic types like Dict[str, _DCBase]
                elif (
                    origin_type is dict
                    and len(args) == 2
                    and isinstance(args[1], type)
                    and issubclass(args[1], DCBase)
                ):
                    filtered_data[key] = {
                        k: args[1].from_dict(v) for k, v in value.items()
                    }
                # Handle generic types like List[_DCBase]
                elif (
                    origin_type is list
                    and len(args) == 1
                    and isinstance(args[0], type)
                    and issubclass(args[0], DCBase)
                ):
                    filtered_data[key] = [args[0].from_dict(v) for v in value]
                else:
                    filtered_data[key] = value

        return cls(**filtered_data)

    def to_dict(self) -> dict:
        return _asdict_with_properties(self)

    def _replace(self, **kwargs):
        # backward compatible
        return replace(self, **kwargs)

    def validate(self):
        pass

    def post_init(self):
        pass

    def __post_init__(self):
        self.validate()
        self.post_init()


def _asdict_with_properties(obj) -> dict:
    """Convert a dataclass instance to a dictionary, including properties."""
    result = asdict(obj)  # Get regular fields
    props = {
        name: getattr(obj, name)
        for name in dir(obj)
        if isinstance(
            getattr(type(obj), name, None), property
        )  # Check if it's a property
    }
    result.update(props)  # Merge properties into the dictionary
    return result
