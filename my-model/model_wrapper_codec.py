import json
from typing import Any, List

from mlserver.codecs import InputCodec, register_input_codec
from mlserver.codecs.lists import is_list_of, as_list
from mlserver.types import RequestInput, ResponseOutput, Parameters


@register_input_codec
class ModelWrapperCodec(InputCodec):
    ContentType = "my-model"
    TypeHint = List[bytes]

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        print("can_encode")
        return is_list_of(payload, bytes)

    @classmethod
    def encode_output(
        cls,
        name: str,
        payload: List[bytes],
        **kwargs
    ) -> ResponseOutput:
        """Gets called by MLServer self.encode()."""
        data = list(map(json.dumps, as_list(payload)))
        shape = [-1, len(data)]

        return ResponseOutput(
            name=name,
            shape=shape,
            datatype="BYTES",
            data=data,
            parameters=Parameters(content_type=cls.ContentType),
        )

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> List[bytes]:
        print('decode_output')
        print(response_output.data)
        print('/decode_output')
        print()
        return response_output.data

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> List[bytes]:
        """Gets called by MLServer self.decode()."""
        data = request_input.data.__root__
        data = list(map(json.loads, as_list(data)))
        return data

    @classmethod
    def encode_input(
        cls,
        name: str,
        payload: List[bytes],
        **kwargs
    ) -> RequestInput:
        output = cls.encode_output(name, payload)

        print('encode_input')
        print(payload)
        print('/encode_input')
        print()

        return RequestInput(
            name=output.name,
            shape=output.shape,
            datatype=output.datatype,
            data=output.data,
        )