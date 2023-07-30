import json
import importlib
import sys
from pathlib import Path

from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, RequestOutput

# Super hacky way to load codec. Don't do this in prod.
# Trying to relative import will result in import error:
# ImportError: attempted relative import with no known parent package
# from .model_wrapper_codec import ModelWrapperCodec
model_dir = Path(__file__).resolve().parent
codec_file: Path = model_dir / 'model_wrapper_codec.py'
file_path = codec_file
module_name = codec_file.name
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
ModelWrapperCodec = getattr(module, 'ModelWrapperCodec')


class MyModel(MLModel):

    async def load(self) -> bool:
        print("Initialize model")
        print(self.settings)

        model_dir = Path(__file__).resolve().parent
        print(f'Model dir: {model_dir}')

        artifact_dir: Path = model_dir / self.settings.parameters.uri
        print(f'Artifact dir: {artifact_dir}')

        model_file: Path = artifact_dir / 'model.py'
        print(f'Original model file: {model_file}')

        file_path = model_file
        module_name = model_file.name

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        self._model_func = getattr(module, 'infer')

        return await super().load()

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # print('--- parameters ---')
        # print(f'Request content_type: {payload.parameters.content_type}')

        outputs = []
        for inp in payload.inputs:
            decoded = self.decode(inp, default_codec=ModelWrapperCodec)
            results = self._model_func(decoded[0])  # TODO: this is returning as list of lists. fix this
            output = RequestOutput(name="results")
            outputs.append(self.encode(results, output, default_codec=ModelWrapperCodec))

        return InferenceResponse(model_name=self.name, outputs=outputs)
