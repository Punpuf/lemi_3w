import os
from typing import Optional, Text, List
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from constants import config
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen.custom_executors import parquet_executor
from tfx.v1 import proto
from tfx.dsl.components.base import executor_spec


# PIPELINE_NAME = config.MODEL_PIPELINE_NAME
# PIPELINE_ROOT = str(config.PIPELINE_ROOT)
# METADATA_PATH = str(config.METADATA_PATH)
# ENABLE_CACHE = False  # TODO config.ENABLE_CACHE

DATA_DIR = str(config.DIR_PROJECT_DATA)

""" RAW_DATA_DIR = str(
    config.DIR_CONVERTED_TEST_PREFIX
) """  # TODO config.DIR_CONVERTED_DATASET

DIR_CONVERTED_PREFIX = config.DIR_CONVERTED_TEST_PREFIX  # config.DIR_CONVERTED_PREFIX


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: bool,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
):
    components = []

    input_configuration = proto.Input(
        splits=[
            proto.Input.Split(
                name="raw_data",
                pattern=DIR_CONVERTED_PREFIX + "{SPAN}/*/*.parquet",
            ),
        ]
    )
    output_configuration = proto.Output(
        split_config=proto.SplitConfig(
            splits=[
                proto.SplitConfig.Split(name="train", hash_buckets=4),
                proto.SplitConfig.Split(name="eval", hash_buckets=1),
            ],
        )
    )

    example_gen = FileBasedExampleGen(
        custom_executor_spec=executor_spec.BeamExecutorSpec(parquet_executor.Executor),
        input_base=DATA_DIR,
        input_config=input_configuration,
        output_config=output_configuration,
    )
    logging.info(f"exampleGen example uri is {example_gen.outputs['examples']}")
    components.append(example_gen)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline():
    pipeline_lemi_3w = create_pipeline(
        pipeline_name=config.SCHEMA_PIPELINE_NAME,
        pipeline_root=str(config.SCHEMA_PIPELINE_ROOT),
        enable_cache=False,  # ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            str(config.SCHEMA_METADATA_PATH)
        ),
    )

    tfx.orchestration.LocalDagRunner().run(pipeline_lemi_3w)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()
