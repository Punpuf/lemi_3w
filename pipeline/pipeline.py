import os
from typing import Optional, Text, List
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from constants import config


PIPELINE_NAME = config.PIPELINE_NAME
PIPELINE_ROOT = config.PIPELINE_ROOT
METADATA_PATH = config.METADATA_PATH
ENABLE_CACHE = config.ENABLE_CACHE


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    enable_cache: bool,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
):
    components = []

    # TODO add components :D

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
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
    )

    tfx.orchestration.LocalDagRunner().run(pipeline_lemi_3w)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run_pipeline()
