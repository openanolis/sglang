from contextlib import contextmanager
from typing import Dict, Optional, Union

from sglang.srt.observability.req_time_stats import RequestStageConfig
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext


class MooncakeRequestStage:
    MOONCAKE_SEND = RequestStageConfig(
        "mooncake_send",
        level=1,
    )
    MOONCAKE_RECV = RequestStageConfig(
        "mooncake_recv",
        level=1,
    )
    MOONCAKE_WORKER_SEND = RequestStageConfig(
        "mooncake_worker_send",
        level=1,
    )
    MOONCAKE_WORKER_SEND_SESSION = RequestStageConfig(
        "mooncake_worker_send_session",
        level=2,
    )
    MOONCAKE_WORKER_RECV = RequestStageConfig(
        "mooncake_worker_recv",
        level=1,
    )


@contextmanager
def mooncake_trace_blk(
    trace_ctx: Union[TraceReqContext, TraceNullContext],
    stage: RequestStageConfig,
    attrs: Optional[Dict] = None,
    thread_finish_flag=False,
):
    if trace_ctx is None:
        yield
        return
    trace_ctx.trace_slice_start(stage.stage_name, stage.level)
    yield
    trace_ctx.trace_slice_end(
        stage.stage_name,
        stage.level,
        attrs=attrs,
        thread_finish_flag=thread_finish_flag,
    )


def mooncake_trace_func(stage: RequestStageConfig):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with mooncake_trace_blk(self.trace_ctx, stage):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
