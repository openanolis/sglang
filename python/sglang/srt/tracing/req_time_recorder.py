# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""definition for requests stage timing recorder"""
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.tracing.trace import (
    SglangTraceReqContext,
    get_cur_time_ns,
    get_opentelemetry_initialized,
)
from sglang.srt.utils import get_bool_env_var

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")


@dataclass
class TimeStats:
    # some important timestamps for scheduler maybe.
    ### !!! need to check record points.
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0

    # lb_entry_time: float = 0.0
    # prefill_transfer_queue_entry_time: float = 0.0
    # decode_transfer_queue_entry_time: float = 0.0

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time


@dataclass
class RequestStageConfig:
    stage_name: str
    # record stage start time to the corresponding field of TimeStats when call trace_slice_start.
    # will not be recorded if trace_slice_start(anon=True)
    start_time_stat_field: Optional[str] = None
    # record stage end time to the corresponding field of TimeStats when call trace_slice_end.
    end_time_stat_field: Optional[str] = None
    level: int = 0
    metrics_collector_specific_func: Optional[str] = None
    metrics_collector_args: Optional[List] = None
    # whether to call metrics_collector.observe_per_stage_req_latency
    metrics_is_observed: bool = False


class RequestStage:
    # Tokenizer
    TOKENIZE = RequestStageConfig(
        "tokenize",
        level=1,
    )
    TOKENIZER_DISPATCH = RequestStageConfig(
        "dispatch",
        level=2,
    )

    # DP controller
    DC_DISPATCH = RequestStageConfig(
        "dc_dispatch",
        level=2,
    )

    # common/non-disaggregation
    REQUEST_PROCESS = RequestStageConfig(
        "request_process",
        end_time_stat_field="wait_queue_entry_time",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_WAITING = RequestStageConfig(
        "prefill_waiting",
        end_time_stat_field="forward_entry_time",
        level=1,
        metrics_collector_specific_func="observe_queue_time",
        metrics_is_observed=True,
    )
    DECODE_LOOP = RequestStageConfig(
        "decode_loop",
        level=2,
    )
    PREFILL_FORWARD = RequestStageConfig(
        "prefill_forward",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_CHUNKED_FORWARD = RequestStageConfig(
        "chunked_prefill",
        level=3,
        metrics_is_observed=True,
    )

    # disaggregation prefill
    PREFILL_PREPARE = RequestStageConfig(
        "prefill_prepare",
        end_time_stat_field="prefill_bootstrap_queue_entry_time",
        level=1,
    )
    PREFILL_BOOTSTRAP = RequestStageConfig(
        "prefill_bootstrap",
        end_time_stat_field="wait_queue_entry_time",
        level=1,
        metrics_is_observed=True,
    )
    PREFILL_TRANSFER_KV_CACHE = RequestStageConfig(
        "prefill_transfer_kv_cache",
        level=1,
        metrics_is_observed=True,
    )

    # disaggregation decode
    DECODE_PREPARE = RequestStageConfig(
        "decode_prepare",
        end_time_stat_field="decode_prealloc_queue_entry_time",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_BOOTSTRAP = RequestStageConfig(
        "decode_bootstrap",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_WAITING = RequestStageConfig(
        "decode_waiting",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_TRANSFERRED = RequestStageConfig(
        "decode_transferred",
        end_time_stat_field="wait_queue_entry_time",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_FAKE_OUTPUT = RequestStageConfig(
        "fake_output",
        end_time_stat_field="forward_entry_time",
        level=1,
        metrics_is_observed=True,
    )
    DECODE_QUICK_FINISH = RequestStageConfig(
        "quick_finish",
        level=1,
        metrics_is_observed=True,
    )

    # mini lb
    MINI_LB_LAUNCH = RequestStageConfig(
        "mini_lb_launch",
        level=1,
    )

    WAIT_PD_FINISH = RequestStageConfig(
        "wait_pd_finish",
        level=2,
    )

    # other
    ANONYMOUS = RequestStageConfig("")


class RequestTimeRecorder(SglangTraceReqContext):
    def __init__(
        self,
        rid,
        bootstrap_room,
        module_name,
        server_args,
        metrics_collector=None,
        propagation_context: Optional[Dict[str, Any]] = None,
        time_stat_cls=None,
        role: Optional[str] = None,
        ts: Optional[int] = None,
    ):
        self.module_name = module_name
        self.enable_metrics = getattr(server_args, "enable_metrics", False)
        self.enable_request_time_stats_logging = getattr(
            server_args, "enable_request_time_stats_logging", False
        )
        self.disagg_mode = getattr(server_args, "disaggregation_mode", "null")
        opentelemetry_initialized = get_opentelemetry_initialized()

        self.metrics_collector = metrics_collector

        if not metrics_collector:
            self.enable_metrics = False

        if isinstance(time_stat_cls, type):
            self.time_stats = time_stat_cls()

        trace_level = getattr(server_args, "trace_level", 0)
        tracing_enable = (
            True
            if getattr(server_args, "trace_module", None) == module_name
            and trace_level > 0
            and opentelemetry_initialized
            else False
        )
        time_record_enable = (
            tracing_enable
            or self.enable_metrics
            or self.enable_request_time_stats_logging
        )

        if not role:
            role = self.disagg_mode
        super().__init__(
            rid=str(rid),
            bootstrap_room=bootstrap_room,
            role=role,
            tracing_enable=tracing_enable,
            time_record_enable=time_record_enable,
            trace_level=trace_level,
        )

        if isinstance(propagation_context, dict):
            super().trace_set_proc_propagate_context(propagation_context)
        else:
            super().trace_req_start(ts)

        # stage latency cache for log print. (s)
        self.stage_time_cache: Dict[str, List] = {}

    def metric_trace_slice_start(
        self,
        stage: RequestStageConfig,
        ts: Optional[int] = None,
    ):
        if not self.time_record_enable:
            return

        if stage.start_time_stat_field and self.enable_request_time_stats_logging:
            ts = ts or get_cur_time_ns()
            try:
                setattr(self.time_stats, stage.start_time_stat_field, ts / 1e9)
            except AttributeError:
                pass

        super().trace_slice_start(
            stage.stage_name,
            ts=ts,
            anonymous=(stage == RequestStage.ANONYMOUS),
            level=stage.level,
        )

    def metric_trace_slice_end(
        self,
        stage: RequestStageConfig,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        auto_next_anon: bool = False,
        thread_finish_flag: bool = False,
    ):
        if not self.time_record_enable:
            return

        if not self.thread_context.cur_slice:
            return

        ts = ts or get_cur_time_ns()

        if self.enable_metrics:
            lat = (ts - self.thread_context.cur_slice.start_time_ns) / 1e9
            if stage.metrics_collector_specific_func:
                try:
                    metrics_collector_args = (
                        stage.metrics_collector_args
                        if stage.metrics_collector_args
                        else []
                    )
                    getattr(
                        self.metrics_collector, stage.metrics_collector_specific_func
                    )(*metrics_collector_args, lat)
                except AttributeError:
                    pass

            if stage.metrics_is_observed:
                try:
                    self.metrics_collector.observe_per_stage_req_latency(
                        stage.stage_name,
                        lat,
                    )
                except AttributeError:
                    pass

        if self.enable_request_time_stats_logging:
            lat = (ts - self.thread_context.cur_slice.start_time_ns) / 1e9
            self.stage_time_cache.setdefault(stage.stage_name, []).append(lat)

            if stage.end_time_stat_field:
                try:
                    setattr(self.time_stats, stage.end_time_stat_field, ts / 1e9)
                    if thread_finish_flag:
                        setattr(self.time_stats, "completion_time", ts / 1e9)
                except AttributeError:
                    pass

        self.trace_slice_end(
            stage.stage_name,
            ts=ts,
            attrs=attrs,
            auto_next_anon=auto_next_anon,
            thread_finish_flag=thread_finish_flag,
            level=stage.level,
        )

    metric_trace_slice = metric_trace_slice_end

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def convert_to_duration(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL.value:
            queue_duration = self.stage_time_cache.get(
                RequestStage.PREFILL_WAITING.name, [0]
            )[0]
            forward_duration = (
                self.time_stats.completion_time - self.time_stats.wait_queue_entry_time
            )

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time:.3f}"

        elif self.disagg_mode == DisaggregationMode.PREFILL.value:
            bootstrap_duration = self.stage_time_cache.get(
                RequestStage.PREFILL_BOOTSTRAP.name, [0]
            )[0]
            queue_duration = self.stage_time_cache.get(
                RequestStage.PREFILL_WAITING.name, [0]
            )[0]
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        bootstrap_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.time_stats.prefill_bootstrap_queue_entry_time:.3f}"

        elif self.disagg_mode == DisaggregationMode.DECODE.value:
            prealloc_duration = self.stage_time_cache.get(
                RequestStage.DECODE_BOOTSTRAP.name, [0]
            )
            transfer_duration = self.stage_time_cache.get(
                RequestStage.DECODE_TRANSFER.name, [0]
            )
            queue_duration = self.stage_time_cache.get(
                RequestStage.DECODE_WAITING.name, [0]
            )
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        prealloc_duration >= 0
                        and transfer_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0. {self=}"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time:.3f}"
        else:
            return "Unknown Time Stats"

    def disagg_mode_str(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL.value:
            return "unified"
        elif self.disagg_mode == DisaggregationMode.DECODE.value:
            return "decode"
        elif self.disagg_mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        else:
            return "unknown"


class NoOpTimeRecorder:
    __slots__ = ()

    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        pass

    def __delattr__(self, name):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __repr__(self):
        return "<NullObject>"


def metric_trace_slice_batch(
    stage: RequestStageConfig,
    reqs: List,
):
    if not reqs or not reqs[0].time_recorder.time_record_enable:
        return

    for req in reqs:
        req.time_recorder.metric_trace_slice(
            stage,
            auto_next_anon=not req.finished(),
            thread_finish_flag=req.finished(),
        )


def trace_event_batch(
    name: str,
    reqs: List,
    ts: Optional[int] = None,
    attrs: Dict[str, Any] = None,
):
    if not reqs or not reqs[0].time_recorder.time_record_enable:
        return

    for req in reqs:
        req.time_recorder.trace_event(name, ts=ts, attrs=attrs)


"""
Used when the time_recorder cannot be integrated into the request object.

format:
    {
        thread_id: {
            "rid": RequestTimeRecorder
        }
    }
"""
global_time_recorder_table: Dict[int, Dict[str, RequestTimeRecorder]] = {}


def global_init_time_recorder(
    rid,
    bootstrap_room,
    module_name,
    server_args,
    metrics_collector=None,
    propagation_context: Optional[Dict[str, Any]] = None,
    time_stat_cls=None,
    role: Optional[str] = None,
):
    pid = threading.get_native_id()
    rid = str(rid)
    time_recorder = RequestTimeRecorder(
        rid=rid,
        bootstrap_room=bootstrap_room,
        module_name=module_name,
        server_args=server_args,
        metrics_collector=metrics_collector,
        propagation_context=propagation_context,
        time_stat_cls=time_stat_cls,
        role=role,
    )

    global_time_recorder_table.setdefault(pid, {})[rid] = time_recorder

    return time_recorder


def global_get_time_recorder(rid) -> Union[RequestTimeRecorder, NoOpTimeRecorder]:
    pid = threading.get_native_id()
    rid = str(rid)
    if pid in global_time_recorder_table:
        if rid in global_time_recorder_table[pid]:
            return global_time_recorder_table[pid][rid]
    return NoOpTimeRecorder()


def global_set_time_recorder(time_recorder):
    pid = threading.get_native_id()
    rid = time_recorder.rid
    global_time_recorder_table.setdefault(pid, {})[rid] = time_recorder


def gloabl_del_timer_recorder(rid):
    pid = threading.get_native_id()
    rid = str(rid)
    if pid in global_time_recorder_table:
        if rid in global_time_recorder_table[pid]:
            del global_time_recorder_table[pid][rid]


def trace_inject_propagate_context(obj):
    if hasattr(obj, "time_recorder"):
        old_time_recorder = obj.time_recorder
        obj.time_recorder = obj.time_recorder.trace_get_proc_propagate_context()
        return old_time_recorder
    else:
        return None


def trace_restore_time_recorder(obj, old_time_recorder):
    if hasattr(obj, "time_recorder"):
        obj.time_recorder = old_time_recorder
