from __future__ import annotations
import logging
import math
import multiprocessing as mp
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import Flag, auto
from typing import Any, Dict, List, Optional

import zmq
from cachetools import TTLCache

from sglang.srt.tracing.trace import (
    TracePropagateContext,
    TraceReqContext,
    TraceThreadInfo,
    get_cur_time_ns,
)
from sglang.srt.tracing.trace import process_tracing_init as process_sync_tracing_init
from sglang.srt.tracing.trace import threads_info
from sglang.srt.tracing.trace import trace_set_thread_info as sync_trace_set_thread_info
from sglang.srt.utils import get_zmq_socket

logger = logging.getLogger(__name__)

import traceback

# connect resource for worker
ipc_name = None
worker_process = None
sender_socket = None

def py_debug(text):
    import threading
    pid = threading.get_native_id()
    text = f"[PID: {pid}] {text}"
    with open("/tmp/debug.txt", "a") as f:
        print(text, file=f)

class TraceContextTable(TTLCache):
    def __init__(self, ttl):
        super().__init__(maxsize=math.inf, ttl=ttl)

    def get(self, key):
        val = super().get(key)
        if val is not None:
            self[key] = val
        return val


# global trace context table
trace_context_table: Optional[TraceContextTable] = None


class TraceAction(Flag):
    # May be set simultaneously and need to be parsed in sequence.
    TRACE_ADD_ATTRS = auto()
    TRACE_EVENT = auto()
    TRACE_SLICE_END = auto()
    TRACE_REQ_ABORT = auto()
    TRACE_SLICE_START = auto()

    # Will not be set simultaneously with other actions.
    TRACE_REQ_START = auto()
    TRACE_REQ_FINISH = auto()
    TRACE_SET_THREAD_INFO = auto()


@dataclass
class TraceMessage:
    actions: TraceAction
    ts: Optional[int] = None
    rids: Optional[List[str]] = None
    # slice name or event name
    name: Optional[str] = None
    anonymous: bool = False
    # slice level
    trace_level: Optional[int] = None
    # slice virt id
    virt_id: Optional[int] = None
    # slice attributes or event attributes
    attrs: Optional[dict] = None

    req_ctx: Optional[TraceReqContext] = None
    propagate_ctx: Optional[Dict] = None
    thread_info: Optional[TraceThreadInfo] = None

    action_process = {
        TraceAction.TRACE_REQ_START: "_trace_req_start_process",
        TraceAction.TRACE_REQ_FINISH: "_trace_req_abort_process",
        TraceAction.TRACE_SET_THREAD_INFO: "_trace_set_thread_info_process",
        TraceAction.TRACE_SLICE_START: "_trace_slice_start_process",
        TraceAction.TRACE_SLICE_END: "_trace_slice_end_process",
        TraceAction.TRACE_ADD_ATTRS: "_trace_add_attrs_process",
        TraceAction.TRACE_EVENT: "_trace_event_process",
        TraceAction.TRACE_REQ_ABORT: "_trace_req_abort_process",
    }

    def _trace_req_start_process(self):
        assert len(self.rids) == 1
        rid = self.rids[0]
        #convert TraceReqContextAsync to TraceReqContext
        self.req_ctx = TraceReqContext.from_instance(self.req_ctx)
        self.req_ctx.trace_set_proc_propagate_context(self.propagate_ctx, self.ts)
        trace_context_table[rid] = self.req_ctx

    def _trace_req_abort_process(self):
        for rid in self.rids:
            req_ctx = trace_context_table.get(rid)
            if not req_ctx:
                continue
            req_ctx.abort(ts=self.ts, abort_info=self.attrs)

    def _trace_set_thread_info_process(self):
        threads_info[self.thread_info.pid] = self.thread_info

    def _trace_slice_start_process(self):
        for rid in self.rids:
            req_ctx = trace_context_table.get(rid)
            if not req_ctx:
                continue
            req_ctx.trace_slice_start(
                name="" if self.anonymous else self.name,
                ts=self.ts,
                anonymous=self.anonymous,
                level=self.trace_level,
                virt_id=self.virt_id,
            )

    def _trace_slice_end_process(self):
        for rid in self.rids:
            req_ctx = trace_context_table.get(rid)
            if not req_ctx:
                continue
            req_ctx.trace_slice_end(
                name=self.name,
                ts=self.ts,
                attrs=self.attrs,
                level=self.trace_level,
            )

    def _trace_add_attrs_process(self):
        for rid in self.rids:
            req_ctx = trace_context_table.get(rid)
            if not req_ctx:
                continue
            req_ctx.trace_slice_add_attr( self.attrs)

    def _trace_event_process(self):
        for rid in self.rids:
            req_ctx = trace_context_table.get(rid)
            if not req_ctx:
                continue
            req_ctx.trace_event(
                name=self.name,
                ts=self.ts,
                attrs=self.attrs,
            )

    def process(self):
        for action in self.actions:
            method = getattr(self, self.action_process[action])
            method()

class TraceReqContextAsync(TraceReqContext):
    def __init__(
        self,
        rid,
        bootstrap_room=None,
        role="null",
        tracing_enable=False,
        trace_level=1,
        module_name="",
    ):
        super().__init__(
            rid, bootstrap_room, role, tracing_enable, trace_level, module_name
        )

        self.span_virt_id_stack = []
        self.has_aborted = False

    def trace_req_start(
        self,
        ts: Optional[int] = None,
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()
        self._create_root_span(ts, external_trace_header)

        msg = TraceMessage(
            actions=TraceAction.TRACE_REQ_START,
            ts=ts,
            rids=[self.rid],
            req_ctx=self,
        )

        propagate_ctx = TracePropagateContext(self.root_span_context, None)
        msg.propagate_ctx = propagate_ctx.to_dict()

        sender_socket.send_pyobj(msg)

    def trace_req_finish(
        self, ts: Optional[int] = None, attrs: Optional[Dict[str, Any]] = None
    ):
        if not self.tracing_enable:
            return

        ts = ts or get_cur_time_ns()

        msg = TraceMessage(
            actions=TraceAction.TRACE_REQ_FINISH,
            ts=ts,
            rids=[self.rid],
        )
        sender_socket.send_pyobj(msg)
        if attrs:
            self.root_span.set_attributes(attrs)

        self.root_span.end(end_time=ts)
        self.has_aborted = True

    def trace_slice_start(
        self,
        name: str,
        ts: Optional[int] = None,
        anonymous: bool = False,
        level: int = 1,
        virt_id: Optional[int] = None,
    ):
        if not self.tracing_enable:
            return
        ts = ts or get_cur_time_ns()
        virt_id = virt_id or uuid.uuid4().hex[:8]
        self.span_virt_id_stack.append(virt_id)

        msg = TraceMessage(
            actions=TraceAction.TRACE_SLICE_START,
            ts=ts,
            rids=[self.rid],
            name=name,
            anonymous=anonymous,
            trace_level=level,
            virt_id=virt_id,
        )

        sender_socket.send_pyobj(msg)

    def trace_slice_end(
        self,
        name: str,
        ts: Optional[int] = None,
        attrs: Optional[Dict[str, Any]] = None,
        auto_next_anon: bool = False,
        thread_finish_flag: bool = False,
        level: int = 1,
    ):
        if not self.tracing_enable:
            return
        ts = ts or get_cur_time_ns()
        if self.span_virt_id_stack:
            self.span_virt_id_stack.pop()

        msg = TraceMessage(
            actions=TraceAction.TRACE_SLICE_END,
            ts=ts,
            rids=[self.rid],
            name=name,
            trace_level=level,
            attrs=attrs,
        )

        if thread_finish_flag:
            msg.actions = msg.actions | TraceAction.TRACE_REQ_ABORT
        elif auto_next_anon:
            virt_id = uuid.uuid4().hex[:8]
            self.span_virt_id_stack.append(virt_id)
            msg.actions = msg.actions | TraceAction.TRACE_SLICE_START
            msg.anonymous = True

        sender_socket.send_pyobj(msg)

    def trace_event(
        self, name: str, ts: Optional[int] = None, attrs: Dict[str, Any] = None
    ):

        if not self.tracing_enable:
            return
        ts = ts or get_cur_time_ns()

        msg = TraceMessage(
            actions=TraceAction.TRACE_EVENT,
            ts=ts,
            rids=[self.rid],
            name=name,
            attrs=attrs,
        )

        sender_socket.send_pyobj(msg)

    def trace_slice_add_attr(self, attrs: Dict[str, Any]):
        if not self.tracing_enable:
            return

        msg = TraceMessage(
            actions=TraceAction.TRACE_ADD_ATTRS,
            rids=[self.rid],
            attrs=attrs,
        )

        sender_socket.send_pyobj(msg)

    def trace_set_proc_propagate_context(self, trace_context: Optional[Dict[str, Any]]):
        if not self.tracing_enable:
            return

        self.is_copy = True
        _trace_context = TracePropagateContext.instance_from_dict(trace_context)
        self.root_span_context = _trace_context.root_span_context
        ts = get_cur_time_ns()
        msg = TraceMessage(
            actions=TraceAction.TRACE_REQ_START,
            rids=[self.rid],
            ts=ts,
            req_ctx=self,
        )

        msg.propagate_ctx = trace_context
        sender_socket.send_pyobj(msg)

    def trace_get_proc_propagate_context(self):
        if not self.tracing_enable:
            return None

        if not self.root_span_context:
            return None

        root_span_context = self.root_span_context
        prev_span_virt_id = None
        if self.span_virt_id_stack:
            prev_span_virt_id = self.span_virt_id_stack[-1]
        propagate_ctx = TracePropagateContext(root_span_context, prev_span_virt_id)
        return propagate_ctx.to_dict()

    def abort(self, ts=None, abort_info: Optional[Dict] = None):
        if not self.tracing_enable:
            return

        if self.has_aborted:
            return

        msg = TraceMessage(
            actions=TraceAction.TRACE_REQ_ABORT,
            ts=ts,
            rids=[self.rid],
            attrs=abort_info,
        )
        sender_socket.send_pyobj(msg)
        self.has_aborted = True

    def __del__(self):
        if sender_socket:
            self.abort(abort_info={"abort_info": "have unclosed span, auto closed"})

def async_trace_event_batch(
    name: str,
    rids: List,
    ts: Optional[int] = None,
    attrs: Dict[str, Any] = {},
):
    msg = TraceMessage(
        actions=TraceAction.TRACE_EVENT,
        ts=ts,
        rids=rids,
        name=name,
        attrs=attrs,
    )
    if sender_socket:
        sender_socket.send_pyobj(msg)


def async_trace_slice_end_batch(
    name: str,
    rids: List,
    ts: Optional[int] = None,
    attrs: Dict[str, Any] = {},
    auto_next_anon: bool = False,
    level: int = 1,
):
    # FIXME: not maintain span_virt_id_stack
    ts = ts or get_cur_time_ns()
    msg = TraceMessage(
        actions=TraceAction.TRACE_SLICE_END,
        ts=ts,
        rids=rids,
        name=name,
        trace_level=level,
        attrs=attrs,
    )
    if auto_next_anon:
        msg.actions = msg.actions | TraceAction.TRACE_SLICE_START
        msg.anonymous = True

    if sender_socket:
        sender_socket.send_pyobj(msg)


def async_trace_req_abort_batch(
    rids: List, ts: Optional[int] = None, abort_info: Optional[Dict] = None
):
    msg = TraceMessage(
        actions=TraceAction.TRACE_REQ_ABORT,
        ts=ts,
        rids=rids,
        attrs=abort_info,
    )
    if sender_socket:
        sender_socket.send_pyobj(msg)


def async_trace_worker(ipc_name, otlp_endpoint, server_name, init_event):
    process_sync_tracing_init(otlp_endpoint, server_name)

    # Initialize trace context table
    global trace_context_table
    trace_context_table = TraceContextTable(ttl=300)

    # Initialize ZMQ context and socket
    context = zmq.Context()
    receiver_socket = get_zmq_socket(context, zmq.PULL, ipc_name, True)

    logger.debug(f"Trace receiver process started, listening on {ipc_name}")

    init_event.set()

    try:
        while True:
            try:
                msg = receiver_socket.recv_pyobj(zmq.NOBLOCK)
                msg.process()
                
            except zmq.Again:
                time.sleep(0.01)  # Small delay to prevent busy waiting
    except KeyboardInterrupt:
        logger.info("Trace receiver process interrupted")
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error in trace receiver process: {e}")
    finally:
        receiver_socket.close()
        context.term()
        logger.info("Trace receiver process cleaned up")


def process_tracing_init(otlp_endpoint, server_name):
    global ipc_name
    global worker_process
    global sender_socket

    process_sync_tracing_init(otlp_endpoint, server_name)

    ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

    # Create and start a new process for receiving trace messages
    init_event = mp.Event()
    worker_process = mp.Process(
        target=async_trace_worker,
        args=(ipc_name, otlp_endpoint, server_name, init_event),
    )
    worker_process.start()
    if not init_event.wait(timeout=5):
        raise TimeoutError("Trace receiver process did not initialize within 5 seconds")

    context = zmq.Context()
    sender_socket = get_zmq_socket(context, zmq.PUSH, ipc_name, False)

def trace_set_thread_info(
    thread_label: str, tp_rank: Optional[int] = None, dp_rank: Optional[int] = None
):
    thread_info = sync_trace_set_thread_info(thread_label, tp_rank, dp_rank)
    msg = TraceMessage(
        actions=TraceAction.TRACE_SET_THREAD_INFO,
        thread_info=thread_info,
    )
    if sender_socket:
        sender_socket.send_pyobj(msg)
