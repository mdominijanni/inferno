inferno.monitoring
==========================

.. automodule:: inferno.monitoring

.. currentmodule:: torch

The monitoring suite allows for monitoring a :py:class:`nn.Module` object.

When the module is called, monitors can be used to capture inputs to that module,
outputs from that module, or attributes contained within that module---so long as
they are of type :py:class:`Tensor`. This is achieved using the module's
:py:meth:`.register_forward_pre_hook` and :py:meth:`.register_forward_hook` methods,
which automatically are called when the module itself are called.

Each monitor is passed a reducer when its constructed. The reducer controls how
the monitored attribute is processed. For example, the history of that attribute may
be stored in its entirety, only its most recent value, the average value of that
attribute over time, etc.

Strictly, reducers can also be used on their own to record and process the history of
any :py:class:`Tensor` passed to them.

.. currentmodule:: inferno.monitoring

In general, reducers can be moved between devices and their datatype can be changed.
The exception to this is :py:class:`SinglePassthroughReducer` which takes its state
from the last input only. The :py:meth:`peak` method of reducers typically expose
the underlying state tensor and as such should not be modified externally.

.. currentmodule:: inferno.monitoring

Monitors
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AbstractMonitor
    AbstractPreMonitor
    InputMonitor
    OutputMonitor
    StateMonitor
    StatePreMonitor

Reducers
---------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AbstractReducer
    PassthroughReducer
    SinglePassthroughReducer
    TraceReducer
    AdditiveTraceReducer
    ScalingTraceReducer
    LastEventReducer
    FuzzyLastEventReducer
    SMAReducer
    CMAReducer
    EMAReducer