import pytest
import random
import torch

import inferno
from inferno.extra import ExactNeuron
from inferno.neural import (
    LinearDense,
    LinearDirect,
    LinearLateral,
    RecurrentSerial,
    DeltaCurrent,
)
from inferno.observe import InputMonitor, PassthroughReducer


def random_shape(mindims=1, maxdims=9, minsize=1, maxsize=9):
    return tuple(
        random.randint(mindims, maxdims)
        for _ in range(random.randint(minsize, maxsize))
    )


class TestRecurrentSerial:

    def test_trainable_cell_creation(self):
        feedfwd_conn = LinearDense(
            (5,),
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        lateral_conn = LinearDirect(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedback_conn = LinearLateral(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedfwd_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        feedback_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        layer = RecurrentSerial(
            feedfwd_conn,
            lateral_conn,
            feedback_conn,
            feedfwd_neur,
            feedback_neur,
            trainable_feedback=True,
        )
        assert layer.feedfwd_cell is not None
        assert layer.lateral_cell is not None
        assert layer.feedback_cell is not None

    def test_untrainable_cell_creation(self):
        feedfwd_conn = LinearDense(
            (5, 3),
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        lateral_conn = LinearDense(
            (3,),
            (5, 5),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedback_conn = LinearDense(
            (5, 5),
            (5, 3),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedfwd_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        feedback_neur = ExactNeuron((5, 5), 1.0, rest_v=-60.0, thresh_v=-50.0)
        layer = RecurrentSerial(
            feedfwd_conn,
            lateral_conn,
            feedback_conn,
            feedfwd_neur,
            feedback_neur,
            trainable_feedback=False,
        )
        assert layer.feedfwd_cell is not None
        assert layer.lateral_cell is None
        assert layer.feedback_cell is None

    def test_initial_zeros(self):
        feedfwd_conn = LinearDense(
            (5,),
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        lateral_conn = LinearDirect(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedback_conn = LinearLateral(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedfwd_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        feedback_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        layer = RecurrentSerial(
            feedfwd_conn,
            lateral_conn,
            feedback_conn,
            feedfwd_neur,
            feedback_neur,
            trainable_feedback=True,
        )
        lateral_mon = InputMonitor(PassthroughReducer(1.0), lateral_conn)
        feedback_mon = InputMonitor(PassthroughReducer(1.0), feedback_conn)
        _ = layer(
            inferno.ones(feedfwd_neur.spike, shape=feedfwd_conn.batched_inshape),
            feedback_neuron_kwargs={"override": inferno.ones(feedback_neur.spike)},
            capture_intermediate=True,
        )

        assert torch.all(lateral_mon.peek() == feedfwd_neur.spike)
        assert torch.all(feedback_mon.peek() == 0)

    def test_forward(self):
        feedfwd_conn = LinearDense(
            (5,),
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        lateral_conn = LinearDirect(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedback_conn = LinearLateral(
            (3,),
            1.0,
            synapse=DeltaCurrent.partialconstructor(1.0),
        )
        feedfwd_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        feedback_neur = ExactNeuron((3,), 1.0, rest_v=-60.0, thresh_v=-50.0)
        layer = RecurrentSerial(
            feedfwd_conn,
            lateral_conn,
            feedback_conn,
            feedfwd_neur,
            feedback_neur,
            trainable_feedback=True,
        )

        feedfwd_mon = InputMonitor(PassthroughReducer(1.0), feedfwd_conn)
        lateral_mon = InputMonitor(PassthroughReducer(1.0), lateral_conn)
        feedback_mon = InputMonitor(PassthroughReducer(1.0), feedback_conn)

        feedfwd_in = None
        feedback_in = inferno.zeros(feedback_neur.spike)

        for _ in range(random.randint(5, 12)):
            feedfwd_in = (
                inferno.uniform(
                    feedfwd_neur.spike,
                    shape=feedfwd_conn.batched_inshape,
                    dtype=torch.float32,
                )
                > 0.5
            )
            feedfwd_out = inferno.uniform(feedfwd_neur.spike, dtype=torch.float32) > 0.5
            feedback_out = (
                inferno.uniform(feedback_neur.spike, dtype=torch.float32) > 0.5
            )

            L, res = layer(
                feedfwd_in,
                feedfwd_neuron_kwargs={"override": feedfwd_out},
                feedback_neuron_kwargs={"override": feedback_out},
                capture_intermediate=True,
            )

            assert torch.all(feedfwd_mon.peek() == feedfwd_in)
            assert torch.all(lateral_mon.peek() == feedfwd_out)
            assert torch.all(feedback_mon.peek() == feedback_in)

            assert torch.all(res["feedfwd"] == feedfwd_conn.forward(feedfwd_in))
            assert torch.all(res["lateral"] == lateral_conn.forward(feedfwd_out))
            assert torch.all(res["feedback"] == feedback_conn.forward(feedback_in))

            feedback_in = feedback_neur.spike
