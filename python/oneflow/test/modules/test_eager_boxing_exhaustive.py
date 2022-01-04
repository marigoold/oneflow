"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import os

import oneflow.unittest
from test_util import GenArgList


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerConsistentCastExhaustiveTesting(flow.unittest.TestCase):
    def test_eager_consistent_cast_1d_exhaustive_testing(test_case):
        import itertools

        sbps = [
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
        ]
        np.random.seed(10)
        np_arr = np.random.uniform(-1e2, 1e2, (4, 8))
        placement = flow.placement("cuda", {0: range(4)})
        for elem in itertools.product(sbps, sbps):
            if flow.env.get_rank() == 0:
                print(elem)
            x = flow.tensor(
                np_arr,
                dtype=flow.float32,
                placement=placement,
                sbp=[elem[0]],
                requires_grad=False,
            )
            y = x.to_consistent(placement=placement, sbp=[elem[1]])

            z = y.to_consistent(placement=placement, sbp=[flow.sbp.broadcast])
            test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr),)

    def test_eager_consistent_cast_2d_exhaustive_testing(test_case):
        import itertools

        sbps = [
            flow.sbp.split(0),
            flow.sbp.split(1),
            flow.sbp.broadcast,
            flow.sbp.partial_sum,
        ]
        nd_sbps = itertools.product(
            itertools.product(sbps, sbps), itertools.product(sbps, sbps)
        )
        np.random.seed(10)
        np_arr = np.random.uniform(-1e2, 1e2, (32, 96, 64))
        placement = flow.placement("cuda", {0: range(4)}, (2, 2))
        failed_boxing = []
        for elem in nd_sbps:
            if flow.env.get_rank() == 0:
                print(elem)
            try:
                x = flow.tensor(
                    np_arr,
                    dtype=flow.float32,
                    placement=placement,
                    sbp=elem[0],
                    requires_grad=False,
                )
                y = x.to_consistent(placement=placement, sbp=elem[1])

                z = y.to_consistent(
                    placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
                )
                test_case.assertTrue(np.allclose(z.to_local().numpy(), np_arr,),)
            except oneflow._oneflow_internal.exception.UnimplementedException:
                failed_boxing.append(elem)

        if flow.env.get_rank() == 0:
            print("unsuported boxing 2d type", failed_boxing)


if __name__ == "__main__":
    unittest.main()
