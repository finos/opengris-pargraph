import contextlib
import json
import sys
import types
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, Dict, List, cast
from unittest.mock import patch

from pargraph import GraphEngine, delayed, graph
from pargraph.engine.engine import Backend
from pargraph.graph.objects import FunctionCall, Graph

try:
    import pandas as pd  # noqa

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False

try:
    import parfun as pf  # noqa

    PARFUN_INSTALLED = True
except ImportError:
    PARFUN_INSTALLED = False


class TestGraphGeneration(unittest.TestCase):
    engine: GraphEngine

    @classmethod
    def setUpClass(cls):
        cls.engine = GraphEngine()

    def test_basic(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dict(w=1, x=2, y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_task_graph_positional_arguments(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dict(1, 2, 3, 4))[0], sample_graph(1, 2, 3, 4))

    def test_task_graph_positional_and_keyword_arguments(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dict(1, 2, y=3, z=4))[0], sample_graph(1, 2, y=3, z=4)
        )

    def test_subgraph(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_subgraph(x: int, y: int) -> int:
            return add(x, y)

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return sample_subgraph(sample_subgraph(w, x), sample_subgraph(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dict(w=1, x=2, y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_basic_partial(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph(w=1, x=2).to_dict(y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_variadic_arguments(self):
        @delayed
        def add(*args: int) -> int:
            return sum(args)

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(w, x, y, z)

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dict(w=1, x=2, y=3, z=4))[0], add(1, 2, 3, 4))

    def test_operator_override(self):
        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return (w + x) + (y + z)

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dict(w=1, x=2, y=3, z=4))[0], sample_graph(w=1, x=2, y=3, z=4)
        )

    def test_operator_override_complex(self):
        @graph
        def fibonacci(n: int) -> int:
            phi = (1 + 5**0.5) / 2
            return round(((phi**n) + ((1 - phi) ** n)) / 5**0.5)

        self.assertEqual(self.engine.get(*fibonacci.to_graph().to_dict(n=6))[0], fibonacci(n=6))

    def test_getitem(self):
        @delayed
        def return_tuple(x: int, y: int) -> Any:
            return x, y

        @graph
        def sample_graph(x: int, y: int) -> int:
            return return_tuple(x, y)[0]

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dict(x=1, y=2))[0], sample_graph(x=1, y=2))

    @unittest.skipIf(not PANDAS_INSTALLED, "pandas must be installed")
    def test_call(self):
        import pandas as pd  # noqa

        @graph
        def sample_graph(s: pd.Series) -> int:
            return s.sum()

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().to_dict(s=pd.Series([1, 2, 3])))[0],
            sample_graph(s=pd.Series([1, 2, 3])),
        )

    @unittest.skipIf(not PANDAS_INSTALLED, "pandas must be installed")
    def test_call_complex(self):
        import pandas as pd  # noqa

        @graph
        def sample_graph(s: pd.Series) -> pd.Series:
            return s[s > s.mean()]

        pd.testing.assert_series_equal(
            self.engine.get(*sample_graph.to_graph().to_dict(s=pd.Series([1, 2, 3])))[0],
            sample_graph(s=pd.Series([1, 2, 3])),
        )

    @unittest.skipIf(not PANDAS_INSTALLED, "pandas must be installed")
    def test_call_kw(self):
        import pandas as pd  # noqa

        @graph
        def sample_graph(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
            return df1.merge(df2, how="inner", on="a")

        pd.testing.assert_frame_equal(
            self.engine.get(
                *sample_graph.to_graph().to_dict(
                    df1=pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]}),
                    df2=pd.DataFrame({"a": ["foo", "baz"], "c": [3, 4]}),
                )
            )[0],
            sample_graph(
                df1=pd.DataFrame({"a": ["foo", "bar"], "b": [1, 2]}),
                df2=pd.DataFrame({"a": ["foo", "baz"], "c": [3, 4]}),
            ),
        )

    def test_boolean(self):
        @delayed
        def is_even(value: int) -> bool:
            return value % 2 == 0

        @delayed
        def is_positive(value: int) -> bool:
            return value >= 0

        @graph
        def invalid_graph(value: int) -> bool:
            return is_even(value) or is_positive(value)

        self.assertRaises(TypeError, invalid_graph.to_graph)

    def test_explode_subgraphs(self):
        @graph
        def sample_subgraph(x: int, y: int) -> int:
            return (x + y) * 2

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return sample_subgraph(sample_subgraph(w, x), sample_subgraph(y, z))

        self.assertGreater(
            len(sample_graph.to_graph().explode_subgraphs().nodes), len(sample_subgraph.to_graph().nodes)
        )

        self.assertEqual(
            self.engine.get(*sample_graph.to_graph().explode_subgraphs().to_dict(w=1, x=2, y=3, z=4))[0],
            sample_graph(w=1, x=2, y=3, z=4),
        )

    def test_stabilize(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        self.assertNotEqual(
            json.dumps(sample_graph.to_graph().to_json()), json.dumps(sample_graph.to_graph().to_json())
        )

        self.assertEqual(
            json.dumps(sample_graph.to_graph().stabilize().to_json()),
            json.dumps(sample_graph.to_graph().stabilize().to_json()),
        )

    def test_cull(self):
        @delayed
        def add(x: int, y: int) -> int:
            return x + y

        @graph
        def sample_graph(w: int, x: int, y: int, z: int) -> int:
            return add(add(w, x), add(y, z))

        generated_graph = sample_graph.to_graph()
        generated_graph = Graph(
            consts=generated_graph.consts, inputs=generated_graph.inputs, nodes=generated_graph.nodes, outputs={}
        )

        self.assertEqual(len(generated_graph.cull().nodes), 0)

    def test_fuse_sequential(self):
        @graph
        def attr_access(a: SimpleNamespace) -> int:
            return a.b.c.d.e

        self.assertEqual(
            self.engine.get(
                *attr_access.to_graph()
                .fuse_sequential()
                .to_dict(a=SimpleNamespace(b=SimpleNamespace(c=SimpleNamespace(d=SimpleNamespace(e=1)))))
            )[0],
            attr_access(a=SimpleNamespace(b=SimpleNamespace(c=SimpleNamespace(d=SimpleNamespace(e=1))))),
        )

        self.assertLess(len(attr_access.to_graph().fuse_sequential().nodes), len(attr_access.to_graph().nodes))

    def test_valid_delayed_signature(self):
        @delayed
        def valid1(arg: int) -> int: ...

        valid1(1)

        @delayed
        def valid2(arg1: int, arg2: int) -> int: ...

        valid2(1, 1)

        @delayed
        def valid3(*args: List[int]) -> int: ...

        valid3(1, 1, 1)

    def test_invalid_delayed_signature(self):
        with self.assertRaises(ValueError):

            @delayed
            def invalid1(**kwargs: Dict[str, int]) -> int: ...

            invalid1(a=1)

        with self.assertRaises(ValueError):

            @delayed
            def invalid2(arg: int, *args: List[int]) -> int: ...

            invalid2(1, 1)

        with self.assertRaises(ValueError):

            @delayed
            def invalid3(*args: List[int], **kwargs: Dict[str, int]) -> int: ...

            invalid3(1, 1, a=1)

    def test_valid_graph_signature(self):
        @graph
        def valid1(arg: int) -> int: ...

        valid1(1)

        @graph
        def valid2(arg1: int, arg2: int) -> int: ...

        valid2(1, 1)

    def test_invalid_graph_signature(self):
        with self.assertRaises(ValueError):

            @graph
            def invalid1(**kwargs: Dict[str, int]) -> int: ...

            invalid1(a=1)

        with self.assertRaises(ValueError):

            @graph
            def invalid2(*args: List[int]) -> int: ...

            invalid2(1, 1)

        with self.assertRaises(ValueError):

            @graph
            def invalid3(*args: List[int], **kwargs: Dict[str, int]) -> int: ...

            invalid3(1, 1, a=1)

    def test_implicit_tag(self):
        @graph
        def sample_graph(x: int, y: int) -> int:
            return x + y

        function_call = next(iter(sample_graph.to_graph().nodes.values()))

        assert isinstance(function_call, FunctionCall)
        self.assertEqual(getattr(function_call.function, "__implicit", False), True)

    def test_graph_default_argument(self):
        @graph
        def sample_graph(x: int, y: int = 1) -> int:
            return x + y

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dict(x=2, y=3))[0], sample_graph(x=2, y=3))

    def test_graph_default_argument_missing(self):
        @graph
        def sample_graph(x: int, y: int = 1) -> int:
            return x + y

        self.assertEqual(self.engine.get(*sample_graph.to_graph().to_dict(x=2))[0], sample_graph(x=2))

    def test_function_default_argument(self):
        @graph
        def add(x: int, y: int = 1) -> int:
            return x + y

        self.assertEqual(self.engine.get(*add.to_graph().to_dict(x=2, y=3))[0], add(x=2, y=3))

    def test_function_default_argument_missing(self):
        @graph
        def add(x: int, y: int = 1) -> int:
            return x + y

        self.assertEqual(self.engine.get(*add.to_graph().to_dict(x=2))[0], add(x=2))

    def test_missing_input(self):
        @graph
        def add(x: int, y: int) -> int:
            return x + y

        with self.assertRaises(ValueError):
            add.to_graph().to_dict(x=2)

    def test_parfun_backend_propagation_in_process(self):
        class DummyBackend:
            def __init__(self, name: str, nested: bool) -> None:
                self.name = name
                self._nested = nested

            def allows_nested_tasks(self) -> bool:
                return self._nested

        active_backend: Dict[str, Any] = {"current": None}
        fake_parfun = types.ModuleType("parfun")

        def get_parallel_backend():
            return active_backend["current"]

        @contextlib.contextmanager
        def set_parallel_backend_context(backend):
            previous = active_backend["current"]
            active_backend["current"] = None if backend == "none" else backend
            try:
                yield
            finally:
                active_backend["current"] = previous

        fake_parfun.get_parallel_backend = get_parallel_backend  # type: ignore[attr-defined]
        fake_parfun.set_parallel_backend_context = set_parallel_backend_context  # type: ignore[attr-defined]

        @delayed
        def read_active_backend_name(x: int) -> str:
            backend = fake_parfun.get_parallel_backend()
            return f"{backend.name if backend is not None else 'none'}:{x}"

        @graph
        def workflow(x: int) -> str:
            return read_active_backend_name(x)

        class InProcessBackend:
            def submit(self, fn, /, *args, **kwargs) -> Future:
                future: Future[Any] = Future()
                future.set_result(fn(*args, **kwargs))
                return future

        in_process_engine = GraphEngine(cast(Backend, InProcessBackend()))
        nested_backend = DummyBackend("nested_ok", True)
        non_nested_backend = DummyBackend("no_nested", False)

        with patch.dict(sys.modules, {"parfun": fake_parfun}):
            with set_parallel_backend_context(nested_backend):
                task_graph, keys = workflow.to_graph().to_dict(x=7)
            self.assertEqual(in_process_engine.get(task_graph, keys)[0], "nested_ok:7")

            with set_parallel_backend_context(non_nested_backend):
                task_graph_non_nested, keys_non_nested = workflow.to_graph().to_dict(x=9)
                self.assertEqual(in_process_engine.get(task_graph_non_nested, keys_non_nested)[0], "none:9")

    @unittest.skipIf(not PARFUN_INSTALLED, "parfun must be installed")
    def test_parfun_backend_propagation_across_workers(self):
        import parfun as pf  # noqa

        @pf.parallel(split=pf.per_argument(values=pf.py_list.by_chunk), combine_with=sum, fixed_partition_size=1)
        def parallel_sum(values: List[int]) -> int:
            return sum(values)

        @delayed
        def sub_graph_task(values: List[int]) -> str:
            backend = pf.get_parallel_backend()
            backend_name = backend.__class__.__name__ if backend is not None else "none"
            return f"{backend_name}:{parallel_sum(values)}"

        @graph
        def main_graph_task(values: List[int]) -> str:
            return sub_graph_task(values)

        with pf.set_parallel_backend_context("scaler_local", n_workers=1):
            task_graph, keys = main_graph_task.to_graph().to_dict(values=[1, 2, 3, 4])
            result = self.engine.get(task_graph, keys)[0]
            self.assertEqual(result, "ScalerLocalBackend:10")

        with pf.set_parallel_backend_context("local_multiprocessing"):
            task_graph, keys = main_graph_task.to_graph().to_dict(values=[1, 2, 3, 4])
            result = self.engine.get(task_graph, keys)[0]
            self.assertEqual(result, "none:10")
