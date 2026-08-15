from __future__ import annotations

from asyncio import graph
import unittest as ut
import io
import sys

import gatoh.graphs as gr


class TestGraphSet(ut.TestCase):
    def test_init(self) -> None:
        """
        Test that an empty initialisation of a GraphSet is returning the expected value.
        """
        empty_graphset: gr.GraphSet = gr.GraphSet()
        self.assertIsInstance(
            empty_graphset,
            gr.GraphSet,
            "GraphSet -- Empty initialisation is not returning a GraphSet object",
        )
        self.assertEqual(
            empty_graphset.graphs,
            [],
            "GraphSet -- Empty initialisation is not initialising an empty graphs list",
        )
        self.assertEqual(
            empty_graphset.stochastic_relationships,
            {},
            "GraphSet -- Empty initialisation is not initialising an empty stochastic_relationships dictionary",
        )
        self.assertEqual(
            empty_graphset.stochastic_rels_flags,
            {},
            "GraphSet -- Empty initialisation is not initialising an empty stochastic_rels_flags dictionary",
        )

    def test_add_graph(self) -> None:
        """
        Test that add_graph() is working as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("FooBar", (0.0, 0.0))
        graphset.add_graph(new_graph)
        self.assertEqual(
            len(graphset.graphs),
            1,
            "GraphSet -- add_graph() is not adding the graph object to the graphs list",
        )
        self.assertTrue(
            new_graph in graphset.graphs,
            "GraphSet -- add_graph() is not adding the graph object correctly to the graphs list",
        )

    def test_set_stochastic_rels(self) -> None:
        """
        Test that set_stochastic_rels is working as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        graphset.add_graph(new_graph)
        graphset.set_stochastic_rels(
            "TestGraph",
            True,
            flags=(True, True),
        )
        self.assertTrue(
            graphset.stochastic_relationships["TestGraph"],
            "GraphSet -- set_stochastic_rels is not updating the stochastic relationsips flag for the specified hierarchy",
        )
        self.assertEqual(
            graphset.stochastic_rels_flags["TestGraph"],
            (True, True),
            "GraphSet -- set_stochastic_rels is not updating the formation and disintegration flags for the specified hierarchy",
        )

    def test_set_stochastic_rels_invalid(self) -> None:
        """
        Test that set_stochastic_rels with an invalid hierarchy will produce the expected warning.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        with self.assertWarns(UserWarning, msg="WARNING: attempted to set stochastic relationships for hierarchy FooBar which does not exist in the graphset") as cm:
            graphset.set_stochastic_rels("FooBar", True, flags=(True, True))
        self.assertEqual(
            graphset.stochastic_relationships,
            {},
            "GraphSet -- set_stochastic_rels with an invalid hierarchy is changing a flag in stochastic_relationships",
        )
        self.assertEqual(
            graphset.stochastic_rels_flags,
            {},
            "GraphSet -- set_stochastic_rels with an invalid hierarchy is changing the flags in stochastic_rels_flags",
        )

    def test_graph_at_index_invalid(self) -> None:
        """
        Test that graph_at_index with an invalid index will produce the expected result.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        captured_output = io.StringIO()
        # Redirect the terminal output temporarily
        sys.stdout = captured_output
        graph: gr.Graph | None = graphset.graph_at_index(1312)
        # Ensure the redirect is reset
        sys.stdout = sys.__stdout__
        self.assertEqual(
            captured_output.getvalue(),
            "Index 1312 is out of bounds for the GraphSet. Only 0 social hierarchies have been created.",
            "GraphSet -- graph_at_index with an invalid index is not printing the expected message",
        )
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertIsNone(
            graph,
            "GraphSet -- graph_at_index with an invalid index is not returning None",
        )

    def test_graph_at_index(self) -> None:
        """
        Test that graph_at_index with a valid index will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        graphset.add_graph(new_graph)
        graph: gr.Graph | None = graphset.graph_at_index(0)
        self.assertIsInstance(
            graph,
            gr.Graph,
            "GraphSet -- graph_at_index with a valid index is not returning a Graph object",
        )
        self.assertEqual(
            graph.name,
            new_graph.name,
            "GraphSet -- graph_at_index with a valid index is not returning the correct Graph object",
        )

    def test_graphs_at_indices_invalid(self) -> None:
        """
        Test that graphs_at_indices with all invalid indices will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("1", (0.0, 0.0))
        graphset.add_graph(new_graph)
        # Redirect the output to avoid clutterring the test outputs
        captured_output = io.StringIO()
        sys.stdout = captured_output
        graphs: list[gr.Graph] = graphset.graphs_at_indices([13, 12])
        # Ensure the redirect is reset
        sys.stdout = sys.__stdout__
        # Ensure the captured output is closed
        captured_output.close()
        self.assertEqual(
            graphs,
            [],
            "GraphSet -- graphs_at_indices with all invalid indices is not returning an empty list",
        )

    def test_graphs_at_indices_mixed(self) -> None:
        """
        Test that graphs_at_indices with a mix of valid and invalid indices will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_one: gr.Graph = gr.Graph("1", (0.0, 0.0))
        graphset.add_graph(graph_one)
        graph_two: gr.Graph = gr.Graph("2", (0.0, 0.0))
        graphset.add_graph(graph_two)
        # Redirect the output to avoid clutterring the test outputs
        captured_output = io.StringIO()
        sys.stdout = captured_output
        graphs: list[gr.Graph] = graphset.graphs_at_indices([0, 13, 1, 12])
        # Ensure the redirect is reset
        sys.stdout = sys.__stdout__
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertEqual(
            len(graphs),
            2,
            "GraphSet -- graphs_at_indices with a mix of valid and invalid indices is not returning the expected number of graphs",
        )
        self.assertIn(
            graph_one,
            graphs,
            "GraphSet -- graphs_at_indices with a mix of valid and invalid indices is not returning one or more valid graphs",
        )
        self.assertIn(
            graph_two,
            graphs,
            "GraphSet -- graphs_at_indices with a mix of valid and invalid indices is not returning one or more valid graphs",
        )

    def test_graphs_at_indices(self) -> None:
        """
        Test that graphs_at_indices will all valid indices will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_objects: list[gr.Graph] = []
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graph_objects.append(new_graph)
            graphset.add_graph(new_graph)
        returned_graphs: list[gr.Graph] = graphset.graphs_at_indices([9, 4, 1])
        self.assertEqual(
            len(returned_graphs),
            3,
            "GraphSet -- graphs_at_indices with all valid indices is not returning the expected number of graphs",
        )
        self.assertEqual(
            returned_graphs[0].name,
            graph_objects[9].name,
            "GraphSet -- graphs_at_indices with all valid indices is not returning the correct graphs in order",
        )
        self.assertEqual(
            returned_graphs[1].name,
            graph_objects[4].name,
            "GraphSet -- graphs_at_indices with all valid indices is not returning the correct graphs in order",
        )
        self.assertEqual(
            returned_graphs[2].name,
            graph_objects[1].name,
            "GraphSet -- graphs_at_indices with all valid indices is not returning the correct graphs in order",
        )

    def test_get_hierarchy(self) -> None:
        """
        Test that get_hierarchy with a valid name will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        graphset.add_graph(new_graph)
        second_graph: gr.Graph = gr.Graph("FooBar", (0.0, 0.0))
        graphset.add_graph(second_graph)
        returned_graph: gr.Graph | None = graphset.get_hierarchy("TestGraph")
        self.assertIsInstance(
            returned_graph,
            gr.Graph,
            "GraphSet -- get_hierarchy with a valid hierarchy name is not returning a graph object",
        )
        self.assertEqual(
            returned_graph.name,
            new_graph.name,
            "GraphSet -- get_hierarchy with a valid hierarchy name is not returning the correct graph object",
        )

    def test_get_hierarchy_invalid(self) -> None:
        """
        Test that get_hierarchy with an invalid name will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("TestGraph", (0.0, 0.0))
        graphset.add_graph(new_graph)
        # Capture the terminal output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        returned_graph: gr.Graph | None = graphset.get_hierarchy("Invalid")
        # Ensure the capture is reset
        sys.stdout = sys.__stdout__
        self.assertIsNone(
            returned_graph,
            "GraphSet -- get_hierarchy with an invalid hierarchy name is not returning None",
        )
        self.assertEqual(
            captured_output.getvalue(),
            "No graph representing the social hierarchy 'Invalid' was found...",
            "GraphSet -- get_hierarchy with an invalid hierarchy name is not printing the expected message to the terminal",
        )
        # Ensure the StringIO is closed
        captured_output.close()

    def test_get_hierarchies_invalid(self) -> None:
        """
        Test that get_hierarchies with all invalid names will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        # Temporarily capture the output to not clutter the test output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        returned_graphs: list[gr.Graph] = graphset.get_hierarchies(["Foo", "Bar", "Graph"])
        # Ensure the capture is reset
        sys.stdout = sys.__stdout__
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertEqual(
            returned_graphs,
            [],
            "GraphSet -- get_hierarchies with all invalid names is not returning an empty list",
        )

    def test_get_hierarchies_mixed(self) -> None:
        """
        Test that get_hierarchies with a mix of valid and invalid names will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_one: gr.Graph = gr.Graph("One", (0.0, 0.0))
        graphset.add_graph(graph_one)
        graph_two: gr.Graph = gr.Graph("Two", (0.0, 0.0))
        graphset.add_graph(graph_two)
        # Temporarily capture the output to not clutter the test output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        returned_graphs: list[gr.Graph] = graphset.get_hierarchies(["Foo", "Two", "Bar"])
        # Ensure the capture is reset
        sys.stdout = sys.__stdout__
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertEqual(
            len(returned_graphs),
            1,
            "GraphSet -- get_hierarchies with a mix of valid and invalid names is not returning the expected number of graphs",
        )
        self.assertEqual(
            returned_graphs[0].name,
            "Two",
            "GraphSet -- get_hierarchies with a mix of valid and invalid names is not returning the correct graph objects",
        )

    def test_get_hierarchies_valid(self) -> None:
        """
        Test that get_hierarchies with all valid names will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_objects: list[gr.Graph] = []
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graph_objects.append(new_graph)
            graphset.add_graph(new_graph)
        returned_graphs: list[gr.Graph] = graphset.get_hierarchies(["9", "4"])
        self.assertEqual(
            len(returned_graphs),
            2,
            "GraphSet -- get_hierarchies with all valid names is not returning the expected number of graphs",
        )
        self.assertEqual(
            returned_graphs[0].name,
            graph_objects[9].name,
            "GraphSet -- get_hierarchies with all valid names is not returning the correct graphs in order",
        )
        self.assertEqual(
            returned_graphs[1].name,
            graph_objects[4].name,
            "GraphSet -- get_hierarchies with all valid names is not returning the correct graphs in order",
        )

    def test_get_index_invalid(self) -> None:
        """
        Test that get_index with an invalid name will produce the expected error.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        with self.assertRaises(KeyError, msg="The social hierarchy 'Foo' does not exist in the GraphSet -- cannot return an index.") as cm:
            graph_index: int = graphset.get_index("Foo")

    def test_get_index(self) -> None:
        """
        Test that get_index with a valid name will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_objects: list[gr.Graph] = []
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graph_objects.append(new_graph)
            graphset.add_graph(new_graph)
        graph_index: int = graphset.get_index("4")
        self.assertIsInstance(
            graph_index,
            int,
            "GraphSet -- get_index with a valid name is not returning an int value",
        )
        self.assertEqual(
            graph_index,
            4,
            "GraphSet -- get_index with a valid name is not returning the correct index",
        )

    def test_get_indices_invalid(self) -> None:
        """
        Test that get_indices with all invalid names will produce the expected error.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("Foo", (0.0, 0.0))
        graphset.add_graph(new_graph)
        with self.assertRaises(KeyError, msg="The social hierarchy 'Bar' does not exist in the GraphSet -- cannot return an index.") as cm:
            graph_indices: list[int] = graphset.get_indices(["Bar", "Test", "Graph"])

    def test_get_indices_mixed(self) -> None:
        """
        Test that get_indices with a mix of valid and invalid names will produce the expected error.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        new_graph: gr.Graph = gr.Graph("Foo", (0.0, 0.0))
        graphset.add_graph(new_graph)
        with self.assertRaises(KeyError, msg="The social hierarchy 'Test' does not exist in the GraphSet -- cannot return an index.") as cm:
            graph_indices: list[int] = graphset.get_indices(["Foo", "Test", "Bar"])

    def test_get_indices_valid(self) -> None:
        """
        Test that get_indices with all valid names will work as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        graph_objects: list[gr.Graph] = []
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graph_objects.append(new_graph)
            graphset.add_graph(new_graph)
        graph_indices: list[int] = graphset.get_indices(["8", "6", "4"])
        self.assertIsInstance(
            graph_indices,
            list,
            "GraphSet -- get_indices with all valid names is not returning a list result",
        )
        self.assertEqual(
            len(graph_indices),
            3,
            "GraphSet -- get_indices with all valid names is not returning the expected number of indices",
        )
        self.assertEqual(
            graph_indices[0],
            8,
            "GraphSet -- get_indices with all valid names is not returning the correct graphs in order",
        )
        self.assertEqual(
            graph_indices[1],
            6,
            "GraphSet -- get_indices with all valid names is not returning the correct graphs in order",
        )
        self.assertEqual(
            graph_indices[2],
            4,
            "GraphSet -- get_indices with all valid names is not returning the correct graphs in order",
        )

    def test_list_hierarchies_noprint(self) -> None:
        """
        Test that list_hierarchies with print_out=False is working as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graphset.add_graph(new_graph)
        # Temporarily capture terminal output to verify nothing is printed
        captured_output = io.StringIO()
        sys.stdout = captured_output
        hierarchy_names: list[str] = graphset.list_hierarchies()
        # Ensure the capture is reset
        sys.stdout = sys.__stdout__
        self.assertEqual(
            captured_output.getvalue(),
            "",
            "GraphSet -- list_hierarchies with print_out=False is printing output to the terminal",
        )
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertEqual(
            len(hierarchy_names),
            10,
            "GraphSet -- list_hierarchies (no print) is not listing the correct number of hierarchies in the graphset",
        )
        for i in range(10):
            self.assertEqual(
                hierarchy_names[i],
                f"{i}",
                "GraphSet -- one or more incorrect names are being reported by list_hierarchies (no print)",
            )

    def test_list_hierarchies_print(self) -> None:
        """
        Test that list_hierarchies with print_out=True is working as intended.
        """
        graphset: gr.GraphSet = gr.GraphSet()
        for i in range(10):
            new_graph: gr.Graph = gr.Graph(f"{i}", (0.0, 0.0))
            graphset.add_graph(new_graph)
        # Temporarily capture terminal output to verify the print output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        hierarchy_names: list[str] = graphset.list_hierarchies(print_out=True)
        # Ensure the capture is reset
        sys.stdout = sys.__stdout__
        self.assertEqual(
            captured_output.getvalue(),
            f"\nSocial hierarchies present in the GraphSet:\n\t{hierarchy_names}\n\n",
            "GraphSet -- list_hierarchies with print_out=True is not printing the expected string to the terminal",
        )
        # Ensure the StringIO is closed
        captured_output.close()
        self.assertEqual(
            len(hierarchy_names),
            10,
            "GraphSet -- list_hierarchies (print) is not listing the correct number of hierarchies in the graphset",
        )
        for i in range(10):
            self.assertEqual(
                hierarchy_names[i],
                f"{i}",
                "GraphSet -- one or more incorrect names are being reported by list_hierarchies (print)",
            )
