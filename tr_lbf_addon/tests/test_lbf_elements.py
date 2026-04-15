"""Tests for tr_lbf_addon.lbf_elements — Fruit dataclass and Agent class."""

import pytest
import numpy as np
from tr_lbf_addon.lbf_elements import Fruit, Agent


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_grid():
    """5x5 grid with all ones (no obstacles)."""
    return np.ones((5, 5))


@pytest.fixture
def grid_with_obstacle():
    """5x5 grid with an obstacle at (2, 2)."""
    grid = np.ones((5, 5))
    grid[2, 2] = 0
    return grid


@pytest.fixture
def agent(simple_grid):
    """Agent at (0, 0) with level 2 and a clear 5x5 grid."""
    agent = Agent(id=0, position=np.array([0, 0]), level=2)
    agent.path_finding_grid = simple_grid
    return agent


@pytest.fixture
def fruit_with_free_slots():
    """Fruit at (2, 2), level 1, with four free adjacent slots."""
    return Fruit(
        position=np.array([2, 2]),
        level=1,
        free_slots=[
            np.array([2, 3]),
            np.array([2, 1]),
            np.array([3, 2]),
            np.array([1, 2]),
        ],
    )


@pytest.fixture
def two_agents(simple_grid):
    """Two agents on a 5x5 grid for testing inter-agent logic."""
    agent_a = Agent(id=0, position=np.array([0, 0]), level=2)
    agent_a.path_finding_grid = simple_grid
    agent_a.position_history = [np.array([0, 0])]

    agent_b = Agent(id=1, position=np.array([4, 4]), level=3)
    agent_b.path_finding_grid = simple_grid
    agent_b.position_history = [np.array([4, 4])]

    return agent_a, agent_b


# ── Fruit dataclass ───────────────────────────────────────────────────

class TestFruit:
    """Tests for the Fruit dataclass."""

    def test_fruit_creation(self):
        """Fruit stores position, level, and free_slots correctly."""
        fruit = Fruit(position=np.array([3, 4]), level=2)
        assert np.array_equal(fruit.position, np.array([3, 4]))
        assert fruit.level == 2
        assert fruit.free_slots is None

    def test_fruit_with_free_slots(self, fruit_with_free_slots):
        """Fruit with explicit free_slots stores them."""
        assert len(fruit_with_free_slots.free_slots) == 4
        assert fruit_with_free_slots.level == 1


# ── Agent init and repr ──────────────────────────────────────────────

class TestAgentInit:
    """Tests for Agent initialization and representation."""

    def test_agent_creation(self, agent):
        """Agent initializes with correct id, position, level, and defaults."""
        assert agent.id == 0
        assert np.array_equal(agent.position, np.array([0, 0]))
        assert agent.level == 2
        assert agent.target is None
        assert agent.is_loading is False
        assert agent.round_counter == 0
        assert agent.neural_network is None

    def test_agent_repr(self, agent):
        """__repr__ includes id, position, level, target, and loading status."""
        repr_str = repr(agent)
        assert "id: 0" in repr_str
        assert "is loading: False" in repr_str

    def test_default_is_alive_true(self, agent):
        """New agent defaults to is_alive=True."""
        assert agent.is_alive is True

    def test_default_embedding_and_hidden_dims(self, agent):
        """New agent defaults to embedding_dim=8 and decision_hidden=16."""
        assert agent.embedding_dim == 8
        assert agent.decision_hidden == 16


# ── Survival: dead agent behaviour ────────────────────────────────────

class TestDeadAgent:
    """Tests for agent survival fields and dead-guard behaviour."""

    def test_dead_agent_returns_zero_action(self, agent, fruit_with_free_slots):
        """Agent with is_alive=False returns action 0 without accessing target."""
        agent.is_alive = False
        agent.target = fruit_with_free_slots
        action = agent.choose_next_action()
        assert action == np.int64(0)

    def test_init_neural_network_with_custom_dims(self, agent):
        """Setting embedding_dim and decision_hidden before init_neural_network uses those sizes."""
        agent.embedding_dim = 12
        agent.decision_hidden = 24
        agent.init_neural_network()
        assert agent.neural_network is not None
        assert agent.neural_network.embedding_dim == 12
        assert agent.neural_network.decision_net[0].out_features == 24


# ── Direction and action mapping ─────────────────────────────────────

class TestDirectionMapping:
    """Tests for positional_difference_to_direction and action_string_to_int."""

    @pytest.mark.parametrize("diff, expected_direction", [
        (np.array([-1, 0]), "north"),
        (np.array([1, 0]), "south"),
        (np.array([0, -1]), "west"),
        (np.array([0, 1]), "east"),
        (np.array([0, 0]), "no move"),
        (np.array([1, 1]), "no move"),
    ])
    def test_positional_difference_to_direction(self, agent, diff, expected_direction):
        """Each positional difference maps to the correct direction string."""
        assert agent.positional_difference_to_direction(diff) == expected_direction

    @pytest.mark.parametrize("action_str, expected_int", [
        ("north", np.int64(1)),
        ("south", np.int64(2)),
        ("west", np.int64(3)),
        ("east", np.int64(4)),
        ("load", np.int64(5)),
        ("no move", np.int64(0)),
        ("invalid", np.int64(0)),
    ])
    def test_action_string_to_int(self, agent, action_str, expected_int):
        """Each action string maps to the correct int64 value."""
        assert agent.action_string_to_int(action_str) == expected_int


# ── Pathfinding ──────────────────────────────────────────────────────

class TestPathfinding:
    """Tests for A* pathfinding via get_path."""

    def test_straight_path(self, agent):
        """Agent finds a straight path on an empty grid."""
        path = agent.get_path(np.array([0, 0]), np.array([0, 4]))
        assert len(path) == 5
        assert np.array_equal(path[0], np.array([0, 0]))
        assert np.array_equal(path[-1], np.array([0, 4]))

    def test_path_around_obstacle(self, agent, grid_with_obstacle):
        """Agent routes around an obstacle at (2, 2)."""
        agent.path_finding_grid = grid_with_obstacle
        path = agent.get_path(np.array([2, 0]), np.array([2, 4]))
        # Path must not go through (2, 2)
        for step in path:
            assert not np.array_equal(step, np.array([2, 2]))
        assert np.array_equal(path[0], np.array([2, 0]))
        assert np.array_equal(path[-1], np.array([2, 4]))

    def test_path_same_start_and_end(self, agent):
        """Path from a cell to itself has length 1."""
        path = agent.get_path(np.array([2, 2]), np.array([2, 2]))
        assert len(path) == 1
        assert np.array_equal(path[0], np.array([2, 2]))

    def test_path_only_cardinal_moves(self, agent):
        """Each step in the path is exactly one cardinal move (no diagonals)."""
        path = agent.get_path(np.array([0, 0]), np.array([3, 4]))
        for i in range(len(path) - 1):
            diff = np.abs(path[i + 1] - path[i])
            assert np.sum(diff) == 1, f"Step {i} to {i+1} is not a cardinal move"


# ── choose_next_action ───────────────────────────────────────────────

class TestChooseNextAction:
    """Tests for choose_next_action — walking toward target or loading."""

    def test_load_when_at_target(self, agent, fruit_with_free_slots):
        """Agent returns load action (5) when already at a free slot of the target."""
        agent.position = np.array([2, 3])  # one of fruit's free slots
        agent.target = fruit_with_free_slots
        action = agent.choose_next_action()
        assert action == np.int64(5)

    def test_walk_toward_target(self, agent, fruit_with_free_slots):
        """Agent returns a movement action (1-4) when not at the target."""
        agent.position = np.array([0, 0])
        agent.target = fruit_with_free_slots
        action = agent.choose_next_action()
        assert action in [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]

    def test_walk_sets_path_goal(self, agent, fruit_with_free_slots):
        """choose_next_action sets path_goal to the closest free slot."""
        agent.position = np.array([0, 2])
        agent.target = fruit_with_free_slots
        agent.choose_next_action()
        # closest free slot to (0,2) is (1,2)
        assert np.array_equal(agent.path_goal, np.array([1, 2]))


# ── process_agent_infos ──────────────────────────────────────────────

class TestProcessAgentInfos:
    """Tests for how an agent processes information about other agents."""

    def test_excludes_self(self, two_agents):
        """Agent excludes itself from known_agents."""
        agent_a, agent_b = two_agents
        agent_a.process_agent_infos([agent_a, agent_b])
        assert len(agent_a.known_agents) == 1
        assert agent_a.known_agents[0]["id"] == 1

    def test_stores_agent_info_fields(self, two_agents):
        """Known agent dict contains all expected fields."""
        agent_a, agent_b = two_agents
        agent_a.process_agent_infos([agent_a, agent_b])
        info = agent_a.known_agents[0]
        expected_keys = {"id", "position", "level", "position_history", "last_action", "is_loading"}
        assert set(info.keys()) == expected_keys

    def test_position_history_trimmed_to_memory_size(self, two_agents):
        """Position history is trimmed to agent's memory_size."""
        agent_a, agent_b = two_agents
        agent_b.position_history = [np.array([i, i]) for i in range(10)]
        agent_a.process_agent_infos([agent_a, agent_b])
        assert len(agent_a.known_agents[0]["position_history"]) == agent_a.memory_size


# ── get_possible_coop_level_sums ─────────────────────────────────────

class TestCoopLevelSums:
    """Tests for cooperative level sum calculation."""

    def test_two_agents_same_level(self, agent):
        """Two agents with level 1 produce solo (1) and duo (2) sums."""
        agent.level = 1
        result = agent.get_possible_coop_level_sums([1])
        assert 1 in result  # agent alone
        assert 2 in result  # duo: 1 + 1

    def test_three_agents_mixed_levels(self, agent):
        """Three agents with levels 1, 2, 3 produce all valid cooperative sums."""
        agent.level = 1
        result = agent.get_possible_coop_level_sums([2, 3])
        assert 1 in result  # agent alone
        assert 3 in result  # duo: 1 + 2
        assert 4 in result  # duo: 1 + 3
        assert 5 in result  # duo: 2 + 3
        assert 6 in result  # trio: 1 + 2 + 3

    def test_result_is_sorted(self, agent):
        """Coop level sums are returned sorted."""
        agent.level = 2
        result = agent.get_possible_coop_level_sums([1, 3])
        assert list(result) == sorted(result)

    def test_no_duplicates(self, agent):
        """Coop level sums contain no duplicates."""
        agent.level = 2
        result = agent.get_possible_coop_level_sums([2, 2])
        assert len(result) == len(set(result))


# ── get_next_possible_directions ─────────────────────────────────────

class TestGetNextPossibleDirections:
    """Tests for inferring possible directions from positional difference."""

    def test_north_west(self, agent):
        """Positive row diff + positive col diff gives north and west."""
        result = agent.get_next_possible_directions(np.array([2, 3]))
        assert result == ["north", "west"]

    def test_south_east(self, agent):
        """Negative row diff + negative col diff gives south and east."""
        result = agent.get_next_possible_directions(np.array([-1, -2]))
        assert result == ["south", "east"]

    def test_north_east(self, agent):
        """Positive row diff + negative col diff gives north and east."""
        result = agent.get_next_possible_directions(np.array([3, -1]))
        assert result == ["north", "east"]

    def test_always_returns_two_directions(self, agent):
        """Result always contains exactly two directions."""
        result = agent.get_next_possible_directions(np.array([1, 1]))
        assert len(result) == 2


# ── get_direction_from_other_agent ───────────────────────────────────

class TestGetDirectionFromOtherAgent:
    """Tests for inferring another agent's travel direction from history."""

    def test_moving_south_east(self, agent):
        """Agent moving from (0,0) to (3,3) is heading south-east."""
        history = [np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3])]
        result = agent.get_direction_from_other_agent(history)
        assert "south" in result
        assert "east" in result

    def test_moving_north_west(self, agent):
        """Agent moving from (4,4) to (1,1) is heading north-west."""
        history = [np.array([4, 4]), np.array([3, 3]), np.array([2, 2]), np.array([1, 1])]
        result = agent.get_direction_from_other_agent(history)
        assert "north" in result
        assert "west" in result


# ── init_neural_network ──────────────────────────────────────────────

class TestInitNeuralNetwork:
    """Tests for neural network initialization (requires pytorch)."""

    def test_neural_network_created(self, two_agents):
        """Neural network is created and stored on the agent."""
        agent_a, agent_b = two_agents
        agent_a.process_agent_infos([agent_a, agent_b])
        agent_a.init_neural_network()
        assert agent_a.neural_network is not None

    def test_neural_network_output_shape(self, two_agents):
        """Neural network produces a single output (probability)."""
        import torch
        agent_a, agent_b = two_agents
        agent_a.process_agent_infos([agent_a, agent_b])
        agent_a.init_neural_network()
        fl = torch.rand(1, 1)
        fc = torch.rand(1, 2)
        ot = torch.rand(1, 3, 2)
        with torch.no_grad():
            output = agent_a.neural_network(fl, fc, ot)
        assert output.shape == (1, 1)

    def test_neural_network_output_between_0_and_1(self, two_agents):
        """Sigmoid output is bounded between 0 and 1."""
        import torch
        agent_a, agent_b = two_agents
        agent_a.process_agent_infos([agent_a, agent_b])
        agent_a.init_neural_network()
        fl = torch.rand(1, 1)
        fc = torch.rand(1, 2)
        ot = torch.rand(1, 3, 2)
        with torch.no_grad():
            output = agent_a.neural_network(fl, fc, ot)
        assert 0.0 <= output[0, 0].item() <= 1.0


# ── is_agent_on_predicted_path ────────────────────────────────────────

class TestIsAgentOnPredictedPath:
    """Tests for is_agent_on_predicted_path — path tracking with cursor advancement."""

    def test_no_prediction_returns_false(self, agent):
        """Returns False when no prediction exists for the agent."""
        assert agent.is_agent_on_predicted_path(99, np.array([1, 1])) is False

    def test_position_on_path_returns_true(self, agent):
        """Returns True when the position is found anywhere on the predicted path."""
        path = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        agent.predicted_paths[5] = path
        assert agent.is_agent_on_predicted_path(5, np.array([0, 2])) is True

    def test_position_off_path_returns_false(self, agent):
        """Returns False when the position is not on the predicted path."""
        path = np.array([[0, 0], [0, 1], [0, 2]])
        agent.predicted_paths[5] = path
        assert agent.is_agent_on_predicted_path(5, np.array([3, 3])) is False

    def test_path_trimmed_to_current_position(self, agent):
        """Path is trimmed to start at the found position (cursor advances)."""
        path = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        agent.predicted_paths[5] = path
        agent.is_agent_on_predicted_path(5, np.array([0, 2]))
        remaining = agent.predicted_paths[5]
        assert len(remaining) == 2
        assert np.array_equal(remaining[0], np.array([0, 2]))

    def test_already_at_path_start_returns_true(self, agent):
        """Returns True and keeps full path when position matches the first step."""
        path = np.array([[1, 1], [1, 2], [1, 3]])
        agent.predicted_paths[5] = path
        result = agent.is_agent_on_predicted_path(5, np.array([1, 1]))
        assert result is True
        assert len(agent.predicted_paths[5]) == 3


# ── select_fruit_by_expected_reward ──────────────────────────────────

class TestSelectFruitByExpectedReward:
    """Tests for select_fruit_by_expected_reward — combinatorial expected reward selection."""

    @pytest.fixture
    def agent_with_known(self, simple_grid):
        """Agent with one known other agent, suitable for reward tests."""
        ag = Agent(id=0, position=np.array([0, 0]), level=2)
        ag.path_finding_grid = simple_grid
        ag.known_agents = [{"id": 1, "level": 2, "position": np.array([4, 4])}]
        return ag

    def _make_fruit(self, position, level, num_slots=4):
        slots = [
            np.array([position[0], position[1] + 1]),
            np.array([position[0], position[1] - 1]),
            np.array([position[0] + 1, position[1]]),
            np.array([position[0] - 1, position[1]]),
        ]
        return Fruit(position=np.array(position), level=level, free_slots=slots[:num_slots])

    def test_solo_loadable_fruit_preferred_by_reward(self, agent_with_known):
        """Higher level×level product is preferred among solo-loadable fruits."""
        ag = agent_with_known
        ag.round_counter = 1
        fruit_low = self._make_fruit([1, 1], level=1)
        fruit_high = self._make_fruit([3, 3], level=2)
        result = ag.select_fruit_by_expected_reward([fruit_low, fruit_high])
        assert result is fruit_high  # 2×2=4 > 2×1=2

    def test_returns_none_for_empty_list(self, agent_with_known):
        """Returns None when no fruits are provided."""
        result = agent_with_known.select_fruit_by_expected_reward([])
        assert result is None

    def test_coop_fruit_with_likely_helper_selected(self, agent_with_known):
        """A cooperative fruit is selected when a helper has high predicted probability."""
        ag = agent_with_known
        ag.round_counter = 1
        fruit_coop = self._make_fruit([2, 2], level=3)  # needs level 3, self is 2
        ag.predictions = [{
            "round": 1,
            "agent_id": 1,
            "trainings_data": np.zeros((1, 5)),
            "prediction": 0.95,
            "ground_truth": None,
            "fruit_pos": fruit_coop.position.copy(),
        }]
        result = ag.select_fruit_by_expected_reward([fruit_coop])
        assert result is fruit_coop

    def test_solo_chosen_over_coop_with_unlikely_helper(self, agent_with_known):
        """A solo-loadable fruit is preferred over a cooperative fruit with an unlikely helper."""
        ag = agent_with_known
        ag.round_counter = 1
        fruit_solo = self._make_fruit([1, 1], level=1)     # self.level=2 >= 1
        fruit_coop = self._make_fruit([2, 2], level=4)     # needs level 4, self is 2 + helper 2
        ag.predictions = [{
            "round": 1,
            "agent_id": 1,
            "trainings_data": np.zeros((1, 5)),
            "prediction": 0.05,  # helper very unlikely to show up
            "ground_truth": None,
            "fruit_pos": fruit_coop.position.copy(),
        }]
        result = ag.select_fruit_by_expected_reward([fruit_solo, fruit_coop])
        assert result is fruit_solo


# ── _compute_threshold ────────────────────────────────────────────────

class TestComputeThreshold:
    """Tests for _compute_threshold — dynamic Q1-based threshold."""

    def test_empty_predictions_returns_random_baseline(self, agent):
        """With no predictions, threshold equals 1/num_fruits."""
        assert agent._compute_threshold([], 4) == pytest.approx(0.25)

    def test_threshold_at_least_random_baseline(self, agent):
        """Threshold is always >= 1/num_fruits even when Q1 is very low."""
        probs = [0.01, 0.01, 0.01, 0.01]
        result = agent._compute_threshold(probs, 4)
        assert result >= 0.25

    def test_threshold_uses_q1_when_above_baseline(self, agent):
        """Threshold equals Q1 when Q1 > 1/num_fruits."""
        probs = [0.5, 0.6, 0.7, 0.8]
        q1 = float(np.percentile(probs, 25))
        result = agent._compute_threshold(probs, 4)
        assert result == pytest.approx(q1)


class TestChooseFruitForceReselect:
    """Tests for choose_fruit(force_reselect) — re-selection triggered when a fruit is loaded."""

    def _make_agent_with_target(self, simple_grid, level, target_fruit, other_fruit):
        """Helper: return an agent that already has target_fruit selected."""
        agent = Agent(id=0, position=np.array([0, 0]), level=level)
        agent.path_finding_grid = simple_grid
        agent.known_agents = []
        agent.known_fruits = [target_fruit, other_fruit]
        agent.target = target_fruit
        return agent

    def test_no_reselect_when_target_present_and_no_fruit_loaded(self, simple_grid):
        """Agent keeps its target when the fruit is still on the map and nothing was loaded."""
        target = Fruit(position=np.array([2, 2]), level=1,
                       free_slots=[np.array([2, 1])])
        other = Fruit(position=np.array([4, 4]), level=2,
                      free_slots=[np.array([4, 3])])
        agent = self._make_agent_with_target(simple_grid, level=2,
                                             target_fruit=target, other_fruit=other)

        agent.choose_fruit(force_reselect=False)

        assert np.array_equal(agent.target.position, target.position), (
            "Target should be unchanged when fruit is present and force_reselect is False"
        )

    def test_reselect_triggered_by_force_reselect(self, simple_grid):
        """force_reselect=True causes re-evaluation even if the current target still exists."""
        target = Fruit(position=np.array([2, 2]), level=1,
                       free_slots=[np.array([2, 1])])
        other = Fruit(position=np.array([4, 4]), level=2,
                      free_slots=[np.array([4, 3])])
        agent = self._make_agent_with_target(simple_grid, level=2,
                                             target_fruit=target, other_fruit=other)

        agent.choose_fruit(force_reselect=True)

        # Re-selection ran — a valid target must be set (may or may not be the same fruit)
        assert agent.target is not None

    def test_no_reselect_second_call_without_force(self, simple_grid):
        """A second call with force_reselect=False and same target does not change anything."""
        target = Fruit(position=np.array([2, 2]), level=1,
                       free_slots=[np.array([2, 1])])
        other = Fruit(position=np.array([4, 4]), level=2,
                      free_slots=[np.array([4, 3])])
        agent = self._make_agent_with_target(simple_grid, level=2,
                                             target_fruit=target, other_fruit=other)

        agent.choose_fruit(force_reselect=False)
        agent.choose_fruit(force_reselect=False)

        assert np.array_equal(agent.target.position, target.position)

    def test_force_reselect_defaults_to_false(self, simple_grid):
        """choose_fruit() with no argument behaves the same as force_reselect=False."""
        target = Fruit(position=np.array([2, 2]), level=1,
                       free_slots=[np.array([2, 1])])
        other = Fruit(position=np.array([4, 4]), level=2,
                      free_slots=[np.array([4, 3])])
        agent = self._make_agent_with_target(simple_grid, level=2,
                                             target_fruit=target, other_fruit=other)

        agent.choose_fruit()  # no argument

        assert np.array_equal(agent.target.position, target.position)


class TestChooseFruitStuckReselect:
    """Tests for choose_fruit() stuck-triggered re-evaluation via _stationary_steps."""

    def _make_stuck_agent(self, simple_grid, stationary_steps, target_fruit, other_fruit):
        """Helper: agent with target already set and a given stationary_steps count."""
        from tr_lbf_addon.lbf_elements import _STATIONARY_RESELECT_THRESHOLD
        agent = Agent(id=0, position=np.array([0, 0]), level=2)
        agent.path_finding_grid = simple_grid
        agent.known_agents = []
        agent.known_fruits = [target_fruit, other_fruit]
        agent.target = target_fruit
        agent._stationary_steps = stationary_steps
        return agent

    def test_no_reselect_below_threshold(self, simple_grid):
        """Agent keeps its target when _stationary_steps is below the threshold."""
        from tr_lbf_addon.lbf_elements import _STATIONARY_RESELECT_THRESHOLD
        target = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 1])])
        other  = Fruit(position=np.array([4, 4]), level=2, free_slots=[np.array([4, 3])])
        agent = self._make_stuck_agent(simple_grid,
                                       stationary_steps=_STATIONARY_RESELECT_THRESHOLD - 1,
                                       target_fruit=target, other_fruit=other)

        agent.choose_fruit()

        assert np.array_equal(agent.target.position, target.position)

    def test_reselect_at_threshold(self, simple_grid):
        """choose_fruit() bypasses the early return when _stationary_steps == threshold."""
        from tr_lbf_addon.lbf_elements import _STATIONARY_RESELECT_THRESHOLD
        target = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 1])])
        other  = Fruit(position=np.array([4, 4]), level=2, free_slots=[np.array([4, 3])])
        agent = self._make_stuck_agent(simple_grid,
                                       stationary_steps=_STATIONARY_RESELECT_THRESHOLD,
                                       target_fruit=target, other_fruit=other)

        agent.choose_fruit()

        assert agent.target is not None  # re-evaluated and selected a fruit

    def test_stationary_steps_reset_after_stuck_trigger(self, simple_grid):
        """_stationary_steps resets to 0 when the stuck condition fires."""
        from tr_lbf_addon.lbf_elements import _STATIONARY_RESELECT_THRESHOLD
        target = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 1])])
        other  = Fruit(position=np.array([4, 4]), level=2, free_slots=[np.array([4, 3])])
        agent = self._make_stuck_agent(simple_grid,
                                       stationary_steps=_STATIONARY_RESELECT_THRESHOLD,
                                       target_fruit=target, other_fruit=other)

        agent.choose_fruit()

        assert agent._stationary_steps == 0, (
            "_stationary_steps must reset so the agent gets a fresh window before next re-eval"
        )

    def test_no_reselect_immediately_after_reset(self, simple_grid):
        """After a stuck-triggered reset, the next call does NOT re-evaluate again."""
        from tr_lbf_addon.lbf_elements import _STATIONARY_RESELECT_THRESHOLD
        target = Fruit(position=np.array([2, 2]), level=1, free_slots=[np.array([2, 1])])
        other  = Fruit(position=np.array([4, 4]), level=2, free_slots=[np.array([4, 3])])
        agent = self._make_stuck_agent(simple_grid,
                                       stationary_steps=_STATIONARY_RESELECT_THRESHOLD,
                                       target_fruit=target, other_fruit=other)

        agent.choose_fruit()            # triggers reset, sets target
        original_target = agent.target
        agent.choose_fruit()            # _stationary_steps=0 → early return

        assert agent.target is original_target
