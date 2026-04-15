"""Tests for tr_lbf_addon.lbf_gym — LBF_GYM observation processing and agent management."""

import pytest
import numpy as np
from tr_lbf_addon.lbf_elements import Fruit, Agent
from tr_lbf_addon.lbf_gym import LBF_GYM


# ── Helpers ───────────────────────────────────────────────────────────

def make_observation(field_size=5, players=None, fruit_positions=None):
    """
    Build a mock observation dict matching the full_info_mode format.

    Args:
        field_size: size of the square grid
        players: list of dicts with id, position (tuple), level
        fruit_positions: list of (row, col, level) tuples for fruits
    """
    field = np.zeros((field_size, field_size))

    if fruit_positions is None:
        fruit_positions = [(1, 3, 2)]
    for row, col, level in fruit_positions:
        field[row, col] = level

    if players is None:
        players = [
            {"id": 0, "position": (0, 0), "level": 1},
            {"id": 1, "position": (4, 4), "level": 2},
        ]

    return {"field": field, "player_infos": players}


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_observation():
    """Observation with two players and one fruit on a 5x5 grid."""
    return make_observation()


@pytest.fixture
def gym_instance(simple_observation):
    """LBF_GYM initialized with the simple observation."""
    return LBF_GYM(simple_observation)


@pytest.fixture
def multi_fruit_observation():
    """Observation with two players and three fruits on an 8x8 grid."""
    return make_observation(
        field_size=8,
        players=[
            {"id": 0, "position": (0, 0), "level": 1},
            {"id": 1, "position": (7, 7), "level": 2},
        ],
        fruit_positions=[(2, 3, 1), (4, 5, 2), (6, 1, 3)],
    )


# ── __init__ ──────────────────────────────────────────────────────────

class TestLBFGymInit:
    """Tests for LBF_GYM initialization from an observation."""

    def test_creates_agents(self, gym_instance):
        """LBF_GYM creates one Agent object per player in the observation."""
        assert len(gym_instance.agents) == 2

    def test_creates_fruits(self, gym_instance):
        """LBF_GYM extracts fruit from the field."""
        assert len(gym_instance.fruits) >= 1

    def test_full_info_field_has_negative_players(self, gym_instance):
        """Players are encoded as negative level values in the full_info_field."""
        assert gym_instance.full_info_field[0, 0] == -1  # player 0, level 1
        assert gym_instance.full_info_field[4, 4] == -2  # player 1, level 2


# ── get_full_info_field ───────────────────────────────────────────────

class TestGetFullInfoField:
    """Tests for get_full_info_field — merging players into the field."""

    def test_fruit_values_preserved(self, gym_instance):
        """Fruit cells retain their positive level values."""
        assert gym_instance.full_info_field[1, 3] == 2

    def test_empty_cells_remain_zero(self, gym_instance):
        """Cells with no player or fruit stay 0."""
        assert gym_instance.full_info_field[3, 3] == 0

    def test_multiple_players(self):
        """Each player is placed correctly as a negative value."""
        obs = make_observation(
            players=[
                {"id": 0, "position": (0, 0), "level": 3},
                {"id": 1, "position": (1, 1), "level": 5},
                {"id": 2, "position": (2, 2), "level": 1},
            ]
        )
        gym = LBF_GYM(obs)
        assert gym.full_info_field[0, 0] == -3
        assert gym.full_info_field[1, 1] == -5
        assert gym.full_info_field[2, 2] == -1


# ── get_fruit_infos ───────────────────────────────────────────────────

class TestGetFruitInfos:
    """Tests for extracting fruit positions and free slots from the field."""

    def test_fruit_position_correct(self, gym_instance):
        """Extracted fruit position matches the field."""
        fruit = gym_instance.fruits[0]
        assert np.array_equal(fruit.position, np.array([1, 3]))

    def test_fruit_level_correct(self, gym_instance):
        """Extracted fruit level matches the field value."""
        fruit = gym_instance.fruits[0]
        assert fruit.level == 2

    def test_free_slots_are_empty_cells(self, gym_instance):
        """All free slots around the fruit are cells with value 0 in the field."""
        fruit = gym_instance.fruits[0]
        for slot in fruit.free_slots:
            assert gym_instance.full_info_field[tuple(slot)] == 0

    def test_multiple_fruits(self, multi_fruit_observation):
        """Multiple fruits are all extracted."""
        gym = LBF_GYM(multi_fruit_observation)
        assert len(gym.fruits) == 3

    def test_fruit_adjacent_to_player_has_fewer_free_slots(self):
        """A fruit next to a player has that slot excluded from free_slots."""
        obs = make_observation(
            players=[{"id": 0, "position": (1, 2), "level": 1}],
            fruit_positions=[(1, 3, 2)],
        )
        gym = LBF_GYM(obs)
        fruit = gym.fruits[0]
        # (1,2) is occupied by the player so should not be free
        for slot in fruit.free_slots:
            assert not np.array_equal(slot, np.array([1, 2]))


# ── initialize_agents ────────────────────────────────────────────────

class TestInitializeAgents:
    """Tests for agent initialization from observation player infos."""

    def test_agent_ids(self, gym_instance):
        """Agent IDs match the observation."""
        ids = [a.id for a in gym_instance.agents]
        assert 0 in ids
        assert 1 in ids

    def test_agent_positions(self, gym_instance):
        """Agent positions match the observation."""
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        assert np.array_equal(agent_0.position, np.array([0, 0]))

    def test_agent_levels(self, gym_instance):
        """Agent levels match the observation."""
        agent_1 = [a for a in gym_instance.agents if a.id == 1][0]
        assert agent_1.level == 2

    def test_agents_have_pathfinding_grid(self, gym_instance):
        """Each agent receives a pathfinding grid."""
        for agent in gym_instance.agents:
            assert agent.path_finding_grid is not None
            assert agent.path_finding_grid.shape == (5, 5)


# ── update_observation ────────────────────────────────────────────────

class TestUpdateObservation:
    """Tests for updating the gym with a new observation."""

    def test_agents_positions_updated(self, gym_instance):
        """Agent positions are updated from the new observation."""
        new_obs = make_observation(
            players=[
                {"id": 0, "position": (1, 1), "level": 1},
                {"id": 1, "position": (3, 3), "level": 2},
            ]
        )
        gym_instance.update_observation(new_obs)
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        assert np.array_equal(agent_0.position, np.array([1, 1]))

    def test_round_counter_increments(self, gym_instance):
        """Each update increments the agent's round_counter."""
        new_obs = make_observation()
        gym_instance.update_observation(new_obs)
        for agent in gym_instance.agents:
            assert agent.round_counter == 1

    def test_position_history_grows(self, gym_instance):
        """Position history appends the new position after update."""
        new_obs = make_observation(
            players=[
                {"id": 0, "position": (2, 2), "level": 1},
                {"id": 1, "position": (3, 3), "level": 2},
            ]
        )
        gym_instance.update_observation(new_obs)
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        assert len(agent_0.position_history) == 1
        assert np.array_equal(agent_0.position_history[0], np.array([2, 2]))

    def test_known_fruits_set_after_update(self, gym_instance):
        """Agents have known_fruits set after an update."""
        new_obs = make_observation()
        gym_instance.update_observation(new_obs)
        for agent in gym_instance.agents:
            assert agent.known_fruits is not None

    def test_known_agents_set_after_update(self, gym_instance):
        """Agents have known_agents (excluding self) set after an update."""
        new_obs = make_observation()
        gym_instance.update_observation(new_obs)
        for agent in gym_instance.agents:
            assert agent.known_agents is not None
            assert all(ka["id"] != agent.id for ka in agent.known_agents)

    def test_update_agents_sets_is_loading_from_last_action(self, gym_instance):
        """Agent with last_action=5 (LOAD) gets is_loading=True after update."""
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        agent_0.last_action = np.int64(5)
        gym_instance.update_observation(make_observation())
        assert agent_0.is_loading is True

    def test_update_agents_clears_is_loading_when_not_loading(self, gym_instance):
        """Agent with last_action != 5 gets is_loading=False after update."""
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        agent_0.last_action = np.int64(5)   # was loading
        gym_instance.update_observation(make_observation())
        agent_0.last_action = np.int64(1)   # now walking
        gym_instance.update_observation(make_observation())
        assert agent_0.is_loading is False


# ── create_path_finding_grid ─────────────────────────────────────────

class TestCreatePathfindingGrid:
    """Tests for pathfinding grid creation."""

    def test_fruit_positions_are_obstacles(self, gym_instance):
        """Fruit cells are marked as 0 (obstacle) in the pathfinding grid."""
        agent = gym_instance.agents[0]
        grid = gym_instance.create_path_finding_grid(agent)
        for fruit in gym_instance.fruits:
            assert grid[tuple(fruit.position)] == 0

    def test_empty_cells_are_walkable(self, gym_instance):
        """Cells without fruits or loading agents are 1 (walkable)."""
        agent = gym_instance.agents[0]
        grid = gym_instance.create_path_finding_grid(agent)
        # (3, 0) should be empty and walkable
        assert grid[3, 0] == 1

    def test_loading_neighbor_becomes_obstacle(self):
        """An adjacent agent that is loading creates an obstacle."""
        obs = make_observation(
            players=[
                {"id": 0, "position": (2, 2), "level": 1},
                {"id": 1, "position": (2, 3), "level": 2},  # adjacent to agent 0
            ],
            fruit_positions=[(0, 0, 1)],  # fruit far away
        )
        gym = LBF_GYM(obs)
        # set agent 1 as loading
        agent_1 = [a for a in gym.agents if a.id == 1][0]
        agent_1.is_loading = True
        # re-create grid for agent 0
        agent_0 = [a for a in gym.agents if a.id == 0][0]
        grid = gym.create_path_finding_grid(agent_0)
        # agent 1 is loading so its position should be an obstacle
        assert grid[2, 3] == 0

    def test_grid_shape_matches_field(self, gym_instance):
        """Pathfinding grid has the same shape as the full_info_field."""
        agent = gym_instance.agents[0]
        grid = gym_instance.create_path_finding_grid(agent)
        assert grid.shape == gym_instance.full_info_field.shape


# ── record_ground_truth ───────────────────────────────────────────────

class TestRecordGroundTruth:
    """Tests for record_ground_truth — labelling predictions when fruits are loaded."""

    def _make_prediction(self, agent_id, fruit_pos, round_=0):
        return {
            "round": round_,
            "agent_id": agent_id,
            "trainings_data": np.zeros((1, 5)),
            "prediction": 0.8,
            "ground_truth": None,
            "fruit_pos": np.array(fruit_pos),
        }

    def test_loaded_fruit_labels_adjacent_agent_as_1(self, gym_instance):
        """When a fruit disappears, the agent adjacent to it gets label 1.0."""
        fruit_pos = np.array([1, 3])
        previous_fruits = [Fruit(position=fruit_pos, level=2, free_slots=[np.array([1, 4])])]
        # Place agent 0 adjacent to the fruit
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        agent_0.position = np.array([1, 4])  # adjacent slot
        agent_0.predictions = [self._make_prediction(agent_id=0, fruit_pos=fruit_pos.tolist())]

        # Current fruits: fruit at (1,3) is gone — use empty field
        gym_instance.fruits = []
        gym_instance.record_ground_truth(previous_fruits)

        assert agent_0.predictions[0]["ground_truth"] == 1.0

    def test_loaded_fruit_labels_non_adjacent_agent_as_0(self, gym_instance):
        """When a fruit disappears, a non-adjacent agent gets label 0.0."""
        fruit_pos = np.array([1, 3])
        previous_fruits = [Fruit(position=fruit_pos, level=2, free_slots=[np.array([1, 4])])]
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        agent_0.position = np.array([0, 0])  # far from fruit
        agent_0.predictions = [self._make_prediction(agent_id=0, fruit_pos=fruit_pos.tolist())]

        gym_instance.fruits = []
        gym_instance.record_ground_truth(previous_fruits)

        assert agent_0.predictions[0]["ground_truth"] == 0.0

    def test_present_fruit_not_labelled(self, gym_instance):
        """Predictions for fruits still present in the observation are not labelled."""
        fruit_pos = np.array([1, 3])
        previous_fruits = [Fruit(position=fruit_pos, level=2, free_slots=[np.array([1, 4])])]
        agent_0 = [a for a in gym_instance.agents if a.id == 0][0]
        agent_0.predictions = [self._make_prediction(agent_id=0, fruit_pos=fruit_pos.tolist())]

        # Keep the same fruit in the current state
        gym_instance.fruits = [Fruit(position=fruit_pos.copy(), level=2, free_slots=[])]
        gym_instance.record_ground_truth(previous_fruits)

        assert agent_0.predictions[0]["ground_truth"] is None

    def test_empty_previous_fruits_is_noop(self, gym_instance):
        """record_ground_truth with an empty list does nothing."""
        gym_instance.record_ground_truth([])  # must not raise


# ── Phase 2: food_growth, dead agents, ca_map ────────────────────────

class TestFoodGrowthFilter:
    """Tests for food_growth filtering in get_fruit_infos."""

    def test_unripe_fruit_is_hidden(self):
        """Fruit with food_growth < 1.0 is excluded from lbf_gym.fruits."""
        obs = make_observation(fruit_positions=[(1, 3, 2)])
        gym_inst = LBF_GYM(obs)
        food_growth = {(1, 3): 0.5}
        gym_inst.get_fruit_infos(food_growth)
        assert len(gym_inst.fruits) == 0

    def test_ripe_fruit_is_visible(self):
        """Fruit with food_growth >= 1.0 appears in lbf_gym.fruits."""
        obs = make_observation(fruit_positions=[(1, 3, 2)])
        gym_inst = LBF_GYM(obs)
        food_growth = {(1, 3): 1.0}
        gym_inst.get_fruit_infos(food_growth)
        assert len(gym_inst.fruits) == 1

    def test_mixed_growth_only_shows_ripe(self):
        """Only fruits with growth >= 1.0 are visible when multiple fruits exist."""
        obs = make_observation(
            field_size=8,
            players=[{"id": 0, "position": (0, 0), "level": 1}],
            fruit_positions=[(1, 1, 1), (3, 3, 2), (5, 5, 1)],
        )
        gym_inst = LBF_GYM(obs)
        food_growth = {(1, 1): 1.0, (3, 3): 0.3, (5, 5): 1.0}
        gym_inst.get_fruit_infos(food_growth)
        visible_positions = [tuple(f.position) for f in gym_inst.fruits]
        assert (1, 1) in visible_positions
        assert (5, 5) in visible_positions
        assert (3, 3) not in visible_positions

    def test_no_food_growth_shows_all_fruits(self):
        """When food_growth is None, all fruits in the field are visible."""
        obs = make_observation(
            field_size=8,
            players=[{"id": 0, "position": (0, 0), "level": 1}],
            fruit_positions=[(2, 3, 2), (4, 5, 1)],
        )
        gym_inst = LBF_GYM(obs)
        gym_inst.get_fruit_infos(food_growth=None)
        assert len(gym_inst.fruits) == 2


class TestDeadAgents:
    """Tests for dead agent handling in update_agents and agents_choose_actions."""

    def test_dead_agent_skips_cognition(self):
        """Dead agent's known_fruits is not set during update_agents."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        # First update to give agent known_fruits
        gym_inst.update_agents(obs["player_infos"])
        agent_0 = next(a for a in gym_inst.agents if a.id == 0)
        assert agent_0.known_fruits is not None

        # Reset known_fruits and mark agent 0 as dead
        agent_0.known_fruits = None
        gym_inst.update_agents(obs["player_infos"], dead_agents={0})
        assert agent_0.known_fruits is None  # skipped for dead agent

    def test_dead_agent_position_still_updated(self):
        """Dead agent's position is updated even when cognition is skipped."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        new_obs = make_observation(players=[
            {"id": 0, "position": (2, 2), "level": 1},
            {"id": 1, "position": (3, 3), "level": 2},
        ])
        gym_inst.update_agents(new_obs["player_infos"], dead_agents={0})
        agent_0 = next(a for a in gym_inst.agents if a.id == 0)
        assert np.array_equal(agent_0.position, np.array([2, 2]))

    def test_dead_agents_get_action_zero(self):
        """agents_choose_actions returns 0 for agents in dead_agents."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        gym_inst.update_agents(obs["player_infos"])  # give agents known_fruits
        actions = gym_inst.agents_choose_actions(dead_agents={0})
        agent_0_idx = next(i for i, a in enumerate(gym_inst.agents) if a.id == 0)
        assert actions[agent_0_idx] == np.int64(0)

    def test_alive_agent_not_affected_by_dead_set(self):
        """Alive agents are still updated normally when dead_agents is provided."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        gym_inst.update_agents(obs["player_infos"], dead_agents={0})
        agent_1 = next(a for a in gym_inst.agents if a.id == 1)
        assert agent_1.known_fruits is not None


class TestStationaryStepTracking:
    """Tests for _stationary_steps counter tracking in update_agents."""

    def test_stationary_steps_increments_when_position_unchanged(self):
        """_stationary_steps increments each step the agent does not move."""
        obs = make_observation(players=[{"id": 0, "position": (0, 0), "level": 1},
                                        {"id": 1, "position": (4, 4), "level": 2}])
        gym_inst = LBF_GYM(obs)
        agent_0 = next(a for a in gym_inst.agents if a.id == 0)
        assert agent_0._stationary_steps == 0

        gym_inst.update_agents(obs["player_infos"])  # position unchanged
        assert agent_0._stationary_steps == 1

        gym_inst.update_agents(obs["player_infos"])  # still unchanged
        assert agent_0._stationary_steps == 2

    def test_stationary_steps_resets_when_position_changes(self):
        """_stationary_steps resets to 0 when the agent moves."""
        obs = make_observation(players=[{"id": 0, "position": (0, 0), "level": 1},
                                        {"id": 1, "position": (4, 4), "level": 2}])
        gym_inst = LBF_GYM(obs)
        agent_0 = next(a for a in gym_inst.agents if a.id == 0)

        gym_inst.update_agents(obs["player_infos"])   # step in place → 1
        assert agent_0._stationary_steps == 1

        moved_obs = make_observation(players=[{"id": 0, "position": (0, 1), "level": 1},
                                              {"id": 1, "position": (4, 4), "level": 2}])
        gym_inst.update_agents(moved_obs["player_infos"])  # moved → reset
        assert agent_0._stationary_steps == 0

    def test_stationary_steps_zero_for_dead_agents_still_tracked(self):
        """Dead agents' positions still update, so _stationary_steps is still tracked."""
        obs = make_observation(players=[{"id": 0, "position": (0, 0), "level": 1},
                                        {"id": 1, "position": (4, 4), "level": 2}])
        gym_inst = LBF_GYM(obs)
        agent_0 = next(a for a in gym_inst.agents if a.id == 0)

        gym_inst.update_agents(obs["player_infos"], dead_agents={0})
        assert agent_0._stationary_steps == 1  # position tracking still runs for dead agents


class TestCaMapPathfinding:
    """Tests for ca_map stone-cell obstacles in create_path_finding_grid."""

    def test_stone_cells_are_obstacles(self):
        """Cells with ca_map == 0 (stone) become 0 in the pathfinding grid."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        ca_map = np.ones((5, 5), dtype=np.int8)
        ca_map[3, 3] = 0  # stone at (3, 3)
        agent = gym_inst.agents[0]
        grid = gym_inst.create_path_finding_grid(agent, ca_map)
        assert grid[3, 3] == 0

    def test_grass_cells_remain_walkable(self):
        """Cells with ca_map == 1 (grass) stay walkable if not occupied."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        ca_map = np.ones((5, 5), dtype=np.int8)
        agent = gym_inst.agents[0]
        grid = gym_inst.create_path_finding_grid(agent, ca_map)
        # (3, 0) has no fruit and no loading agent — should be walkable
        assert grid[3, 0] == 1

    def test_ca_map_none_does_not_change_grid(self):
        """Without ca_map, the grid is identical to the no-ca_map case."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        agent = gym_inst.agents[0]
        grid_no_map = gym_inst.create_path_finding_grid(agent, ca_map=None)
        grid_with_none = gym_inst.create_path_finding_grid(agent)
        assert np.array_equal(grid_no_map, grid_with_none)

    def test_ca_map_passed_through_initialize_agents(self):
        """ca_map passed to LBF_GYM.__init__ is applied to initial pathfinding grids."""
        ca_map = np.ones((5, 5), dtype=np.int8)
        ca_map[4, 4] = 0  # stone where agent 1 is NOT (agent 1 is at (4,4) but that's a player pos)
        ca_map[2, 2] = 0  # stone at a free cell
        obs = make_observation()
        gym_inst = LBF_GYM(obs, ca_map=ca_map)
        for agent in gym_inst.agents:
            assert agent.path_finding_grid[2, 2] == 0


class TestAnyFruitLoaded:
    """Tests for the any_fruit_loaded flag — set when a fruit disappears between observations."""

    def test_flag_false_initially(self):
        """any_fruit_loaded starts False after construction."""
        obs = make_observation()
        gym_inst = LBF_GYM(obs)
        assert gym_inst.any_fruit_loaded is False

    def test_flag_set_when_fruit_disappears(self):
        """any_fruit_loaded becomes True when a fruit present in the previous step is gone."""
        obs_with_fruit = make_observation(fruit_positions=[(1, 3, 2)])
        gym_inst = LBF_GYM(obs_with_fruit)

        # Next observation: fruit is gone
        obs_no_fruit = make_observation(fruit_positions=[])
        gym_inst.update_observation(obs_no_fruit)

        assert gym_inst.any_fruit_loaded is True

    def test_flag_false_when_fruit_persists(self):
        """any_fruit_loaded stays False when the same fruit is present in both observations."""
        obs = make_observation(fruit_positions=[(1, 3, 2)])
        gym_inst = LBF_GYM(obs)

        gym_inst.update_observation(obs)

        assert gym_inst.any_fruit_loaded is False

    def test_flag_reset_on_subsequent_step_with_no_load(self):
        """any_fruit_loaded resets to False on the next update if no fruit disappears."""
        obs_with_fruit = make_observation(fruit_positions=[(1, 3, 2)])
        gym_inst = LBF_GYM(obs_with_fruit)

        obs_no_fruit = make_observation(fruit_positions=[])
        gym_inst.update_observation(obs_no_fruit)
        assert gym_inst.any_fruit_loaded is True  # fruit just loaded

        gym_inst.update_observation(obs_no_fruit)  # still no fruit, nothing new loaded
        assert gym_inst.any_fruit_loaded is False
