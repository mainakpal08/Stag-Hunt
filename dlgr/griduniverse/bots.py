"""Griduniverse bots."""
import datetime
import itertools
import json
import logging
import math
import operator
import random
import uuid

import gevent
from dallinger.bots import BotBase, HighPerformanceBotBase
from dallinger.config import get_config
from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from .maze_utils import find_path_astar, maze_to_graph, positions_to_maze
import numpy as np


logger = logging.getLogger("griduniverse")
logger.setLevel(logging.INFO)

class BaseGridUniverseBot(BotBase):
    """A base class for GridUniverse bots that implements experiment
    specific helper functions and runs under Selenium"""

    MEAN_KEY_INTERVAL = 1 #Theerage number of seconds between key presses
    MAX_KEY_INTERVAL = 15   #: The maximum number of seconds between key presses
    END_BUFFER_SECONDS = 120  #: Seconds to wait after expected game end before giving up

    def complete_questionnaire(self):
        """Complete the standard debriefing form randomly."""
        difficulty = Select(self.driver.find_element_by_id("difficulty"))
        difficulty.select_by_value(str(random.randint(1, 7)))
        engagement = Select(self.driver.find_element_by_id("engagement"))
        engagement.select_by_value(str(random.randint(1, 7)))
        try:
            fun = Select(self.driver.find_element_by_id("fun"))
            # This is executed by the IEC_demo.py script...
            # No need to fill out a random value.
            fun.select_by_value(str(0))
        except NoSuchElementException:
            pass
        return True

    def get_wait_time(self):
        """Return a random wait time approximately average to
        MEAN_KEY_INTERVAL but never more than MAX_KEY_INTERVAL"""
        return min(
            random.expovariate(1.0 / self.MEAN_KEY_INTERVAL), self.MAX_KEY_INTERVAL
        )

    def wait_for_grid(self):
        """Blocks until the grid is visible"""
        self.on_grid = True
        return WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.ID, "grid"))
        )

    def get_js_variable(self, variable_name):
        """Return an arbitrary JavaScript variable from the browser"""
        try:
            script = "return window.{};".format(variable_name)
            result = self.driver.execute_script(script)
            if result is None:
                # In some cases (older remote Firefox)
                # we need to use window.wrappedJSObject
                script = "return window.wrappedJSObject.{};".format(variable_name)
                result = self.driver.execute_script(script)
        except WebDriverException:
            result = None

        if result is not None:
            return json.loads(result)

    def observe_state(self):
        """Return the current state the player sees"""
        return self.get_js_variable("state")

    def get_player_id(self):
        """Return the current player's ID number"""
        return str(self.get_js_variable("ego"))

    @property
    def animal_positions(self):
        """Return a list of animal types and their coordinates"""
        try:
            return [
                (item["item_id"], tuple(item["position"]))
                for item in self.state["items"]
            ]
        except (AttributeError, TypeError, KeyError):
            return []

    @property
    def wall_positions(self):
        """Return a list of wall coordinates"""
        try:
            return [tuple(item["position"]) for item in self.state["walls"]]
        except (AttributeError, TypeError, KeyError):
            return []

    @property
    def player_positions(self):
        """Return a dictionary that maps player id to their coordinates"""
        return {player["id"]: player["position"] for player in self.state["players"]}

    @property
    def my_position(self):
        """The position of the current player or None if unknown"""
        player_positions = self.player_positions
        if player_positions and self.player_id in player_positions:
            return player_positions[self.player_id]
        else:
            return None

    @property
    def is_still_on_grid(self):
        """Is the grid currently being displayed"""
        return self.on_grid

    def send_next_key(self, grid):
        """Send the next key due to be sent to the server"""
        # This is a roundabout way of sending the key
        # to the grid element; it's needed to avoid a
        # "cannot focus element" error with chromedriver
        try:
            if self.driver.desired_capabilities["browserName"] == "chrome":
                action = ActionChains(self.driver).move_to_element(grid)
                action.click().send_keys(self.get_next_key()).perform()
            else:
                grid.send_keys(self.get_next_key())
        except StaleElementReferenceException:
            self.on_grid = False

    def participate(self):
        """Participate in the experiment.

        Wait a random amount of time, then send a key according to
        the algorithm above.
        """
        self.wait_for_quorum()
        if self._skip_experiment:
            self.log("Participant overrecruited. Skipping experiment.")
            return True
        self.wait_for_grid()
        self.log("Bot player started")

        # Wait for state to be available
        self.state = None
        self.player_id = None
        while (self.state is None) or (self.player_id is None):
            gevent.sleep(0.500)
            self.state = self.observe_state()
            self.player_id = self.get_player_id()

        # Pick an expected finish time far in the future, it will be updated the first time
        # the bot gets a state
        expected_finish_time = datetime.datetime.now() + datetime.timedelta(days=1)

        while self.is_still_on_grid:
            # The proposed finish time is how many seconds we think remain plus the current time
            proposed_finish_time = datetime.datetime.now() + datetime.timedelta(
                seconds=self.grid["remaining_time"]
            )
            # Update the expected finish time iff it is earlier than we thought
            expected_finish_time = min(expected_finish_time, proposed_finish_time)

            # If we expected to finish more than 30 seconds ago then bail out
            now = datetime.datetime.now()
            if (
                expected_finish_time
                + datetime.timedelta(seconds=self.END_BUFFER_SECONDS)
                < now
            ):
                return True

            gevent.sleep(self.get_wait_time())
            try:
                observed_state = self.observe_state()
                if observed_state:
                    self.state = observed_state
                    self.send_next_key()
                else:
                    return False
            except (StaleElementReferenceException, AttributeError):
                return True

    def get_next_key(self):
        """Classes inheriting from this must override this method to provide the logic to
        determine what their next action should be"""
        raise NotImplementedError

    def get_expected_position(self, key):
        """Predict future state given an action.

        Given the current state of players, if we were to push the key
        specified as a parameter, what would we expect the state to become,
        ignoring modeling of other players' behavior.

        :param key: A one character string, especially from
                    :class:`selenium.webdriver.common.keys.Keys`
        """
        positions = self.player_positions
        my_position = self.my_position
        if my_position is None:
            return positions

        if key == Keys.UP:
            my_position = (my_position[0] - 1, my_position[1])
        elif key == Keys.DOWN:
            my_position = (my_position[0] + 1, my_position[1])
        elif key == Keys.LEFT:
            my_position = (my_position[0], my_position[1] - 1)
        elif key == Keys.RIGHT:
            my_position = (my_position[0], my_position[1] + 1)

        if my_position in self.wall_positions:
            # if the new position is in a wall the movement fails
            my_position = self.my_position
        if my_position in self.player_positions.values():
            # If the other position is occupied by a player we assume it fails, but it may not
            my_position = self.my_position

        positions[self.player_id] = my_position
        return positions

    @staticmethod
    def manhattan_distance(coord1, coord2):
        """Return the manhattan (rectilinear) distance between two coordinates."""
        x = coord1[0] - coord2[0]
        y = coord1[1] - coord2[1]
        return abs(x) + abs(y)

    def translate_directions(self, directions):
        """Convert a string of letters representing cardinal directions
        to a tuple of Selenium arrow keys"""
        lookup = {
            "N": Keys.UP,
            "S": Keys.DOWN,
            "E": Keys.RIGHT,
            "W": Keys.LEFT,
        }
        return tuple(map(lookup.get, directions))

    def distance(self, origin, endpoint):
        """Find the number of unit movements needed to
        travel from origin to endpoint, that is the rectilinear distance
        respecting obstacles as well as a tuple of Selenium keys
        that represent this path.

        In particularly difficult mazes this may return an underestimate
        of the true distance and an approximation of the correct path.

        :param origin: The start position
        :type origin: tuple(int, int)
        :param endpoint: The target position
        :type endpoint: tuple(int, int)
        :return: tuple of distance and directions. Distance is None if no route possible.
        :rtype: tuple(int, list(str)) or tuple(None, list(str))
        """
        try:
            maze = self._maze
            graph = self._graph
        except AttributeError:
            self._maze = maze = positions_to_maze(
                self.wall_positions, self.state["rows"], self.state["columns"]
            )
            self._graph = graph = maze_to_graph(maze)
        result = find_path_astar(
            maze, tuple(origin), tuple(endpoint), max_iterations=10000, graph=graph
        )
        if result:
            distance = result[0]
            directions = self.translate_directions(result[1])
            return distance, directions
        else:
            return None, []

    def distances(self):
        """Compute distances to food.

        Returns a dictionary keyed on player_id, with the value being another
        dictionary which maps the index of a food item in the positions list
        to the distance between that player and that food item.
        """
        distances = {}
        for player_id, position in self.player_positions.items():
            player_distances = {}
            for j, animal in enumerate(self.animal_positions):
                animal_position = animal[1] 
                
                logger.debug(f"Player ID: {player_id}, Player position: {position}, Animal position: {animal_position}")

                player_distances[j], _ = self.distance(position, animal_position)
            distances[player_id] = player_distances
        return distances

class HighPerformanceBaseGridUniverseBot(HighPerformanceBotBase, BaseGridUniverseBot):
    """A parent class for GridUniverse bots that causes them to be run as a HighPerformanceBot,
    i.e. a bot that does not use Selenium but interacts directly over underlying network
    protocols"""

    _quorum_reached = False

    _skip_experiment = False

    def _make_socket(self):
        """Connect to the Redis server and announce the connection"""
        import dallinger.db
        from dallinger.experiment_server.sockets import chat_backend

        self.redis = dallinger.db.redis_conn
        chat_backend.subscribe(self, "griduniverse")

        self.publish({"type": "connect", "player_id": self.participant_id})

    def send(self, message):
        """Redis handler to receive a message from the griduniverse channel to this bot."""
        channel, payload = message.split(":", 1)
        data = json.loads(payload)
        if channel == "quorum":
            handler = "handle_quorum"
        else:
            handler = "handle_{}".format(data["type"])
        getattr(self, handler, lambda x: None)(data)

    def publish(self, message):
        """Sends a message from this bot to the `griduniverse_ctrl` channel."""
        self.redis.publish("griduniverse_ctrl", json.dumps(message))

    def handle_state(self, data):
        """Receive a grid state update an store it"""
        if "grid" in data:
            # grid is a json encoded dictionary, we want to selectively
            # update this rather than overwrite it as not all grid changes
            # are sent each time (such as food and walls)
            data["grid"] = json.loads(data["grid"])
            if "grid" not in self.grid:
                self.grid["grid"] = {}
            self.grid["grid"].update(data["grid"])
            data["grid"] = self.grid["grid"]
        self.grid.update(data)

    def handle_stop(self, data):
        """Receive an update that the round has finished and mark the
        remaining time as zero"""
        self.grid["remaining_time"] = 0

    def handle_quorum(self, data):
        """Update an instance attribute when the quorum is reached, so it
        can be checked in wait_for_quorum().
        """
        if "q" in data and data["q"] == data["n"]:
            self.log("Quorum fulfilled... unleashing bot.")
            self._quorum_reached = True

    @property
    def is_still_on_grid(self):
        """Returns True if the bot is still on an active grid,
        otherwise False"""
        return self.grid.get("remaining_time", 0) > 0.25

    def send_next_key(self):
        """Determines the message to send that corresponds to
        the requested Selenium key, such that the message is the
        same as the one the browser Javascript would have sent"""
        key = self.get_next_key()
        message = {}
        if key == Keys.UP:
            message = {
                "type": "move",
                "player_id": self.participant_id,
                "move": "up",
            }
        elif key == Keys.DOWN:
            message = {
                "type": "move",
                "player_id": self.participant_id,
                "move": "down",
            }
        elif key == Keys.LEFT:
            message = {
                "type": "move",
                "player_id": self.participant_id,
                "move": "left",
            }
        elif key == Keys.RIGHT:
            message = {
                "type": "move",
                "player_id": self.participant_id,
                "move": "right",
            }
        if message:
            self.publish(message)

    def on_signup(self, data):
        """Take any needed action on response from /participant call."""
        super(HighPerformanceBaseGridUniverseBot, self).on_signup(data)
        # We may have been the player to complete the quorum, in which case
        # we won't have to wait for status from the backend.
        if data["quorum"]["n"] == data["quorum"]["q"]:
            self._quorum_reached = True
        # overrecruitment is handled by web ui, so high perf bots need to
        # do that handling here instead.
        if data["participant"]["status"] == "overrecruited":
            self._skip_experiment = True

    def wait_for_quorum(self):
        """Sleep until a quorum of players has signed up.

        The _quorum_reached attribute is set to True by handle_quorum() upon
        learning from the server that we have a quorum.
        """
        while not self._quorum_reached:
            gevent.sleep(0.001)

    def wait_for_grid(self):
        """Sleep until the game grid is up and running.

        handle_state() will update self.grid when game state messages
        are received from the server.
        """
        self.grid = {}
        self._make_socket()
        while True:
            if self.grid and self.grid["remaining_time"]:
                break
            gevent.sleep(0.001)

    def get_js_variable(self, variable_name):
        """Emulate the state of various JS variables that would be present
        in the browser using our accumulated state.

        The only values of variable_name supported are 'state' and 'ego'"""
        if variable_name == "state":
            return self.grid["grid"]
        elif variable_name == "ego":
            return self.participant_id

    def get_player_id(self):
        """Returns the current player's id"""
        return self.participant_id

    @property
    def question_responses(self):
        return {"engagement": 4, "difficulty": 3, "fun": 3}


class RandomBot(HighPerformanceBaseGridUniverseBot):
    """A bot that plays griduniverse randomly"""

    #: The Selenium keys that this bot will choose between
    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inicializa o ID do bot
        self.id = str(uuid.uuid4())
    
    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def get_next_key(self):
        """Randomly press one of Up, Down, Left, Right, space"""

        logger.info(f"My position is {self.my_position}")

        logger.info(f"All players positions: {self.player_positions}")
        logger.info(f"Animal positions: {self.animal_positions}")
        logger.info(f"Distances: {self.distances()}")
        chosen_key = random.choice(self.VALID_KEYS)
        logger.info(f"Chosen key for random bot = {repr(chosen_key)}")
        return chosen_key


class FoodSeekingBot(HighPerformanceBaseGridUniverseBot):
    """A bot that actively tries to increase its score.

    The bot moves towards the closest food.
    """

    def __init__(self, *args, **kwargs):
        super(FoodSeekingBot, self).__init__(*args, **kwargs)
        self.target_coordinates = (None, None)

    def get_logical_targets(self):
        """Find a logical place to move.

        When run on a page view that has data extracted from the grid state
        find the best targets for each of the players, where the best target
        is the closest item of food.
        """
        best_choice = 100e10, None
        position = self.my_position
        if position is None:
            return {}
        for j, food in enumerate(self.food_positions):
            distance, _ = self.distance(position, food)
            if distance and distance < best_choice[0]:
                best_choice = distance, j
        return {self.player_id: best_choice[1]}

    def get_next_key(self):
        """Returns the best key to press in order to maximize point scoring, as follows:

        If there is food on the grid and the bot is not currently making its way
        towards a piece of food, find the logical target and store that coordinate
        as the current target.

        If there is a current target and there is food there, move towards that target
        according to the optimal route, taking walls into account.

        If there is a current target but no food there, unset the target and follow
        the method normally.
        ]
        If there is no food on the grid, move away from other players, such that
        the average distance between players is maximized. This makes it more
        likely that players have an equal chance at finding new food items.

        If there are no actions that get the player nearer to food or increase
        player spread, press a random key."""
        valid_keys = []
        my_position = self.my_position
        try:
            if self.target_coordinates in self.food_positions:
                food_position = self.target_coordinates
            else:
                # If there is a most logical target, we move towards it
                target_id = self.get_logical_targets()[self.player_id]
                food_position = self.food_positions[target_id]
                self.target_coordinates = food_position
        except KeyError:
            # Otherwise, move randomly avoiding walls
            for key in (Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT):
                expected = self.get_expected_position(key)
                if expected != my_position:
                    valid_keys.append(key)
        else:
            baseline_distance, directions = self.distance(my_position, food_position)
            if baseline_distance:
                valid_keys.append(directions[0])
        if not valid_keys:
            # If there are no food items available and no movement would
            # cause the average spread of players to increase, fall back to
            # the behavior of the RandomBot
            valid_keys = RandomBot.VALID_KEYS
        return random.choice(valid_keys)


class AdvantageSeekingBot(HighPerformanceBaseGridUniverseBot):
    """A bot that actively tries to increase its score.

    The bot moves towards the closest food that aren't the logical
    target for another player.
    """

    # def __init__(self, *args, **kwargs):
    #     super(AdvantageSeekingBot, self).__init__(*args, **kwargs)
    #     self.target_coordinates = (None, None)


    def __init__(self, *args, **kwargs):
        super(AdvantageSeekingBot, self).__init__(*args, **kwargs)
        # Initializes bot ID
        self.id = str(uuid.uuid4())
        logger.info("AdvantageSeekingBot ID" + self.id)

    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def get_player_spread(self, positions=None):
        """Mean distance between players.

        When run after populating state data, this returns the mean
        distance between all players on the board, to be used as a heuristic
        for 'spreading out' if there are no logical targets.
        """
        # Allow passing positions in, to calculate the spread of a hypothetical
        # future state, rather than the current state
        if positions is None:
            positions = self.player_positions
        positions = positions.values()
        # Find the distances between all pairs of players
        pairs = itertools.combinations(positions, 2)
        distances = itertools.starmap(self.manhattan_distance, pairs)
        # Calculate and return the mean. distances is an iterator, so we
        # convert it to a tuple so we can more easily do sums on its data
        distances = tuple(distances)
        if distances:
            return float(sum(distances)) / len(distances)
        else:
            # There is only one player, so there are no distances between
            # players.
            return 0

    def get_logical_targets(self):
        """Find a logical place to move.

        When run on a page view that has data extracted from the grid state
        find the best targets for each of the players, where the best target
        is the closest item of food, excluding all food items that are the best
        target for another player. When the same item of food is the closest
        target for multiple players the closest player would get there first,
        so it is excluded as the best target for other players.

        For example:
        Player 1 is 3 spaces from food item 1 and 5 from food item 2.
        Player 2 is 4 spaces from food item 1 and 6 from food item 2.

        The logical targets are:
        Player 1: food item 1
        Player 2: food item 2
        """
        best_choices = {}
        # Create a mapping of (player_id, food_id) tuple to the distance between
        # the relevant player and food item
        for player, food_info in self.distances().items():
            for food_id, distance in food_info.items():
                if distance is None:
                    # This food item is unreachable
                    continue
                best_choices[player, food_id] = distance
        # Sort that list based on the distance, so the closest players/food
        # pairs are first, then discard the distance
        get_key = operator.itemgetter(0)
        get_food_distance = operator.itemgetter(1)
        best_choices = sorted(best_choices.items(), key=get_food_distance)
        best_choices = map(get_key, best_choices)
        # We need to find the optimum solution, so we iterate through the
        # sorted list, discarding pairings that are inferior to previous
        # options. We keep track of player and food ids, once either has been
        # used we know that player or food item has a better choice.
        seen_players = set()
        seen_food = set()
        choices = {}
        for player_id, food_id in best_choices:
            if player_id in seen_players:
                continue
            if food_id in seen_food:
                continue
            seen_players.add(player_id)
            seen_food.add(food_id)
            choices[player_id] = food_id
        return choices

    def get_next_key(self):
        """Returns the best key to press in order to maximize point scoring, as follows:

        If there is food on the grid and the bot is not currently making its way
        towards a piece of food, find the logical target and store that coordinate
        as the current target.

        If there is a current target and there is food there, move towards that target
        according to the optimal route, taking walls into account.

        If there is a current target but no food there, unset the target and follow
        the method normally.
        ]
        If there is no food on the grid, move away from other players, such that
        the average distance between players is maximized. This makes it more
        likely that players have an equal chance at finding new food items.

        If there are no actions that get the player nearer to food or increase
        player spread, press a random key."""
        valid_keys = []
        my_position = self.my_position
        try:
            if self.target_coordinates in self.food_positions:
                food_position = self.target_coordinates
            else:
                # If there is a most logical target, we move towards it
                target_id = self.get_logical_targets()[self.player_id]
                food_position = self.food_positions[target_id]
                self.target_coordinates = food_position
        except KeyError:
            # Otherwise, move in a direction that increases average spread.
            current_spread = self.get_player_spread()
            for key in (Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT):
                expected = self.get_expected_position(key)
                if self.get_player_spread(expected) > current_spread:
                    valid_keys.append(key)
        else:
            baseline_distance, directions = self.distance(my_position, food_position)
            if baseline_distance:
                valid_keys.append(directions[0])
        if not valid_keys:
            # If there are no food items available and no movement would
            # cause the average spread of players to increase, fall back to
            # the behavior of the RandomBot
            valid_keys = RandomBot.VALID_KEYS
        chosen_key = random.choice(valid_keys)
        logger.info("Chosen key = " + chosen_key)
        return chosen_key

class ProbabilisticBot(HighPerformanceBaseGridUniverseBot):
    """" A bot that uses probability to perform actions. This bot expects only 2 hares e 1 stag, and only 1 more player. """

    #: The Selenium keys that this bot will choose between
    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.player_probabilities = {}
        self.alpha = 0.05  
        self.threshold = 0.9
        self.initialized_probabilities = False
        self.is_going_for_stag = False
        self.iterations = 0
        self.previous_player_positions = {}

    
    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def collect_grid_info(self):
        """Shows grid information"""
        logger.info(f"My Position: {self.my_position}")
        logger.info(f"Players Positions: {self.player_positions}")
        logger.info(f"Animal Positions: {self.animal_positions}")

    def initialize_probabilities(self):
        """Initializes the probabilities for all players going for the stag."""
        if not self.player_positions:
            logger.error("Player positions not initialized; cannot initialize probabilities.")
            return

        for player_id, position in self.player_positions.items():
            if player_id == 1: 
                logger.info(f"Skipping self (Bot ID: {player_id}) at position {position}")
                continue  

            self.player_probabilities[player_id] = [1/3, 1/3, 1/3]

        self.initialized_probabilities = True
        logger.info(f"Initialized probabilities: {self.player_probabilities}")

    def update_probabilities(self):
        """Updates the probabilities based on player movements and reward calculation."""
        if not self.player_positions:
            logger.error("Player positions not initialized; cannot update probabilities.")
            return

        # Extract positions of stag and hares from animal_positions
        stag_position = None
        hare_positions = []

        for animal_id, position in self.animal_positions:
            if "stag" in str(animal_id).lower():
                stag_position = position
            elif "hare" in str(animal_id).lower():
                hare_positions.append(position)

        if not stag_position:
            logger.error("Stag position not found.")
            return
        if len(hare_positions) < 2:
            logger.error("Insufficient hare positions.")
            return

        logger.info("-------------------------------------------------------------")
        logger.info(f"Iteration: {self.iterations}")

        for player_id, position in self.player_positions.items():
            if player_id == 1:
                continue

            previous_position = self.previous_player_positions.get(player_id)

            if previous_position is None:
                self.previous_player_positions[player_id] = position
                logger.info(f"Initializing previous position for Player {player_id}. Skipping update.")
                continue

            if previous_position == position:
                logger.info(f"Player {player_id} has not moved. Skipping probability update.")
                continue
            
            logger.info("1....................")

            dist_p1_stag_t1 = self.manhattan_distance(position, stag_position)
            dist_p1_hare1_t1 = self.manhattan_distance(position, hare_positions[0])
            dist_p1_hare2_t1 = self.manhattan_distance(position, hare_positions[1])

            dist_p1_stag_t0 = self.manhattan_distance(previous_position, stag_position)
            dist_p1_hare1_t0 = self.manhattan_distance(previous_position, hare_positions[0])
            dist_p1_hare2_t0 = self.manhattan_distance(previous_position, hare_positions[1])

            delta_stag = dist_p1_stag_t1 - dist_p1_stag_t0
            delta_hare1 = dist_p1_hare1_t1 - dist_p1_hare1_t0
            delta_hare2 = dist_p1_hare2_t1 - dist_p1_hare2_t0

            total_distance = dist_p1_stag_t1 + dist_p1_hare1_t1 + dist_p1_hare2_t1
            stag_percentage = total_distance / dist_p1_stag_t1 if dist_p1_stag_t1 != 0 else total_distance
            hare1_percentage = total_distance / dist_p1_hare1_t1 if dist_p1_hare1_t1 != 0 else total_distance
            hare2_percentage = total_distance / dist_p1_hare2_t1  if dist_p1_hare2_t1 != 0 else total_distance

            reward_stag = - (delta_stag * stag_percentage)
            reward_hare1 = - (delta_hare1 * hare1_percentage)
            reward_hare2 = - (delta_hare2 * hare2_percentage)

            logger.info(f"Reward stag = {reward_stag}, Reward hare 1 = {reward_hare1}, Reward hare 2 = {reward_hare2}")

            exp_stag = math.exp(self.alpha * reward_stag)
            exp_hare1 = math.exp(self.alpha * reward_hare1)
            exp_hare2 = math.exp(self.alpha * reward_hare2)

            lhood_denom = exp_stag + exp_hare1 + exp_hare2 

            if lhood_denom == 0:
                logger.error("Likelihood denominator is zero! Assigning equal probabilities.")
                lhood_stag = lhood_hare1 = lhood_hare2 = 1/3
            else:
                lhood_stag = exp_stag / lhood_denom
                lhood_hare1 = exp_hare1 / lhood_denom
                lhood_hare2 = exp_hare2 / lhood_denom
            
            prior_stag = self.player_probabilities[player_id][0] * lhood_stag
            prior_hare1 = self.player_probabilities[player_id][1] * lhood_hare1
            prior_hare2 = self.player_probabilities[player_id][2] * lhood_hare2


            self.player_probabilities[player_id] = self.normalize([prior_stag, prior_hare1, prior_hare2])
            self.previous_player_positions[player_id] = position

        self.iterations += 1

        logger.info(f"Updated probabilities: {self.player_probabilities}")
        logger.info("-------------------------------------------------------------")


    def normalize(self, arr, epsilon = 0.02):
        total = sum(arr)
        if total == 0:
            return [1/3, 1/3, 1/3]
        else:
            arr = [x / total for x in arr]
        arr = [max(min(x, 1 - epsilon), epsilon) for x in arr]
        total = sum(arr)
        return [x / total for x in arr]

    def move_towards(self, current_position, target_position):
        """Determines the direction to move toward the target."""
        if target_position[0] > current_position[0]:
            return Keys.DOWN
        elif target_position[0] < current_position[0]:
            return Keys.UP
        elif target_position[1] > current_position[1]:
            return Keys.RIGHT
        elif target_position[1] < current_position[1]:
            return Keys.LEFT
        return Keys.SPACE

    def decide_action(self):
        """Decides which target to pursue based on the highest probability."""
        max_probability = 0
        best_target = None
        
        for player_id, probability in self.player_probabilities.items():
            if probability[0] > max_probability:
                max_probability = probability[0]
                best_target = "stag"
            if probability[2] > max_probability:
                max_probability = probability[2]
                best_target = "hare_1"
            if probability[1] > max_probability:
                max_probability = probability[1]
                best_target = "hare_2"
        
        return best_target

    def get_next_key(self):
        """Decides the next action based on the highest probability target."""
        if not self.initialized_probabilities:
            self.initialize_probabilities()

        if self.player_positions and self.animal_positions:
            self.update_probabilities()
            best_target = self.decide_action()

            if best_target == "stag":
                target_position = next((pos[1] for pos in self.animal_positions if pos[0] == "stag"), None)
            elif best_target == "hare_1":
                target_position = self.animal_positions[0][1]
            elif best_target == "hare_2":
                target_position = self.animal_positions[1][1]
            else:
                target_position = None

            if target_position:
                next_move = self.move_towards(self.my_position, target_position)
                logger.info(f"Going for {best_target} at {target_position}, moving: {repr(next_move)}")
                return next_move
            else:
                logger.warning(f"No {best_target} found; moving randomly.")
        
        elif self.is_going_for_stag:
            stag_position = next((pos[1] for pos in self.animal_positions if pos[0] == "stag"), None)
            if stag_position:
                next_move = self.move_towards(self.my_position, stag_position)
                logger.info(f"Continuing towards stag at {stag_position}, moving: {repr(next_move)}")
                return next_move

        return Keys.SPACE

class GeneralizedProbabilisticBot(HighPerformanceBaseGridUniverseBot):
    """" A bot that uses probability to perform actions. """

    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.player_probabilities = {}
        self.alpha = 0.05  
        self.initialized_probabilities = False
        self.iterations = 0
        self.previous_player_positions = {}

    def client_info(self):
        return {"id": self.id, "type": "bot"}
    
    def initialize_probabilities(self):
        """Initializes the probabilities for all players going for each animal."""
        if not self.player_positions or not self.animal_positions:
            logger.error("Player or animal positions not initialized; cannot initialize probabilities.")
            return

        total_animals = len(self.animal_positions)
        logger.info("Len animal positions: " + str(total_animals))
        
        if total_animals == 0:
            logger.error("No animals found; cannot initialize probabilities.")
            return

        initial_prob = 1 / total_animals

        for player_id in self.player_positions:
            if player_id == 1:
                continue
            
            self.player_probabilities[player_id] = []

            for _ in range(total_animals):
                self.player_probabilities[player_id].append(initial_prob)

        self.initialized_probabilities = True
        logger.info(f"Initialized probabilities: {self.player_probabilities}")
    
    def update_probabilities(self):
        """Updates probabilities for each animal based on player movements."""
        if not self.player_positions or not self.animal_positions:
            logger.error("Player or animal positions not initialized; cannot update probabilities.")
            return

        for player_id, position in self.player_positions.items():
            if player_id == 1:
                continue

            previous_position = self.previous_player_positions.get(player_id)
            
            if previous_position is None:
                self.previous_player_positions[player_id] = position
                logger.info(f"Initialized previous position ({position}) for Player {player_id}. Skipping update.")
                continue

            if previous_position == position:
                logger.info(f"Player {player_id} has not moved. Skipping probability update.")
                continue
            
            total_distance = 0
            movement_deltas = []
            current_distance = []

            for _, animal_position in self.animal_positions:
                dist_t1 = self.manhattan_distance(position, animal_position)
                dist_t0 = self.manhattan_distance(previous_position, animal_position)
                movement_deltas.append(dist_t1 - dist_t0)
                current_distance.append(dist_t1)
                total_distance += dist_t1

            logger.info(f"Movement deltas: {movement_deltas}")

            rewards = []
            exponentials = []
            lhood_denom = 0

            for i in range(len(self.animal_positions)):
                distance_factor = total_distance / current_distance[i] if current_distance[i] != 0 else total_distance
                rewards.append(-movement_deltas[i] * distance_factor)
                calc_exponential = math.exp(self.alpha * rewards[i])
                exponentials.append(calc_exponential)
                lhood_denom += calc_exponential

            logger.info(f"Rewards: {rewards}")
            

            new_probabilities = []
            for i in range(len(self.animal_positions)):
                if lhood_denom == 0:
                    likelihood = 1 / len(self.animal_positions)
                else:
                    likelihood = exponentials[i] / lhood_denom
                    prior = self.player_probabilities[player_id][i] * likelihood
                    new_probabilities.append(prior)
            
            self.player_probabilities[player_id] = self.normalize(new_probabilities)
            self.previous_player_positions[player_id] = position
            
        logger.info(f"Updated probabilities: {self.player_probabilities}")

    
    def normalize(self, prob_list, epsilon=0.02):
        total = sum(prob_list)
        if total == 0:
            return [1 / len(prob_list)] * len(prob_list)
        
        prob_list = [v / total for v in prob_list]
        prob_list = [max(min(v, 1 - epsilon), epsilon) for v in prob_list]
        total = sum(prob_list)
        return [v / total for v in prob_list]

    
    def decide_action(self):
        """Decides which animal to pursue based on the highest probability across players."""
        
        # Create a reference list of animals in the order they appear in self.animal_positions
        animal_types = [animal_id for animal_id, _ in self.animal_positions]  
        best_targets = {} 
        
        for player_id, probabilities in self.player_probabilities.items():
            if player_id == 1:
                continue
            
            max_prob = max(probabilities)
            best_index = probabilities.index(max_prob)
            best_targets[player_id] = (animal_types[best_index], best_index)  # (animal type, index)
            logger.info(f"Player {player_id} is going for {animal_types[best_index]} at index {best_index} with probability {max_prob}")

        # Check if any player has a stag as their best option
        stag_targets = [index for player, (animal, index) in best_targets.items() if animal == 'stag']
        hare_targets = [index for player, (animal, index) in best_targets.items() if animal == 'hare']
        
        if stag_targets:  # If any player has a stag as the best target, go to the closest one
            return min(stag_targets, key=lambda idx: self.manhattan_distance(self.my_position, self.animal_positions[idx][1]))
        
        # Otherwise, look for the closest hare that is not already targeted by another player
        available_hares = [idx for idx in range(len(animal_types)) if animal_types[idx] == 'hare' and idx not in hare_targets]
        
        if available_hares:
            logger.info("Available hares: " + str(available_hares))
            return min(available_hares, key=lambda idx: self.manhattan_distance(self.my_position, self.animal_positions[idx][1]))
        
        logger.info("No available hares; choosing the closest among those already targeted.")
        # If no hares are available, choose the closest among those already targeted
        return min(hare_targets, key=lambda idx: self.manhattan_distance(self.my_position, self.animal_positions[idx][1]))
    
    def move_towards(self, current_position, target_position):
        """Determines the direction to move toward the target with variation."""
        options = []
        
        if target_position[0] > current_position[0]:
            options.append(Keys.DOWN)
        if target_position[0] < current_position[0]:
            options.append(Keys.UP)
        if target_position[1] > current_position[1]:
            options.append(Keys.RIGHT)
        if target_position[1] < current_position[1]:
            options.append(Keys.LEFT)
        
        return random.choice(options) if options else Keys.SPACE

    
    def get_next_key(self):
        """Decides the next action based on probabilities."""
        if not self.initialized_probabilities:
            self.initialize_probabilities()

        logger.info("-------------------------------------------------------------")
        logger.info(f"Iteration: {self.iterations}")
        
        self.update_probabilities()
        best_target = self.decide_action()
        target_position = self.animal_positions[best_target][1]
        next_move = self.move_towards(self.my_position, target_position)

        logger.info(f"Going for {best_target} at {target_position}, moving: {repr(next_move)}")
        self.iterations += 1
        
        return next_move
    




class PartialObsGeneralizedProbabilisticBot(HighPerformanceBaseGridUniverseBot):
    """" A bot that uses probability to perform actions. """

    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.player_probabilities = {}
        self.alpha = 0.05  
        self.initialized_probabilities = False
        self.iterations = 0
        self.previous_player_positions = {}

        self.n_col = 0
        self.n_row = 0
        self.visibility = 0
        self.game_config_loaded = False

        self.initialized_belief = False
        self.belief = {}
        self.idx2animal = {}
        self.team_goal = {'hare': 0.5, 'stag': 0.5}
        self.num_stag = 0
        self.num_hare = 0

    def client_info(self):
        return {"id": self.id, "type": "bot"}
    
    def load_game_config(self):
        config = get_config()
        self.n_col = config.get("columns")
        self.n_row = config.get("rows")
        self.visibility = config.get("visibility")
        self.game_config_loaded = True
    
    def initialize_probabilities(self):
        """Initializes the probabilities for all players going for each animal."""
        if not self.player_positions or not self.animal_positions:
            #logger.error("Player or animal positions not initialized; cannot initialize probabilities.")
            return

        total_animals = len(self.animal_positions)
        #logger.info("Len animal positions: " + str(total_animals))
        
        if total_animals == 0:
            #logger.error("No animals found; cannot initialize probabilities.")
            return

        initial_prob = 1 / total_animals

        for player_id in self.player_positions:
            if player_id == 1:
                continue
            
            self.player_probabilities[player_id] = []

            for _ in range(total_animals):
                self.player_probabilities[player_id].append(initial_prob)

        for i in range(len(self.animal_positions)):
            self.idx2animal[i] = self.animal_positions[i][0]
        logger.info(f"idx2animal: {self.idx2animal}")
        self.num_stag = sum([1 for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'stag'])
        self.num_hare = sum([1 for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'hare'])

        self.initialized_probabilities = True
        #logger.info(f"animal_positions: {self.animal_positions}")
        logger.info(f"Initialized probabilities: {self.player_probabilities}")

    def update_probabilities2(self, visible_players):
        if not self.player_positions or not self.animal_positions:
            #logger.error("Player or animal positions not initialized; cannot update probabilities.")
            return
        
        #robot_pos = self.my_position
        obs_player = [(k, v) for k, v in visible_players.items() if v is not None]
        unobs_player = [(k, v) for k, v in visible_players.items() if v is None]
        logger.info(f"obs_player: {obs_player}")
        logger.info(f"unobs_player: {unobs_player}")
        #obs_prob = []
        for player_id, position in obs_player:
            previous_position = self.previous_player_positions.get(player_id)
            if previous_position is None:
                self.previous_player_positions[player_id] = position
                logger.info(f"Initialized previous position ({position}) for Player {player_id}. Skipping update.")
                continue
            if previous_position == position:
                logger.info(f"Player {player_id} has not moved. Skipping probability update.")
                continue

            self.player_probabilities[player_id] = self.infer_intention(position, previous_position, self.player_probabilities[player_id])
            self.previous_player_positions[player_id] = position
        #     obs_prob.append(self.player_probabilities[player_id])
        # if obs_prob:
        #    obs_prob_avg = self.get_avg_probs(obs_prob)
        # else:
        #    obs_prob_avg = self.get_avg_probs(list(self.player_probabilities.values()))
        # stag_probs = sum([obs_prob_avg[i] for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'stag'])# / self.num_stag
        # hare_probs = sum([obs_prob_avg[i] for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'hare'])# / self.num_hare
        # self.team_goal['stag'] = stag_probs #self.normalize([stag_probs, hare_probs])[0]
        # self.team_goal['hare'] = hare_probs #self.normalize([stag_probs, hare_probs])[1]
        # logger.info(f"team_goal: {self.team_goal}")
        for player_id, _ in unobs_player:
            possible_positions = self.get_probable_locations(self.belief[player_id])
            logger.info(f"Player {player_id} is invisible. Possible locations: {possible_positions}")
            previous_position = self.previous_player_positions.get(player_id)
            if previous_position is None:
                self.previous_player_positions[player_id] = random.choice(possible_positions)
                logger.info(f"Initialized probable previous position ({self.previous_player_positions[player_id]}) for Player {player_id}. Skipping update.")
                continue
            # CHECK LATER
            if previous_position in possible_positions:
                logger.info(f"Player {player_id} probably has not moved. Skipping probability update.")
                continue
            probs = []
            prior_prob = self.player_probabilities[player_id]
            for pos in possible_positions:
                probs.append(self.infer_intention(pos, previous_position, prior_prob))
            self.player_probabilities[player_id] = self.get_avg_probs(probs)
            #self.player_probabilities[player_id] = self.adjust_forteam(self.player_probabilities[player_id])
            self.previous_player_positions[player_id] = random.choice(possible_positions)
        logger.info(f"Updated probabilities: {self.player_probabilities}")
        avg_probs = self.get_avg_probs(list(self.player_probabilities.values()))
        self.team_goal['stag'] = sum([avg_probs[i] for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'stag'])
        self.team_goal['hare'] = sum([avg_probs[i] for i in range(len(self.animal_positions)) if self.idx2animal[i] == 'hare'])
        logger.info(f"team_goal: {self.team_goal}")


    def adjust_forteam(self, probs):
        probs = [self.team_goal['stag'] * probs[i] if self.idx2animal[i] == 'stag' else self.team_goal['hare'] * probs[i] for i in range(len(probs))]
        return self.normalize(probs)


    def infer_intention(self, current_position, previous_position, prob):
        total_distance = 0
        movement_deltas = []
        current_distance = []

        for _, animal_position in self.animal_positions:
            dist_t1 = self.manhattan_distance(current_position, animal_position)
            dist_t0 = self.manhattan_distance(previous_position, animal_position)
            movement_deltas.append(dist_t1 - dist_t0)
            current_distance.append(dist_t1)
            total_distance += dist_t1

        #logger.info(f"Movement deltas: {movement_deltas}")

        rewards = []
        exponentials = []
        lhood_denom = 0

        for i in range(len(self.animal_positions)):
            distance_factor = total_distance / current_distance[i] if current_distance[i] != 0 else total_distance
            rewards.append(-movement_deltas[i] * distance_factor)
            calc_exponential = math.exp(self.alpha * rewards[i])
            exponentials.append(calc_exponential)
            lhood_denom += calc_exponential

        #logger.info(f"Rewards: {rewards}")

        new_probabilities = []
        for i in range(len(self.animal_positions)):
            if lhood_denom == 0:
                likelihood = 1 / len(self.animal_positions)
            else:
                likelihood = exponentials[i] / lhood_denom
                prior = prob[i] * likelihood
                new_probabilities.append(prior)
        
        return self.normalize(new_probabilities)

    
    def update_probabilities(self, visible_players):
        """Updates probabilities for each animal based on player movements."""
        if not self.player_positions or not self.animal_positions:
            #logger.error("Player or animal positions not initialized; cannot update probabilities.")
            return

        for player_id, position in visible_players.items():
            if player_id == 1:
                continue

            if position is not None:
                previous_position = self.previous_player_positions.get(player_id)
                
                if previous_position is None:
                    self.previous_player_positions[player_id] = position
                    #logger.info(f"Initialized previous position ({position}) for Player {player_id}. Skipping update.")
                    continue

                if previous_position == position:
                    #logger.info(f"Player {player_id} has not moved. Skipping probability update.")
                    continue
                
                total_distance = 0
                movement_deltas = []
                current_distance = []

                for _, animal_position in self.animal_positions:
                    dist_t1 = self.manhattan_distance(position, animal_position)
                    dist_t0 = self.manhattan_distance(previous_position, animal_position)
                    movement_deltas.append(dist_t1 - dist_t0)
                    current_distance.append(dist_t1)
                    total_distance += dist_t1

                #logger.info(f"Movement deltas: {movement_deltas}")

                rewards = []
                exponentials = []
                lhood_denom = 0

                for i in range(len(self.animal_positions)):
                    distance_factor = total_distance / current_distance[i] if current_distance[i] != 0 else total_distance
                    rewards.append(-movement_deltas[i] * distance_factor)
                    calc_exponential = math.exp(self.alpha * rewards[i])
                    exponentials.append(calc_exponential)
                    lhood_denom += calc_exponential

                #logger.info(f"Rewards: {rewards}")
                

                new_probabilities = []
                for i in range(len(self.animal_positions)):
                    if lhood_denom == 0:
                        likelihood = 1 / len(self.animal_positions)
                    else:
                        likelihood = exponentials[i] / lhood_denom
                        prior = self.player_probabilities[player_id][i] * likelihood
                        new_probabilities.append(prior)
                
                self.player_probabilities[player_id] = self.normalize(new_probabilities)
                self.previous_player_positions[player_id] = position
            
            else:
                possible_positions = self.get_probable_locations(self.belief[player_id])
                logger.info(f"Player {player_id} is invisible. Possible locations: {possible_positions}")
                previous_position = self.previous_player_positions.get(player_id)
                
                if previous_position is None:
                    self.previous_player_positions[player_id] = random.choice(possible_positions)
                    logger.info(f"Initialized probable previous position ({self.previous_player_positions[player_id]}) for Player {player_id}. Skipping update.")
                    continue
                if previous_position in possible_positions:
                    logger.info(f"Player {player_id} probably has not moved. Skipping probability update.")
                    continue
                probs = []
                for pos in possible_positions:                    
                    total_distance = 0
                    movement_deltas = []
                    current_distance = []

                    for _, animal_position in self.animal_positions:
                        dist_t1 = self.manhattan_distance(pos, animal_position)
                        dist_t0 = self.manhattan_distance(previous_position, animal_position)
                        movement_deltas.append(dist_t1 - dist_t0)
                        current_distance.append(dist_t1)
                        total_distance += dist_t1

                    #logger.info(f"Movement deltas: {movement_deltas}")

                    rewards = []
                    exponentials = []
                    lhood_denom = 0

                    for i in range(len(self.animal_positions)):
                        distance_factor = total_distance / current_distance[i] if current_distance[i] != 0 else total_distance
                        rewards.append(-movement_deltas[i] * distance_factor)
                        calc_exponential = math.exp(self.alpha * rewards[i])
                        exponentials.append(calc_exponential)
                        lhood_denom += calc_exponential

                    #logger.info(f"Rewards: {rewards}")
                    

                    new_probabilities = []
                    for i in range(len(self.animal_positions)):
                        if lhood_denom == 0:
                            likelihood = 1 / len(self.animal_positions)
                        else:
                            likelihood = exponentials[i] / lhood_denom
                            prior = self.player_probabilities[player_id][i] * likelihood
                            new_probabilities.append(prior)
                    #logger.info(f"newprobs for player {player_id}: {new_probabilities}")
                    probs.append(new_probabilities)
                if not probs:
                    logger.info(f"Player {player_id} has not probably not moved. Skipping probability update.")
                    continue
                else:
                    #logger.info(f"Probabilities for player {player_id}: {probs}")
                    estimated_probs = self.get_avg_probs(probs)
                    self.player_probabilities[player_id] = self.normalize(estimated_probs)
                    logger.info(f"Probabilities for player {player_id}: {self.player_probabilities[player_id]}")
                    self.previous_player_positions[player_id] = random.choice(possible_positions)

                    
        #logger.info(f"Updated probabilities: {self.player_probabilities}")

    
    def normalize(self, prob_list, epsilon=0.02):
        total = sum(prob_list)
        if total == 0:
            return [1 / len(prob_list)] * len(prob_list)
        
        prob_list = [v / total for v in prob_list]
        prob_list = [max(min(v, 1 - epsilon), epsilon) for v in prob_list]
        total = sum(prob_list)
        return [v / total for v in prob_list]
    

    def get_avg_probs(self, probs):
        averaged = []
        for group in zip(*probs):
            averaged.append(sum(group) / len(group))
        return averaged

    
    def decide_action(self, robot_pos):
        """Decides which animal to pursue based on the highest probability across players."""
        
        # Create a reference list of animals in the order they appear in self.animal_positions
        animal_types = [animal_id for animal_id, _ in self.animal_positions]  
        best_targets = {}
        all_probs_equal = True
        
        for player_id, probabilities in self.player_probabilities.items():
            if player_id == 1:
                continue
            
            max_prob = max(probabilities)
            min_prob = min(probabilities)
            
            if max_prob != min_prob:  
                all_probs_equal = False  
                best_index = probabilities.index(max_prob)
                best_targets[player_id] = (animal_types[best_index], best_index)
                logger.info(f"Player {player_id} is going for {animal_types[best_index]} at index {best_index} with probability {max_prob}")

        if all_probs_equal:
            #logger.info("All probabilities are equal; bot will stay in place.")
            return None
        
        # Check if any player has a stag as their best option
        stag_targets = [index for player, (animal, index) in best_targets.items() if animal == 'stag']
        hare_targets = [index for player, (animal, index) in best_targets.items() if animal == 'hare']
        
        # If team goal for stag is greater than hare, or if the difference is marginal, go for stag
        if self.team_goal['stag'] > self.team_goal['hare'] or abs(self.team_goal['stag'] - self.team_goal['hare']) < 0.10:
            if stag_targets:
                return min(stag_targets, key=lambda idx: self.manhattan_distance(robot_pos, self.animal_positions[idx][1]))
        
        # Otherwise, look for the closest hare that is not already targeted by another player
        available_hares = [idx for idx in range(len(animal_types)) if animal_types[idx] == 'hare' and idx not in hare_targets]
        
        if available_hares:
            #logger.info("Available hares: " + str(available_hares))
            return min(available_hares, key=lambda idx: self.manhattan_distance(robot_pos, self.animal_positions[idx][1]))
        
        #logger.info("No available hares; choosing the closest among those already targeted.")
        # If no hares are available, choose the closest among those already targeted
        return min(hare_targets, key=lambda idx: self.manhattan_distance(robot_pos, self.animal_positions[idx][1]))
    
    def move_towards(self, current_position, target_position):
        """Determines the direction to move toward the target with variation."""
        options = []
        
        if target_position[0] > current_position[0]:
            options.append(Keys.DOWN)
        if target_position[0] < current_position[0]:
            options.append(Keys.UP)
        if target_position[1] > current_position[1]:
            options.append(Keys.RIGHT)
        if target_position[1] < current_position[1]:
            options.append(Keys.LEFT)
        
        return random.choice(options) if options else Keys.SPACE
    
    def get_gaussian_detect_prob(self, d, sigma):
        return np.exp(- (d**2) / (2 * sigma**2))

    def get_observation(self):
        robot_position = self.my_position
        player_positions = {}
        for player_id, pos in self.player_positions.items():
            if player_id == 1:
                continue
            d = self.manhattan_distance(robot_position, pos)
            p_detect = self.get_gaussian_detect_prob(d, self.visibility)
            if np.random.rand() < p_detect:
                player_positions[player_id] = pos
            else:
                player_positions[player_id] = None
        
        logger.info(f"Visible players: {player_positions}")
        return robot_position, player_positions
    
    def init_belief_state(self):
        for player_id, _ in self.player_positions.items():
            if player_id == 1:
                continue
            self.belief[player_id] = np.ones((self.n_row, self.n_col)) / (self.n_row*self.n_col)

        self.adjacency_list = self._compute_adjacency_list()
        
        self.initialized_belief = True
        logger.info(f"Initialized belief state: {self.belief}")

    def _compute_adjacency_list(self):
        adjacency_list = {}
        for r in range(self.n_row):
            for c in range(self.n_col):
                neighbors = []
                for (dr, dc) in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < self.n_row and 0 <= nc < self.n_col:
                        neighbors.append((nr, nc))
                adjacency_list[(r, c)] = neighbors
        return adjacency_list

    def transition_update(self, old_belief):
        new_belief = np.zeros_like(old_belief)
        for r in range(self.n_row):
            for c in range(self.n_col):
                p = old_belief[r, c]
                if p > 1e-12:
                    neighbors = self.adjacency_list[(r, c)]
                    for (nr, nc) in neighbors:
                        new_belief[nr, nc] += p / len(neighbors)
        return new_belief
    
    def observation_update(self, robot_pos, predicted_belief, observed_location):
        updated_belief = predicted_belief.copy()
        
        if observed_location is not None:
            obs_r, obs_c = observed_location
            updated_belief[:] = 0.0
            updated_belief[obs_r, obs_c] = 1.0
            return updated_belief
        
        #robot_pos = self.my_position
        for r in range(self.n_row):
            for c in range(self.n_col):
                d = self.manhattan_distance((r, c), robot_pos)
                p_detect = self.get_gaussian_detect_prob(d, self.visibility)
                updated_belief[r, c] *= (1.0 - p_detect)
        
        total = updated_belief.sum()
        if total > 1e-12:
            updated_belief /= total
        else:
            updated_belief = predicted_belief
        
        return updated_belief
    
    def update_beliefs(self, robot_pos, visible_players):
        for player_id, position in visible_players.items():
            if player_id == 1:
                continue
            self.belief[player_id] = self.transition_update(self.belief[player_id])
            self.belief[player_id] = self.observation_update(robot_pos, self.belief[player_id], position)
            logger.info(f"Probable location of Player {player_id} is: {self.get_probable_locations(self.belief[player_id])}")

    def get_probable_locations(self, matrix):
        max_value = np.max(matrix)
        max_indices = np.argwhere(matrix == max_value)
        max_coordinates = list(map(tuple, max_indices))
        return max_coordinates

    def get_next_key(self):
        """Decides the next action based on probabilities."""
        if self.iterations < 3:
            self.iterations += 1
            return None
        if not self.game_config_loaded:
            self.load_game_config()
        if not self.initialized_belief:
            self.init_belief_state()
        if not self.initialized_probabilities:
            self.initialize_probabilities()

        self.iterations += 1

        logger.info("-------------------------------------------------------------")
        logger.info(f"Iteration: {self.iterations}")
        
        robot_position, visible_players = self.get_observation()
        #no_one_visable = all(value is None for value in visible_players.values())
        self.update_beliefs(robot_position, visible_players)
        # if no_one_visable:
        #     logger.info("No one visible; moving randomly.")
        #     return random.choice([Keys.UP, Keys.DOWN, Keys.LEFT, Keys.RIGHT])
        self.update_probabilities2(visible_players)
        best_target = self.decide_action(robot_position)

        if best_target is None:
            #logger.info("No clear target determined; staying in place.")
            return None

        target_position = self.animal_positions[best_target][1]
        next_move = self.move_towards(robot_position, target_position)

        logger.info(f"Going for {best_target} at {target_position}, moving: {repr(next_move)}")
        
        return next_move
    
        
    


class BayesianStagHunterBot(HighPerformanceBaseGridUniverseBot):
    
    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.beta = 2.5  # Rationality parameter
        self.threshold = 0.5  # Belief threshold
        
        # Prior beliefs P(Intentions)
        self.beliefs = {
            'stag': 1/3,
            'hare1': 1/3,
            'hare2': 1/3
        }
        
        # Track human's previous position
        self.prev_human_pos = None

    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def get_human_position(self):
        """Identify human position from player positions"""
        self.bot_id = self.get_player_id()
        for player_id, pos in self.player_positions.items():
            if player_id != self.bot_id:
                return pos
        return None
    
    def get_animal_positions(self):
        stag_pos = None
        hare_pos = []
        for animal_id, pos in self.animal_positions:
            if str(animal_id).lower() == 'stag':
                stag_pos = pos
            elif str(animal_id).lower() == 'hare':
                hare_pos.append(pos)
        return stag_pos, hare_pos
    
    def normalize(self, arr, epsilon=0.05):
        """Normalizes an array so that its elements sum to 1 and are clamped to [epsilon, 1-epsilon]."""
        total = sum(arr)
        if total == 0:
            arr = [0.5, 0.5]
        else:
            arr = [x / total for x in arr]
        # Clamp each element so that none is lower than epsilon or higher than 1-epsilon
        arr = [max(min(x, 1 - epsilon), epsilon) for x in arr]
        # Renormalize to ensure the sum is 1
        total = sum(arr)
        return [x / total for x in arr]
        

    def update_beliefs(self, stag_pos, hare_pos, human_action, epsilon=0.05):
        likelihoods = {}
        for intention in self.beliefs:
            target_pos = stag_pos if intention == 'stag' else (
                hare_pos[0] if intention == 'hare1' else hare_pos[1])
            old_dist = self.manhattan_distance(self.prev_human_pos, target_pos)
            new_dist = self.manhattan_distance(human_action, target_pos)
            delta_d = old_dist - new_dist
            likelihoods[intention] = math.exp(self.beta * delta_d)

        total = sum(likelihoods.values())
        for intention in likelihoods:
            likelihoods[intention] /= total

        for intention in self.beliefs:
            self.beliefs[intention] *= likelihoods[intention]

        # Normalize beliefs
        total_belief = sum(self.beliefs.values())
        for intention in self.beliefs:
            self.beliefs[intention] = self.beliefs[intention] / total_belief

        # Clamp each belief and renormalize
        for intention in self.beliefs:
            self.beliefs[intention] = max(min(self.beliefs[intention], 1 - epsilon), epsilon)
        total_belief = sum(self.beliefs.values())
        for intention in self.beliefs:
            self.beliefs[intention] /= total_belief

    def get_direction_to_target(self, target_pos):
        """Determine best direction to move towards target"""
        my_x, my_y = self.my_position
        t_x, t_y = target_pos

        if t_x > my_x:
            return Keys.DOWN
        elif t_x < my_x:
            return Keys.UP
        elif t_y > my_y:
            return Keys.RIGHT
        elif t_y < my_y:
            return Keys.LEFT
        return Keys.SPACE

    def get_next_key(self):
        """Main decision function using Bayesian inference"""
        human_pos = self.get_human_position()
        stag_pos, hare_pos = self.get_animal_positions()
        
        # Update beliefs if human has moved
        if human_pos and self.prev_human_pos and human_pos != self.prev_human_pos:
            self.update_beliefs(stag_pos, hare_pos, human_pos)
            logger.info(f"Previous human position: {self.prev_human_pos}")
            logger.info(f"Human moved to: {human_pos}")
            logger.info(f"Stag position: {stag_pos}, Hare positions: {hare_pos}")
            logger.info(f"Beliefs: {self.beliefs}")

        
        # Store current human position for next iteration
        self.prev_human_pos = human_pos
        
        # Decision logic
        if self.beliefs['stag'] >= self.threshold:
            # Move towards stag
            if self.my_position == stag_pos:
                logger.info("Stag reached, pressing SPACE.")
                return Keys.SPACE
            logger.info("Moving towards stag.")
            return self.get_direction_to_target(stag_pos)
        else:
            # Move towards least likely hare
            hare_target = min(['hare1', 'hare2'], key=lambda x: self.beliefs[x])
            hare_2go = hare_pos[0] if hare_target == 'hare1' \
                      else hare_pos[1]
            
            if self.my_position == hare_2go:
                logger.info("Hare reached, pressing SPACE.")
                return Keys.SPACE
            logger.info(f"Moving towards {hare_target}.")
            return self.get_direction_to_target(hare_2go)

    


class StalkerBot(HighPerformanceBaseGridUniverseBot):
    """A bot that follows the closest player"""

    #: The Selenium keys that this bot will choose between
    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        logger.info("Initializing bot...")
    
    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def get_furthest_player_position(self):
        """Find the furthest player's position."""
        my_position = self.my_position  # The bot's current position
        furthest_player = None
        furthest_distance = -float("inf")  # Usar "-inf" para comparar a maior distância

        logger.info(f"My position: {my_position}")
        logger.info(f"Player positions: {self.player_positions}")
        bot_id = self.get_player_id()
        logger.info(f"Bot ID:{bot_id}")

        for player_id, position in self.player_positions.items():
            if str(player_id) == str(bot_id): #trings para ignorar o bot
                logger.info(f"Skipping self (Bot ID: {self.id}) at position {position}")
                continue  # Skip the bot itself

            # Calculate Manhattan distance
            distance = abs(my_position[0] - position[0]) + abs(my_position[1] - position[1])
            logger.info(f"Checking player {player_id} at position {position} with distance {distance}")

            if distance > furthest_distance:  # Verificar maior distância
                furthest_distance = distance
                furthest_player = position
                logger.info(f"New furthest player: {player_id} at position {position} with distance {distance}")

        logger.info(f"Furthest player determined: {furthest_player}")
        return furthest_player




    def get_next_key(self):
        """Move towards the furthest player."""
        furthest_player = self.get_furthest_player_position()

        if not furthest_player:
            logger.info("No players found, staying idle or picking a random key.")
            return random.choice(self.VALID_KEYS)

        my_position = self.my_position

        logger.info(f"My position: {my_position}")
        logger.info(f"Furthest player position: {furthest_player}")

        # Determine the direction to move to get closer to the furthest player
        if furthest_player[0] > my_position[0]:
            chosen_key = Keys.DOWN  # Move down
        elif furthest_player[0] < my_position[0]:
            chosen_key = Keys.UP  # Move up
        elif furthest_player[1] > my_position[1]:
            chosen_key = Keys.RIGHT  # Move right
        elif furthest_player[1] < my_position[1]:
            chosen_key = Keys.LEFT  # Move left
        else:
            chosen_key = Keys.SPACE  # No movement needed

        logger.info(f"Chosen key for bot = {repr(chosen_key)}")
        return chosen_key








def Bot(*args, **kwargs):
    """Pick a bot implementation based on a configuration parameter.

    This can be set in config.txt in this directory or by environment variable.
    """

    config = get_config()
    bot_implementation = config.get("bot_policy", "ProbabilisticBot")
    bot_class = globals().get(bot_implementation, None)
    if bot_class and issubclass(bot_class, BotBase):
        return bot_class(*args, **kwargs)
    else:
        raise NotImplementedError