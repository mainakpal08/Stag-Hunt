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

logger = logging.getLogger("griduniverse")
logger.setLevel(logging.INFO) 

class BaseGridUniverseBot(BotBase):
    """A base class for GridUniverse bots that implements experiment
    specific helper functions and runs under Selenium"""

    MEAN_KEY_INTERVAL = 2 #Theerage number of seconds between key presses
    MAX_KEY_INTERVAL = 15   #: The maximum number of seconds between key presses
    END_BUFFER_SECONDS = 30  #: Seconds to wait after expected game end before giving up

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

# class ProbabilisticStagHuntBot(HighPerformanceBaseGridUniverseBot):
#     """A bot that plays griduniverse with a strategy to aid in stag capture."""

#     VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

#     def __init__(self, *args, **kwargs):
#         super(ProbabilisticStagHuntBot, self).__init__(*args, **kwargs)
#         self.id = str(uuid.uuid4())
#         self.player_probabilities = {}  # Tracks probabilities of other players going for the stag
#         self.alpha = 0.8  # Parameter for probability update
#         self.threshold = 0.75  # Threshold for deciding to go for the stag
#         logger.info("Initializing bot...")

#     def client_info(self):
#         return {"id": self.id, "type": "bot"}

#     def initialize_probabilities(self):
#         """Initializes probabilities for all players."""
#         if not self.player_positions:
#             logger.error("Player positions not initialized; cannot initialize probabilities.")
#             return
#         for player_id in self.player_positions:
#             if player_id != self.id:  # Exclude the bot itself
#                 self.player_probabilities[player_id] = 0.5  # Start with equal probability
#         logger.info(f"Initialized probabilities: {self.player_probabilities}")

#     def update_probabilities(self):
#         """Updates probabilities of players going for the stag based on their movements."""
        
#         # Check if there are any animals on the grid
#         if not self.animal_positions:
#             logger.warning("No animal positions found; skipping probability update.")
#             return

#         # Ensure there is at least one stag; otherwise, no need to update probabilities
#         if not any(animal[0] == "stag" for animal in self.animal_positions):
#             logger.warning("No stags found; skipping probability update.")
#             return

#         bot_id = self.get_player_id()  # Get the actual bot ID
#         logger.info(f"Bot ID: {bot_id}")

#         for player_id, position in self.player_positions.items():
#             if str(player_id) == str(bot_id):  # Skip the bot itself
#                 continue

#             # Initialize probability if not already set
#             if player_id not in self.player_probabilities:
#                 self.player_probabilities[player_id] = 0.5  # Start with an equal probability

#             # Get the player's previous position, defaulting to the current position if unknown
#             previous_position = self.previous_player_positions.get(player_id, position)

#             # Compute displacement towards the stag
#             displacement_to_stag = min(
#                 self.manhattan_distance(position, stag[1]) - self.manhattan_distance(previous_position, stag[1])
#                 for stag in self.animal_positions if stag[0] == "stag"
#             )

#             # Compute average displacement towards hares (if any exist) to avoid division by zero
#             hare_positions = [hare[1] for hare in self.animal_positions if hare[0] == "hare"]
#             if hare_positions:
#                 displacement_to_hares = sum(
#                     self.manhattan_distance(position, hare) - self.manhattan_distance(previous_position, hare)
#                     for hare in hare_positions
#                 ) / len(hare_positions)
#             else:
#                 displacement_to_hares = 0  # If no hares exist, set displacement to zero

#             # Calculate rewards based on displacement changes
#             reward_stag = -displacement_to_stag
#             reward_no_stag = -displacement_to_hares

#             # Apply Bayesian probability update
#             likelihood_stag = math.exp(self.alpha * reward_stag)
#             likelihood_no_stag = math.exp(self.alpha * reward_no_stag)

#             prior_stag = self.player_probabilities[player_id]
#             prior_no_stag = 1 - prior_stag

#             posterior_stag = prior_stag * likelihood_stag
#             posterior_no_stag = prior_no_stag * likelihood_no_stag

#             # Normalize probabilities
#             normalization_constant = posterior_stag + posterior_no_stag
#             self.player_probabilities[player_id] = posterior_stag / normalization_constant

#         logger.info(f"Updated probabilities: {self.player_probabilities}")


#     def decide_action(self):
#         """Decides whether to go for the stag based on updated probabilities."""
#         for player_id, probability in self.player_probabilities.items():
#             if probability > self.threshold:
#                 logger.info(f"Deciding to go for stag due to player {player_id} with probability {probability}.")
#                 return True  # Go for the stag if any player exceeds the threshold
#         return False

#     def get_next_key(self):
#         """Decides the next action based on the strategy."""

#         if not self.player_positions or not self.my_position or not self.animal_positions:
#             logger.warning("Missing data (player positions, my position, or animal positions); moving randomly.")
#             return random.choice(self.VALID_KEYS)

#         logger.info(f"My position is {self.my_position}")
#         logger.info(f"All players positions: {self.player_positions}")
#         logger.info(f"Animal positions: {self.animal_positions}")

#         # Update probabilities only if there is available data
#         if self.player_positions and self.animal_positions:
#             logger.info("one..")
#             self.update_probabilities()
#         logger.info("two...")
#         if self.decide_action():
#             logger.info("two...")
#             stag_position = next((pos[1] for pos in self.animal_positions if pos[0] == "stag"), None)
#             if stag_position:
#                 next_move = self.move_towards(self.my_position, stag_position)
#                 logger.info(f"Going for stag at {stag_position}, moving: {next_move}")
#                 return next_move
#             else:
#                 logger.warning("No stag found; moving randomly.")


#         chosen_key = random.choice(self.VALID_KEYS)
#         logger.info("three...")
#         logger.info(f"No decision to go for stag, moving randomly: {repr(chosen_key)}")
#         return chosen_key

#     def manhattan_distance(self, pos1, pos2):
#         """Calculates the Manhattan distance between two positions."""
#         return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#     def move_towards(self, current_position, target_position):
#         """Determines the direction to move toward the target."""
#         if target_position[0] > current_position[0]:
#             return Keys.DOWN
#         elif target_position[0] < current_position[0]:
#             return Keys.UP
#         elif target_position[1] > current_position[1]:
#             return Keys.RIGHT
#         elif target_position[1] < current_position[1]:
#             return Keys.LEFT
#         return Keys.SPACE

class ProbabilisticBot(HighPerformanceBaseGridUniverseBot):
    """" A bot that uses probability to perform actions. """

    #: The Selenium keys that this bot will choose between
    VALID_KEYS = [Keys.UP, Keys.DOWN, Keys.RIGHT, Keys.LEFT, Keys.SPACE]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = str(uuid.uuid4())
        self.player_probabilities = {}  # Tracks probabilities of other players going for the stag
        self.alpha = 0.8  # Parameter for probability update
        self.threshold = 0.75  # Threshold for deciding to go for the stag
        self.previous_player_positions = {}
        self.initialized_probabilities = False
        logger.info("Initializing bot...")
        
    
    def client_info(self):
        return {"id": self.id, "type": "bot"}

    def collect_grid_info(self):
        """Coleta e exibe informações sobre a posição do bot, outros jogadores e animais."""
        logger.info(f"My Position: {self.my_position}")
        logger.info(f"Players Positions: {self.player_positions}")
        logger.info(f"Animal Positions: {self.animal_positions}")

    def initialize_probabilities(self):
        """Initializes the probabilities for all players going for the stag."""
        if not self.player_positions:
            logger.error("Player positions not initialized; cannot initialize probabilities.")
            return

        for player_id, position in self.player_positions.items():
            if player_id == 1:  # Assuming ID 1 is always the bot, skip it
                logger.info(f"Skipping self (Bot ID: {player_id}) at position {position}")
                continue  # Skip the bot itself

            # Initialize the probability only for the other player
            self.player_probabilities[player_id] = 0.5  # You can adjust this value as needed

        logger.info(f"Initialized probabilities: {self.player_probabilities}")
        self.initialized_probabilities = True  # Set the flag to True after initialization



    def update_probabilities(self):
        """Updates the probabilities based on player movements and reward calculation."""
        logger.info("one...")
        if not self.player_positions:
            logger.error("Player positions not initialized; cannot update probabilities.")
            return
        logger.info("two...")

        # Extract positions of stag and hares from animal_positions
        stag_position = None
        hare_positions = []

        for animal_id, position in self.animal_positions:
            if "stag" in str(animal_id).lower():  # Assumption: "stag" in animal_id identifies the stag
                stag_position = position
            elif "hare" in str(animal_id).lower():  # Assumption: "hare" in animal_id identifies a hare
                hare_positions.append(position)

        if not stag_position:
            logger.error("Stag position not found.")
            return
        if len(hare_positions) < 2:
            logger.error("Insufficient hare positions.")
            return

        logger.info(f"Stag position: {stag_position}")
        logger.info(f"Hare positions: {hare_positions}")

        for player_id, position in self.player_positions.items():
            if player_id == 1:  # Skip the bot itself
                logger.info("three... bot skipped")
                continue

            # Get previous position for the player
            previous_position = self.previous_player_positions.get(player_id)
            logger.info(f"four... {previous_position}")
            if previous_position:
                # Calculate the Manhattan distances between previous and current position for stag and hares
                dist_p1_stag_t1 = self.manhattan_distance(position, stag_position)
                dist_p1_hare1_t1 = self.manhattan_distance(position, hare_positions[0])
                dist_p1_hare2_t1 = self.manhattan_distance(position, hare_positions[1])
                
                dist_p1_stag_t0 = self.manhattan_distance(previous_position, stag_position)
                dist_p1_hare1_t0 = self.manhattan_distance(previous_position, hare_positions[0])
                dist_p1_hare2_t0 = self.manhattan_distance(previous_position, hare_positions[1])

                # Calculate rewards for moving towards the stag and hares
                reward_stag = -(dist_p1_stag_t1 - dist_p1_stag_t0)  # Positive if moving closer to stag
                reward_no_stag = -(dist_p1_hare1_t1 - dist_p1_hare1_t0 + dist_p1_hare2_t1 - dist_p1_hare2_t0) / 2

                # Calculate the updated probability using the given formula
                reward = reward_stag
                exp_term = math.exp(self.alpha * reward)
                updated_prob = exp_term / (exp_term + math.exp(self.alpha * -reward))

                # Store the updated probability for the player
                self.player_probabilities[player_id] = updated_prob

            # Store the current position as the previous position for the next update
            self.previous_player_positions[player_id] = position

        # Log the updated probabilities
        logger.info(f"Updated probabilities: {self.player_probabilities}")



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
        """Decides whether to go for the stag based on updated probabilities."""
        for player_id, probability in self.player_probabilities.items():
            if probability > self.threshold:
                logger.info(f"Deciding to go for stag due to player {player_id} with probability {probability}.")
                return True  # Go for the stag if any player exceeds the threshold
        return False

    def get_next_key(self):
        """Decides the next action based on the strategy."""
        # Inicializa as probabilidades na primeira vez
        if not self.initialized_probabilities:
            self.initialize_probabilities()

        # Atualiza as probabilidades com base nas mudanças de posição
        if self.player_positions and self.animal_positions:
            self.update_probabilities()

        # Verifica se deve ir para o stag
        if self.decide_action():
            stag_position = next((pos[1] for pos in self.animal_positions if pos[0] == "stag"), None)
            if stag_position:
                next_move = self.move_towards(self.my_position, stag_position)
                logger.info(f"Going for stag at {stag_position}, moving: {next_move}")
                return next_move
            else:
                logger.warning("No stag found; moving randomly.")

        # Caso contrário, o bot pode escolher uma ação alternativa (como ficar parado ou se mover aleatoriamente)
        logger.info("No action decided, moving randomly.")
        return random.choice(self.VALID_KEYS)




    


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
    # bot_implementation = config.get("bot_policy", "StalkerBot")
    bot_implementation = config.get("bot_policy", "ProbabilisticBot")
    # bot_implementation = config.get("bot_policy", "RandomBot")
    bot_class = globals().get(bot_implementation, None)
    if bot_class and issubclass(bot_class, BotBase):
        return bot_class(*args, **kwargs)
    else:
        raise NotImplementedError
