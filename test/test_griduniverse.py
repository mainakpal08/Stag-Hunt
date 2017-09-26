"""
Tests for `dlgr.griduniverse` module.
"""
import collections
import mock
import pytest
from dallinger.experiments import Griduniverse


class TestDependenciesLoaded(object):

    def test_odo_importable(self):
        import odo
        assert odo is not None

    def test_tablib_importable(self):
        import tablib
        assert tablib is not None


@pytest.mark.usefixtures('env')
class TestExperimentClass(object):

    def test_initialization(self, exp):
        from dallinger.experiment import Experiment
        assert isinstance(exp, Experiment)

    def test_recruiter(self, exp):
        from dallinger.recruiters import Recruiter
        assert isinstance(exp.recruiter, Recruiter)

    def test_has_grid(self, exp):
        from dlgr.griduniverse.experiment import Gridworld
        assert isinstance(exp.grid, Gridworld)

    def test_create_network(self, exp):
        from dallinger.networks import FullyConnected
        net = exp.create_network()
        assert isinstance(net, FullyConnected)

    def test_session(self, exp, db_session):
        # Experiment creates its own session despite being passed one in the
        # __init__() method. Odd?
        assert exp.session is not db_session()

    def test_environment(self, exp):
        exp.environment.network in exp.networks()

    def test_recruit(self, exp):
        exp.recruit()

    def test_handle_connect_creates_node(self, exp, participant):
        exp.handle_connect({'player_id': participant.id})
        assert participant.id in exp.node_by_player_id

    def test_handle_connect_adds_player_to_grid(self, exp, participant):
        exp.handle_connect({'player_id': participant.id})
        assert participant.id in exp.grid.players

    def test_handle_connect_is_noop_for_spectators(self, exp):
        exp.handle_connect({'player_id': 'spectator'})
        assert exp.node_by_player_id == {}

    def test_records_events(self, exp, participant, config):
        exp.handle_connect({'player_id': participant.id})
        exp.send(
            'griduniverse_ctrl:{"type":"move","player_id":1,"move":"left"}'
        )
        data = exp.retrieve_data()
        assert (
            data.infos.dict['details'] ==
            u'{"move": "left", "type": "move", "actual": "left", "player_id": 1}'
        )
        results = self.experiment.average_score(data)
        assert results >= 0.0

    def test_group_donations_distributed_evenly_across_team(self, exp, a):
        donor = a.participant()
        teammate = a.participant()
        exp.handle_connect({'player_id': donor.id})
        exp.handle_connect({'player_id': teammate.id})
        donor_player = exp.grid.players[1]
        teammate_player = exp.grid.players[2]
        # put them on the same team:
        exp.handle_change_color(
            {
                'player_id': teammate_player.id,
                'color': donor_player.color
            }
        )
        # make donation active
        exp.grid.donation_group = True
        exp.grid.donation_amount = 1
        donor_player.score = 2

        exp.handle_donation(
            {
                'donor_id': donor_player.id,
                'recipient_id': 'group:{}'.format(donor_player.color_idx),
                'amount': 2
            }
        )
        assert donor_player.score == 1
        assert teammate_player.score == 1

    def test_public_donations_distributed_evenly_across_players(self, exp, a):
        donor = a.participant()
        opponent = a.participant()
        exp.handle_connect({'player_id': donor.id})
        exp.handle_connect({'player_id': opponent.id})
        donor_player = exp.grid.players[1]
        opponent_player = exp.grid.players[2]
        exp.grid.donation_public = True
        exp.grid.donation_amount = 1
        donor_player.score = 2

        exp.handle_donation(
            {
                'donor_id': donor_player.id,
                'recipient_id': 'all',
                'amount': 2
            }
        )

        assert donor_player.score == 1
        assert opponent_player.score == 1

    def test_colors_distributed_evenly(self, exp, participants):
        exp.grid.num_players = 9
        exp.networks()[0].max_size = 10
        players = [
            exp.handle_connect({'player_id': participant.id}) or exp.grid.players[participant.id]
            for participant in participants[:9]
        ]
        colors = collections.Counter([player.color_idx for player in players])
        assert colors == {0: 3, 1: 3, 2: 3}

    def test_colors_distributed_almost_evenly_if_on_edge(self, exp, participants):
        exp.grid.num_colors = 2
        exp.grid.num_players = 9
        exp.networks()[0].max_size = 10
        players = [
            exp.handle_connect({'player_id': participant.id}) or exp.grid.players[participant.id]
            for participant in participants[:9]
        ]
        colors = collections.Counter([player.color_idx for player in players])
        assert colors == {0: 5, 1: 4}


@pytest.mark.usefixtures('env')
class TestGriduniverseBotExecution(object):

    def test_bot_api(self):
        """Run bots using headless chrome and collect data."""
        self.experiment = Griduniverse()
        data = self.experiment.run(
            mode=u'debug',
            webdriver_type=u'chrome',
            recruiter=u'bots',
            bot_policy=u"AdvantageSeekingBot",
            max_participants=1,
            num_dynos_worker=1,
            time_per_round=10.0,
        )
        results = self.experiment.average_score(data)
        assert results >= 0.0


@pytest.mark.usefixtures('exp_dir', 'env')
class TestCommandline(object):

    @pytest.fixture
    def debugger_unpatched(self, output):
        from dallinger.command_line import DebugSessionRunner
        debugger = DebugSessionRunner(
            output, verbose=True, bot=False, proxy_port=None, exp_config={}
        )
        return debugger

    @pytest.fixture
    def debugger(self, debugger_unpatched):
        from dallinger.heroku.tools import HerokuLocalWrapper
        debugger = debugger_unpatched
        debugger.notify = mock.Mock(return_value=HerokuLocalWrapper.MONITOR_STOP)
        return debugger

    def test_startup(self, debugger):
        debugger.run()
        "Server is running" in str(debugger.out.log.call_args_list[0])

    def test_raises_if_heroku_wont_start(self, debugger):
        mock_wrapper = mock.Mock(
            __enter__=mock.Mock(side_effect=OSError),
            __exit__=mock.Mock(return_value=False)
        )
        with mock.patch('dallinger.command_line.HerokuLocalWrapper') as Wrapper:
            Wrapper.return_value = mock_wrapper
            with pytest.raises(OSError):
                debugger.run()


# @pytest.mark.usefixtures('env_with_home')
# class TestGriduniverse(object):

#     def test_bot_api(self):
#         """Run bots using headless chrome and collect data."""
#         self.experiment = Griduniverse()
#         data = self.experiment.run(
#             mode=u'debug',
#             webdriver_type=u'chrome',
#             recruiter=u'bots',
#             bot_policy=u"AdvantageSeekingBot",
#             max_participants=1,
#             num_dynos_worker=1,
#             time_per_round=20.0,
#         )
#         results = self.experiment.average_score(data)
#         assert results > 0
