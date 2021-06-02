from collections.abc import Iterable
import copy
import re

from onmt.rotowire import RotowireConfig
from onmt.utils.logging import logger
from onmt.rotowire.exceptions import (
    MissingPlayer,
    UnderspecifiedTemplateError,
    ElaborationSpecificationError,
    SecondMatchGroupError,
    UnexpectedPlayerData,
    UnexpectedTeamData
)


class TemplateFile:
    def __init__(self, filename, config=None, dynamic=False, sep='<sep>'):
        self.dynamic = dynamic
        self.config = config
        if self.config is None:
            logger.info('Loading default config.')
            self.config = RotowireConfig.from_defaults()

        self.sep = sep
        self.filename = filename
        with open(filename, mode="r", encoding='utf8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    @property
    def static(self):
        return not self.dynamic

    def instantiate_template_from_game(self, tidx, raw_data):
        """
        tidx: index of the template to use
        game: raw data containing all info from the game
        """

        if self.static:
            return TemplatePlan(self.lines, raw_data, config=self.config)

        return TemplatePlan(
            [s.strip() for s in self.lines[tidx].split(self.sep)],
            raw_data=raw_data, config=self.config
        )


class TemplatePlan:
    elab_pattern = re.compile('^<[a-z]+>')
    ent_pattern = re.compile(r'^(team|player)(\[(([a-zA-Z]+=[a-zA-Z0-9]+)(,\s*)?)+\])')

    def __init__(self, sentences, raw_data, config=None):
        self.config = config
        if self.config is None:
            logger.info('Loading default config.')
            self.config = RotowireConfig.from_defaults()

        self.game = Game(raw_data)
        self.sentences = [self.read_line(idx, sentence)
                          for idx, sentence in enumerate(sentences)]

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        yield from self.sentences

    def __getitem__(self, item):
        return self.sentences[item]

    def read_line(self, idx, line):
        elab = self.elab_pattern.search(line)
        if elab is None:
            raise ElaborationSpecificationError('No elaboration given.')

        elab = elab.group()
        if elab not in self.config.elaboration_vocab:
            raise ElaborationSpecificationError(f'Unknown elaboration: {elab}')

        # remove elaboration & split at entities & get entity indices
        entities = [self.find_entity(e) for e in line[len(elab):].split('|')]

        return elab, entities

    def find_entity(self, entity):
        # Split entity into 'item' from: [team] | [player] | [team, player]
        items = [item.strip() for item in entity.split('>')]
        first_match = self.ent_pattern.search(items[0])

        players = self.game.players
        team = None
        if first_match.group(1) == 'team':
            specifiers = [s.strip().split('=') for s in first_match.group(2)[1:-1].split(',')]
            for _team in self.game.teams:
                if all(_team[k] == v for k, v in specifiers):
                    team = _team
                    break
            if team is None:
                raise UnderspecifiedTemplateError('No team satisfies all specifiers')

            if len(items) == 1:
                return team.idx

            players = team.players

        second_match = self.ent_pattern.search(items[-1])
        if second_match.group(1) != 'player':
            raise SecondMatchGroupError(entity, second_match.group(1))
        specifiers = [s.strip().split('=') for s in second_match.group(2)[1:-1].split(',')]
        for key, value in specifiers:
            players = players.filter_by(key, value)
        if len(players) == 1:
            return players[0].idx
        raise UnderspecifiedTemplateError(entity)


class Team:
    numerical_attrs = ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TOV',
                       'PF', 'OREB', 'DREB', 'MIN_CEIL', 'MIN_FLOOR', 'FGM',
                       'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'FG_PCT',
                       'FG3_PCT', 'FT_PCT', 'WINS', 'LOSSES', 'GAME_NUMBER',
                       'ATTENDANCE']

    litteral_attrs = ['FULL_NAME', 'TEAM_NAME', 'TEAM_PLACE', 'STADIUM',
                      'STADIUM_LOCATION', 'DIVISION', 'CONFERENCE', 'DAY_NAME',
                      'MONTH_NAME', 'DAY_OF_MONTH']

    boolean_attrs = ['HOME', 'WINNER']

    def __init__(self, raw_data, idx, players=None):
        self.raw_data = raw_data  # cloning purposes
        self.idx = idx

        _raw_data = copy.deepcopy(self.raw_data)
        self.attrs = self.load_attrs(_raw_data)
        self.attrs = {key: str(val) for key, val in self.attrs.items()}
        self.players = PlayerList(players)

        # sanity check
        assert all(p['TEAM_NAME'] == self['TEAM_NAME'] for p in self.players)

    def clone(self):
        return Team(self.raw_data, self.idx, self.players)

    def load_attrs(self, data):
        attrs = dict()
        for nattr in self.numerical_attrs:
            v = data.pop(nattr.upper())
            attrs[nattr.lower()] = -1 if v == 'N/A' else int(v)
        for lattr in self.litteral_attrs:
            attrs[lattr.lower()] = data.pop(lattr.upper())
        for battr in self.boolean_attrs:
            attrs[battr.lower()] = data.pop(battr).lower() == 'true'
        if len(data) > 0:
            raise UnexpectedTeamData(data)
        return attrs

    @property
    def name(self):
        return self['FULL_NAME']

    def __contains__(self, item):
        return item in self.players

    def __getitem__(self, item):
        return self.attrs[item.lower()]

    def __repr__(self):
        return f'Team({self.name})'

    def __iter__(self):
        yield from self.players


class Player:
    numerical_attrs = ['AST', 'BLK', 'DREB', 'FG3A', 'FG3M', 'FG3_PCT', 'FGA',
                       'FGM', 'FG_PCT', 'FTA', 'FTM', 'FT_PCT', 'MIN_CEIL',
                       'MIN_FLOOR', 'OREB', 'PF', 'PTS', 'REB', 'STL', 'TOV']

    litteral_attrs = ['FULL_NAME', 'DAY_NAME', 'FIRST_NAME', 'LAST_NAME',
                      'MONTH_NAME', 'TEAM_PLACE', 'TEAM_NAME', 'DOUBLE_TYPE']

    boolean_attrs = ['STARTER', 'HOME', 'WINNER']

    def __init__(self, raw_data, idx):
        self.raw_data = raw_data  # cloning purposes
        self.idx = idx

        _raw_data = copy.deepcopy(self.raw_data)
        self.attrs = self.load_attrs(_raw_data)
        self.attrs = {key: str(val) for key, val in self.attrs.items()}

    def clone(self):
        return Player(self.raw_data, self.idx)

    def load_attrs(self, data):
        attrs = dict()
        for nattr in self.numerical_attrs:
            v = data.pop(nattr)
            attrs[nattr.lower()] = -1 if v == 'N/A' else int(v)
        for lattr in self.litteral_attrs:
            attrs[lattr.lower()] = data.pop(lattr)
        for battr in self.boolean_attrs:
            attrs[battr.lower()] = data.pop(battr).lower() == 'true'
        if len(data) > 0:
            raise UnexpectedPlayerData(data)
        return attrs

    @property
    def name(self):
        return self['full_name']

    def __eq__(self, other):
        if isinstance(other, Player):
            return self is other
        if isinstance(other, str):
            return self.name == other
        return False

    def __getitem__(self, item):
        return self.attrs[item.lower()]

    def __repr__(self):
        return f'Player({self["full_name"]})'


class PlayerList(Iterable):

    def __init__(self, players=None):
        self.players = list()
        self.player_by_names = dict()
        if players: self.extend(players)

        self._add_rank_to_players()

    def __len__(self):
        return len(self.players)

    def append(self, player):
        self.players.append(player.clone())
        self.player_by_names[player.name] = self.players[-1]

    def extend(self, players):
        for player in players:
            self.append(player)

    def _add_rank_to_players(self):
        sorted_players = sorted(self.players, key=lambda p: int(p['pts']), reverse=True)
        for player in self.players:
            player.attrs['rank'] = str(sorted_players.index(player) + 1)

    def sort_by(self, attr, reverse=True):
        return PlayerList(sorted(self.players, key=lambda p: p[attr], reverse=reverse))

    def filter_by(self, attr, value):
        return PlayerList([p for p in self.players if p[attr] == value])

    def __getitem__(self, item):
        if isinstance(item, Player):
            if item not in self:
                raise MissingPlayer(f'{item} is not in this PlayerList.')
            return item
        elif isinstance(item, int):
            return self.players[item]
        elif isinstance(item, str):
            if item in self.player_by_names:
                return self.player_by_names[item]
            raise MissingPlayer(f'{item} is not in this PlayerList.')
        elif isinstance(item, Iterable):
            return PlayerList([self[i] for i in item])
        raise TypeError(f'Cannot get {type(item)} from PlayerList')

    def get(self, item, default=None):
        try:
            return self[item]
        except MissingPlayer as e:
            return default

    def __iter__(self):
        yield from self.players

    def __contains__(self, item):
        if isinstance(item, Player):
            return item in self.players
        elif isinstance(item, str):
            return item in self.player_by_names
        return False

    def __repr__(self):
        sep = ',\n  '
        return f"PlayerList([\n  {sep.join([str(p) for p in self.players])}\n])"


class Game:
    def __init__(self, raw_game):
        self.raw_game = copy.deepcopy(raw_game)  # clone purposes

        home_team, visiting_team = self.raw_game[:2]
        home_idx, visiting_idx = 0, 1
        if home_team['HOME'].lower() == 'false':
            visiting_team, home_team = home_team, visiting_team
            home_idx, visiting_idx = 1, 0

        self.players = PlayerList(Player(p, idx)
                                  for idx, p in enumerate(self.raw_game[2:], 2))

        visiting_players = self.players.filter_by('TEAM_NAME', visiting_team['TEAM_NAME'])
        home_players = self.players.filter_by('TEAM_NAME', home_team['TEAM_NAME'])

        self.visiting_team = Team(visiting_team, visiting_idx, visiting_players)
        self.home_team = Team(home_team, home_idx, home_players)

        if self.visiting_team['winner'] == 'True':
            self.winning_team = self.visiting_team
            self.losing_team = self.home_team
        else:
            self.losing_team = self.visiting_team
            self.winning_team = self.home_team

    @property
    def teams(self):
        return self.winning_team, self.losing_team

    def clone(self):
        return Game(self.raw_game)

    @property
    def num_players(self):
        return len(self.players)

    def __contains__(self, item):
        if isinstance(item, (Player, str)):
            return item in self.players
        elif isinstance(item, Iterable):
            return all([p in self for p in item])
        raise TypeError(f'Game do not contain items of {type(item)}')

    def __getitem__(self, item):
        if isinstance(item, (Player, str)):
            return self.players[item]
        raise TypeError(f'{type(item)} is not a valid way to get an entity from a game.')

    def __repr__(self):
        return f'Game({self.home_team.name} vs {self.visiting_team.name})'
