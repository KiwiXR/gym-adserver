from typing import Optional, List, Dict, Callable
import gym
from gym import logger, spaces
from gym.utils import seeding

import numpy as np
from numpy.random.mtrand import RandomState

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['toolbar'] = 'None'

from .ad import Ad


class AdServerEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, num_ads: int = 10, time_series_frequency: int = -1, ads_info: List[Dict] = None,
                 click_simulation: Callable = None, seed: int = None):
        self.np_random = None
        self.scenario_name = None
        self.time_series_frequency = time_series_frequency
        self.num_ads = num_ads
        self.click_simulation = click_simulation
        self.click_probabilities = None
        self.ads_info = ads_info
        self.seed(seed)  # avoid error in initialization
        if seed is not None and ads_info is None:
            self._init_ads_info()
        self.deterministic_ads_info = ads_info is not None or seed is not None

        # Initial state (can be reset later)
        ads = [Ad(i, **self.ads_info[i]) for i in range(num_ads)]
        clicks = 0
        impressions = 0
        self.state = (ads, impressions, clicks)
        self.ctr_time_series = []
        self.avg_gain_time_series = []
        self.tot_gain_time_series = []

        # Environment OpenAI metadata
        self.reward_range = (-0.5, 0.5)
        self.action_space = spaces.Discrete(num_ads)  # index of the selected ad
        low = np.array(
            [[
                0.0,  # clicks
                0.0,  # impressions
                0.0,  # cpi
                0.0,  # rpc
            ] for _ in range(num_ads)], dtype=np.float32
        )
        high = np.array(
            [[
                np.inf,  # clicks
                np.inf,  # impressions
                0.5,  # cpi
                1.0,  # rpc
            ] for _ in range(num_ads)], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, shape=(num_ads, 4),
                                            dtype=np.float32)  # clicks and impressions, for each ad

    def _init_ads_info(self):
        self.ads_info = []
        for _ in range(self.num_ads):
            cpi = self.np_random.uniform() * 0.5
            rpc = self.np_random.uniform() * 0.5 + cpi
            self.ads_info.append({"cpi": cpi, "rpc": rpc})
        return self.ads_info

    def _get_obs(self):
        assert isinstance(self.state, tuple) and len(self.state) == 3
        (ads, impressions, clicks) = self.state
        obs = []
        for ad in ads:
            obs.append([ad.clicks, ad.impressions, ad.cpi, ad.rpc])
        obs = np.asarray(obs, dtype=np.float32)
        assert obs.shape == self.observation_space.shape
        return obs

    def seed(self, seed=None):  # pragma: no cover
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        ads, impressions, clicks = self.state

        reward = - ads[action].cpi
        # Update clicks (if any)
        clicked = self._draw_click(action)
        if clicked == 1:
            clicks += 1
            ads[action].clicks += 1
            reward += ads[action].rpc

        # Update impressions
        ads[action].impressions += 1
        impressions += 1

        # Update the ctr time series (for rendering)
        if self.time_series_frequency > 0 and impressions % self.time_series_frequency == 0:
            ctr = 0.0 if impressions == 0 else float(clicks / impressions)
            self.ctr_time_series.append(ctr)
            total_gain = sum(ad.total_gain() for ad in ads)
            avg_gain = 0.0 if impressions == 0 else float(total_gain / impressions)
            self.avg_gain_time_series.append(avg_gain)
            self.tot_gain_time_series.append(total_gain)

        self.state = (ads, impressions, clicks)
        observation = self._get_obs()
        # info = {"ads": self.state[0], "impressions": self.state[1], "clicks": self.state[2]}

        return observation, reward, False, {}

    def reset(self, options=None):
        self.scenario_name = options["scenario_name"] if options is not None else "DefaultName"
        if not self.deterministic_ads_info:
            self._init_ads_info()
        ads = [Ad(i, **self.ads_info[i]) for i in range(self.num_ads)]
        clicks = 0
        impressions = 0
        self.state = (ads, impressions, clicks)
        self.ctr_time_series = []
        self.tot_gain_time_series = []
        self.avg_gain_time_series = []
        observation = self._get_obs()
        return observation

    def render(self, mode: str = 'human', freeze: bool = False, show=False, output_file=None):  # pragma: no cover
        if mode != 'human':
            raise NotImplementedError
        if self.time_series_frequency <= 0:
            raise NotImplementedError

        ads, impressions, clicks = self.state
        ctr = 0.0 if impressions == 0 else float(clicks / impressions)

        logger.info('Scenario: {}, Impressions: {}, CTR: {}, Ads: {}'.format(self.scenario_name, impressions, ctr, ads))

        fig = plt.figure(num=self.scenario_name, figsize=(9, 12))
        grid_size = (9, 2)

        # Plot CTR time series 
        plt.subplot2grid(grid_size, (0, 0), rowspan=2, colspan=2)
        x = [i for i, _ in enumerate(self.ctr_time_series)]
        y = self.ctr_time_series
        axes = plt.gca()
        axes.set_ylim([0, None])
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("CTR")
        plt.xlabel("Impressions")
        plt.plot(x, y, marker='o')
        for x, y in zip(x, y):
            plt.annotate("{:.2f}".format(y), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot impressions
        plt.subplot2grid(grid_size, (2, 0), rowspan=3, colspan=1)
        x = [ad.id for ad in ads]
        impressions = [ad.impressions for ad in ads]
        x_pos = [i for i, _ in enumerate(x)]
        plt.barh(x_pos, impressions)
        plt.ylabel("Ads")
        plt.xlabel("Impressions")
        plt.yticks(x_pos, x)

        # Plot CTRs and probabilities
        plt.subplot2grid(grid_size, (2, 1), rowspan=3, colspan=1)
        x = [ad.id for ad in ads]
        y = [ad.ctr() for ad in ads]
        y_2 = self.click_probabilities
        x_pos = [i for i, _ in enumerate(x)]
        x_pos_2 = [i + 0.4 for i, _ in enumerate(x)]
        plt.ylabel("Ads")
        plt.xlabel("")
        plt.yticks(x_pos, x)
        plt.barh(x_pos, y, 0.4, label='Actual CTR')
        plt.barh(x_pos_2, y_2, 0.4, label='Probability')
        plt.legend(loc='upper right')

        # Plot total gain time series
        plt.subplot2grid(grid_size, (5, 0), rowspan=2, colspan=2)
        x = [i for i, _ in enumerate(self.tot_gain_time_series)]
        y = self.tot_gain_time_series
        axes = plt.gca()
        axes.set_ylim(auto=True)
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("TotalGain")
        plt.xlabel("Impressions")
        plt.plot(x, y, marker='o')
        for x, y in zip(x, y):
            plt.annotate("{:.2f}".format(y), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot average gain time series
        plt.subplot2grid(grid_size, (7, 0), rowspan=2, colspan=2)
        x = [i for i, _ in enumerate(self.avg_gain_time_series)]
        y = self.avg_gain_time_series
        axes = plt.gca()
        axes.set_ylim(auto=True)
        plt.xticks(x, [(i + 1) * self.time_series_frequency for i, _ in enumerate(x)])
        plt.ylabel("AverageGain")
        plt.xlabel("Impressions")
        plt.plot(x, y, marker='o')
        for x, y in zip(x, y):
            plt.annotate("{:.2f}".format(y), (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.tight_layout()

        if output_file is not None:
            fig.savefig(output_file)

        if show:
            if freeze:
                # Keep the plot window open
                # https://stackoverflow.com/questions/13975756/keep-a-figure-on-hold-after-running-a-script
                if matplotlib.is_interactive():
                    plt.ioff()
                plt.show(block=True)
            else:
                plt.show(block=False)
                plt.pause(0.001)

    def _draw_click(self, action: int) -> bool:
        if self.click_simulation is not None:
            return self.click_simulation(action)

        if self.click_probabilities is None:
            self.click_probabilities = [self.np_random.uniform() * 0.5 for i in range(self.num_ads)]

        return True if self.np_random.uniform() <= self.click_probabilities[action] else False

    def close(self):
        plt.close()
