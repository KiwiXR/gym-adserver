from typing import TypeVar, Optional

SelfAd = TypeVar('SelfAd', bound='Ad')


class Ad:
    def __init__(self, id: int, impressions: int = 0, clicks: int = 0, cpi: float = 0., rpc: float = 0.):
        """
        Advertisement information
        :param id: The index of the ad
        :param impressions: The impression of the ad, i.e., how many times the ad is recommended
        :param clicks: The total number of clicks on the ad
        :param cpi: How much to cost if we would like to recommend the ad, i.e., Cost-Per-Impression
        :param rpc: How much we would get if the ad is clicked, i.e., Reward-Per-Click
        """
        self.id = str(id)
        self.impressions = impressions
        self.clicks = clicks
        self.cpi = cpi
        self.rpc = rpc

    def ctr(self):
        """Gets the CTR (Click-through rate) for this ad.

        Returns:
            float: Returns the CTR (between 0 and 1)
        """
        return 0.0 if self.impressions == 0 else float(self.clicks / self.impressions)

    def total_gain(self):
        return self.clicks * self.rpc - self.impressions * self.cpi

    def avg_gain(self):
        # (self.clicks * self.rpc - self.impressions * self.cpi) / self.impressions
        # = self.ctr() * self.rpc - self.cpi * I(self.impressions != 0)
        return self.ctr() * self.rpc - (self.impressions != 0) * self.cpi

    def __repr__(self):
        return "({0}/{1})#cpi={2}#rpc={3})".format(self.clicks, self.impressions, self.cpi, self.rpc)

    def __str__(self):
        return "Ad: {0}, CTR: {1:.4f}, TotGain: {2:.4f}, AvgGain: {3:.4f}".format(self.id, self.ctr(),
                                                                                  self.total_gain(), self.avg_gain())

    def __eq__(self, other: SelfAd):
        return self.id == other.id and self.impressions == other.impressions and self.clicks == other.clicks
