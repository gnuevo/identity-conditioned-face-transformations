"""Scorer tracks accuracies on the long term, computing averages so we
reduce the noise in the evolution and see the tendency over time
"""

from collections import OrderedDict

class Scorer(object):
    """Tracks a series of variables that change over time, and returns the
    current average when asked

    """
    def __init__(self, batch_size, variables=(), queue_size=1000):
        """Initialises the object

        Args:
            batch_size: size of the batch
            variables: iterable with the names of the variables to track
            queue_size: maximum number of values stored per variable
        """
        self.batch_size = batch_size
        self.variables = variables
        self.queue_size = queue_size
        self.queue = OrderedDict()
        for var_name in variables:
            self.queue[var_name] = []

    def _single_add(self, key, value):
        """Adds the value for a single target variable

        Args:
            key: name of the variable
            value: value to store
        """
        self.queue[key].append(value)
        if len(self.queue[key]) > self.queue_size:
            self.queue[key] == self.queue[key][1:]

    def add(self, dict):
        """Receives a dictionary and stores the values in the internal queue

        Args:
            dict: dictionary with containing values for the target variables
        """
        for key, value in dict.items():
            if not key in self.variables:
                raise ValueError("Provided key '{}' to scorer that is not "
                                 "contained in the list of watched keys {"
                                 "}".format(key, self.variables))
            if key in self.variables:
                self._single_add(key, value)

    def get_scores(self, n=None):
        """Computes the average scores

        Args:
            n: computer scores over the last n elements

        Returns: the scores
        """
        if n == None: n = self.queue_size
        scores = OrderedDict()
        for key, value in self.queue.items():
            num = min(n, len(self.queue[key]))
            if num > 0:
                scores[key] = sum(self.queue[key][-num:])/float(num)
            else:
                scores[key] = 0.0
        return scores