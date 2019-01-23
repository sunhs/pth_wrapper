import numpy as np


class EvalMetric(object):
    """Base class for all evaluation metrics.
    Stolen from mxnet.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self, name, output_names=None, label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update(
            {
                'metric': self.__class__.__name__,
                'name': self.name,
                'output_names': self.output_names,
                'label_names': self.label_names
            }
        )
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> Tensor
            name to tensor mapping for labels.

        preds : OrderedDict of str -> Tensor
            name to tensor mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `Tensor`
            The labels of the data.

        preds : list of `Tensor`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))


class LossMetric(EvalMetric):
    def __init__(self, name):
        super(LossMetric, self).__init__(name)
        self.batch_loss = 0

    def update(self, losses, num_inst):
        self.batch_loss = losses[0].item()
        self.sum_metric += self.batch_loss
        self.num_inst += num_inst
        self.batch_loss /= num_inst


class MyPrecMetric(EvalMetric):
    def __init__(self, name):
        super(MyPrecMetric, self).__init__(name)
        self.batch_prec = 0

    def update(self, labels, preds):
        preds = preds[0].detach().numpy()
        labels = labels[0].detach().numpy()

        assert preds.shape == labels.shape
        n_samples = preds.shape[0]
        self.batch_prec = 0

        for i in range(n_samples):
            p, t = preds[i], labels[i]
            true_positive = np.where(t == 1)[0]

            if not len(true_positive):
                continue

            sortind = np.argsort(p, axis=0)[::-1]
            correct = 0

            for j in range(len(true_positive)):
                if sortind[j] in true_positive:
                    correct += 1

            self.batch_prec += correct / max(len(true_positive), 1)

        self.sum_metric += self.batch_prec
        self.num_inst += n_samples
        self.batch_prec /= n_samples


class MAPMetric(EvalMetric):
    def __init__(self, name):
        super(MAPMetric, self).__init__(name)
        self.scores = None
        self.labels = None

    def update(self, labels, preds):
        preds = preds[0].detach().numpy()
        labels = labels[0].detach().numpy()

        if self.scores is None or self.labels is None:
            self.scores = preds
            self.labels = labels
            return

        self.scores = np.concatenate([self.scores, preds])
        self.labels = np.concatenate([self.labels, labels])

    def get(self):
        # copied from https://github.com/zxwu/lsvc2017
        probs = self.scores
        labels = self.labels
        AP = np.zeros((probs.shape[1], ))

        for i in range(probs.shape[1]):
            iClass = probs[:, i]
            iY = labels[:, i]
            idx = np.argsort(-iClass)
            iY = iY[idx]
            count = 0
            prec = 0.0
            skip_count = 0
            for j in range(iY.shape[0]):
                if iY[j] == 1:
                    count += 1
                    prec += count / float(j + 1 - skip_count)
                if iY[j] == -1:
                    skip_count += 1
                if count != 0:
                    AP[i] = prec / count
        return (self.name, AP)
