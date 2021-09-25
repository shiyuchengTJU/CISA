from __future__ import absolute_import

from foolbox.models.base import Model
from foolbox.models.base import DifferentiableModel


class ModelWrapper(Model):
    """Base class for models that wrap other models.

    This base class can be used to implement model wrappers
    that turn models into new models, for example by preprocessing
    the input or modifying the gradient.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.

    """

    def __init__(self, model):
        super(ModelWrapper, self).__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def batch_predictions(self, images):
        return self.wrapped_model.batch_predictions(images)

    def predictions(self, image):
        return self.wrapped_model.predictions(image)

    def num_classes(self):
        return self.wrapped_model.num_classes()


class DifferentiableModelWrapper(ModelWrapper):
    """Base class for models that wrap other models and provide
    gradient methods.

    This base class can be used to implement model wrappers
    that turn models into new models, for example by preprocessing
    the input or modifying the gradient.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.

    """

    def predictions_and_gradient(self, image, label):
        return self.wrapped_model.predictions_and_gradient(image, label)

    def gradient(self, image, label):
        return self.wrapped_model.gradient(image, label)

    def backward(self, gradient, image):
        return self.wrapped_model.backward(gradient, image)


class ModelWithoutGradients(ModelWrapper):
    """Turns a model into a model without gradients.

    """
    pass


class ModelWithEstimatedGradients(DifferentiableModelWrapper):
    """Turns a model into a model with gradients estimated
    by the given gradient estimator.

    Parameters
    ----------
    model : :class:`Model`
        The model that is wrapped.
    gradient_estimator : `callable`
        Callable taking three arguments (pred_fn, image, label) and
        returning the estimated gradients. pred_fn will be the
        batch_predictions method of the wrapped model.
    """

    def __init__(self, model, gradient_estimator):
        super(ModelWithEstimatedGradients, self).__init__(
            model=model)

        assert callable(gradient_estimator)
        self._gradient_estimator = gradient_estimator

    def predictions_and_gradient(self, image, label):
        predictions = self.predictions(image)
        gradient = self.gradient(image, label)
        return predictions, gradient

    def gradient(self, image, label):
        pred_fn = self.batch_predictions
        bounds = self.bounds()
        return self._gradient_estimator(pred_fn, image, label, bounds)

    def backward(self, gradient, image):
        raise NotImplementedError


class CompositeModel(DifferentiableModel):
    """Combines predictions of a (black-box) model with the gradient of a
    (substitute) model.

    Parameters
    ----------
    forward_model : :class:`Model`
        The model that should be fooled and will be used for predictions.
    backward_model : :class:`Model`
        The model that provides the gradients.

    """

    def __init__(self, forward_model, backward_model):
        bounds = forward_model.bounds()
        assert bounds == backward_model.bounds()

        channel_axis = forward_model.channel_axis()
        assert channel_axis == backward_model.channel_axis()

        num_classes = forward_model.num_classes()
        assert num_classes == backward_model.num_classes()

        super(CompositeModel, self).__init__(
            bounds=bounds,
            channel_axis=channel_axis)

        self.forward_model = forward_model
        self.backward_model = backward_model
        self._num_classes = num_classes

    def num_classes(self):
        return self._num_classes

    def batch_predictions(self, images):
        return self.forward_model.batch_predictions(images)

    def predictions_and_gradient(self, image, label):
        predictions = self.forward_model.predictions(image)
        gradient = self.backward_model.gradient(image, label)
        return predictions, gradient

    def backward_model_predictions(self, image):
        return self.backward_model.predictions(image)

    def gradient(self, image, label):
        return self.backward_model.gradient(image, label)

    def backward(self, gradient, image):
        return self.backward_model.backward(gradient, image)

    def __enter__(self):
        assert self.forward_model.__enter__() == self.forward_model
        assert self.backward_model.__enter__() == self.backward_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        r1 = self.forward_model.__exit__(exc_type, exc_value, traceback)
        r2 = self.backward_model.__exit__(exc_type, exc_value, traceback)
        if r1 is None and r2 is None:
            return None
        return (r1, r2)  # pragma: no cover
