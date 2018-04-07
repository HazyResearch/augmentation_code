import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class MultinomialLogisticRegression(nn.Module):
    """Abstract class for multinomial logistic regression.
    Subclasses need to implement @features and @output_from_features.
    """

    def features(self, x):
        raise NotImplementedError()

    def output_from_features(self, feat):
        raise NotImplementedError()

    def forward(self, x):
        return self.output_from_features(self.features(x))

    @staticmethod
    def loss(output, target, reduce=True):
        return F.cross_entropy(output, target, reduce=reduce)

    @staticmethod
    def predict(output):
        return output.data.max(1)[1]


class MultinomialLogisticRegressionAug(MultinomialLogisticRegression):
    """Abstract class for multinomial logistic regression on augmented data.
    Input has size B x T x ..., where B is batch size and T is the number of
    transformations.
    Original i-th data point placed first among the transformed versions, which
    is input[i, 0].
    Output has size B x T.
    Works exactly like the non-augmented version with default options and normal
    loader (where input is of size B x ...).
    """

    def __init__(self, approx=True, feature_avg=True, regularization=False):
        """Parameters:
            approx: whether to use approximation or train on augmented data points.
                    If False, ignore @feature_avg and @regularization.
            feature_avg: whether to average features or just use the features of the original data point.
            regularization: whether to add 2nd order term (variance regularization).
        """
        self.approx = approx
        self.feature_avg = feature_avg
        self.regularization = regularization
        self.regularization_2nd_order_general = self.regularization_2nd_order

    def forward(self, x):
        augmented = x.dim() > 4
        if augmented:
            n_transforms = x.size(1)
            x = combine_transformed_dimension(x)
        feat = self.features(x)
        if self.approx and augmented:
            if not feat.requires_grad:
                feat.requires_grad = True
            feat = split_transformed_dimension(feat, n_transforms)
            if self.feature_avg:
                self._avg_features = feat.mean(dim=1)
            else:
                self._avg_features = feat[:, 0]
            self._centered_features = feat - self._avg_features[:, None]
            feat = self._avg_features
        output = self.output_from_features(feat)
        if not self.approx and augmented:
            output = split_transformed_dimension(output, n_transforms)
        return output

    @staticmethod
    def predict(output):
        if output.dim() > 2:
            # Average over transformed versions of the same data point
            output = output.mean(dim=1)
        return output.data.max(1)[1]

    @classmethod
    def loss_original(cls, output, target, reduce=True):
        """Original cross entropy loss.
        """
        return super().loss(output, target, reduce=reduce)

    @classmethod
    def loss_on_augmented_data(cls, output, target, reduce=True):
        """Loss averaged over augmented data points (no approximation).
        """
        # For each data point, replicate the target then compute the cross
        # entropy loss. Finally stack the result.
        loss = torch.cat([
            cls.loss_original(out, tar.repeat(out.size(0)))
            for out, tar in zip(output, target)
        ])
        return loss.mean() if reduce else loss

    def regularization_2nd_order(self, output, reduce=True):
        """Compute regularization term from output instead of from loss.
        Fast implementation by evaluating the Jacobian directly instead of relying on 2nd order differentiation.
        """
        p = F.softmax(output, dim=-1)
        # Using autograd.grad(output[:, i]) is slower since it creates new node in graph.
        # ones = torch.ones_like(output[:, 0])
        # W = torch.stack([autograd.grad(output[:, i], self._avg_features, grad_outputs=ones, create_graph=True)[0]
        #                  for i in range(10)], dim=1)
        eye = torch.eye(output.size(1)).cuda() if output.is_cuda else torch.eye(output.size(1))
        eye = eye[None, :].expand(output.size(0), -1, -1)
        W = torch.stack([autograd.grad(output, self._avg_features, grad_outputs=eye[:, i], create_graph=True)[0]
                         for i in range(10)], dim=1)
        # t = (W[:, None] * self._centered_features[:, :, None]).view(W.size(0), self._centered_features.size(1), W.size(1), -1).sum(dim=-1)
        t = (W.view(W.size(0), 1, W.size(1), -1) @ self._centered_features.view(*self._centered_features.shape[:2], -1, 1)).squeeze(-1)
        term_1 = (t**2 * p[:, None]).sum(dim=-1).mean(dim=-1)
        # term_1 = (t**2 @ p[:, :, None]).squeeze(2).mean(dim=-1)
        term_2 = ((t * p[:, None]).sum(dim=-1)**2).mean(dim=-1)
        # term_2 = ((t @ p[:, :, None]).squeeze(2)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg

    def regularization_2nd_order_linear(self, output, reduce=True):
        """Variance regularization (2nd order) term when the model is linear.
        Fastest implementations since it doesn't rely on pytorch's autograd.
        Equal to E[(W phi - W psi)^T (diag(p) - p p^T) (W phi - W psi)] / 2,
        where W is the weight matrix, phi is the feature, psi is the average
        feature, and p is the softmax probability.
        In this case @output is W phi + bias, but the bias will be subtracted away.
        """
        p = F.softmax(output, dim=-1)
        unreduced_output = self.output_from_features(self._centered_features + self._avg_features[:, None])
        reduced_output = self.output_from_features(self._avg_features)
        centered_output = unreduced_output - reduced_output[:, None]
        term_1 = (centered_output**2 * p[:, None]).sum(dim=-1).mean(dim=-1)
        term_2 = ((centered_output * p[:, None]).sum(dim=-1)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg

    def regularization_2nd_order_slow(self, output, reduce=True):
        """Compute regularization term from output, but uses pytorch's 2nd order differentiation.
        Slow implementation, only faster than @regularization_2nd_order_from_loss.
        """
        p = F.softmax(output, dim=-1)
        g, = autograd.grad(output, self._avg_features, grad_outputs=p, create_graph=True)
        term_1 = []
        for i in range(self._centered_features.size(1)):
            gg, = autograd.grad(g, p, grad_outputs=self._centered_features[:, i], create_graph=True)
            term_1.append((gg**2 * p).sum(dim=-1))
        term_1 = torch.stack(term_1, dim=-1).mean(dim=-1)
        term_2 = ((g[:, None] * self._centered_features).view(*self._centered_features.shape[:2], -1).sum(dim=-1)**2).mean(dim=-1)
        reg = (term_1 - term_2) / 2
        return reg.mean() if reduce else reg

    def regularization_2nd_order_from_loss(self, loss, reduce=True):
        """Variance regularization (2nd order) term.
        Computed from loss, using Pytorch's 2nd order differentiation.
        This is much slower but more likely to be correct. Used to check other implementations.
        """
        g, = autograd.grad(loss * self._avg_features.size(0), self._avg_features, create_graph=True)
        reg = []
        for i in range(self._centered_features.size(1)):
            gg, = autograd.grad(g, self._avg_features, grad_outputs=self._centered_features[:, i], create_graph=True)
            reg.append((gg * self._centered_features[:, i]).view(gg.size(0), -1).sum(dim=-1))
        reg = torch.stack(reg, dim=-1).mean(dim=-1) / 2
        return reg.mean() if reduce else reg

    def loss(self, output, target, reduce=True):
        """Cross entropy loss, with optional variance regularization.
        """
        if not self.approx:  # No approximation, loss on all augmented data points
            return self.loss_on_augmented_data(output, target, reduce=reduce)
        loss = self.loss_original(output, target, reduce=reduce)
        if self.regularization:
            return loss + self.regularization_2nd_order(output, reduce=reduce)
        else:
            return loss

    def all_losses(self, x, target, reduce=True):
        """All losses: true loss on augmented data, loss on original image, approximate
           loss with feature averaging (1st order), approximate loss with
           variance regularization and no feature averaging, and approximate
           loss with feature averaging and variance regularization (2nd order).
           Used to compare the effects of different approximations.

           Parameters:
               x: the input of size B (batch size) x T (no. of transforms) x ...
               target: target of size B (batch size)

        """
        approx, feature_avg = self.approx, self.feature_avg
        self.approx, self.feature_avg = True, True
        output = self(x)
        features = self._centered_features + self._avg_features[:, None]
        n_transforms = features.size(1)
        unreduced_output = self.output_from_features(combine_transformed_dimension(features))
        unreduced_output = split_transformed_dimension(unreduced_output, n_transforms)
        true_loss = self.loss_on_augmented_data(unreduced_output, target, reduce=reduce)
        reduced_output = output
        loss_original = self.loss_original(unreduced_output[:, 0], target, reduce=reduce)
        loss_1st_order = self.loss_original(reduced_output, target, reduce=reduce)
        reg_2nd_order = self.regularization_2nd_order(output, reduce=reduce)
        loss_2nd_order = loss_1st_order + reg_2nd_order
        loss_2nd_no_1st = loss_original + reg_2nd_order
        self.approx, self.feature_avg = approx, feature_avg
        return true_loss, loss_original, loss_1st_order, loss_2nd_no_1st, loss_2nd_order


class LinearLogisticRegression(MultinomialLogisticRegression):
    """Simple linear logistic regression model.
    """

    def __init__(self, n_features, n_classes):
        """Parameters:
            n_features: number of input features.
            n_classes: number of classes.
        """
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def features(self, x):
        return x.view(x.size(0), x.size(1), -1) if x.dim() > 4 else x.view(x.size(0), -1)

    def output_from_features(self, feat):
        return self.fc(feat)


class RBFLogisticRegression(MultinomialLogisticRegression):
    """Logistic regression with RBF kernel approximation (random Fourier features).
    Equivalent to neural network with 2 layers: first layer is random
    projection with sine-cosine nonlinearity, and second trainable linear
    layer.
    """

    def __init__(self, n_features, n_classes, gamma=1.0, n_components=100):
        """Parameters:
            n_features: number of input features.
            n_classes: number of classes.
            gamma: hyperparameter of the RBF kernel k(x, y) = exp(-gamma*||x-y||^2)
            n_components: number of components used to approximate kernel, i.e.
                          number of hidden units.
        """
        super().__init__()
        n_components //= 2  # Need 2 slots each for sine and cosine
        self.fc = nn.Linear(n_components * 2, n_classes)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.random_directions = nn.Parameter(
            torch.randn(n_features, n_components), requires_grad=False)

    def features(self, x):
        x = x.view(x.size(0), x.size(1), -1) if x.dim() > 4 else x.view(x.size(0), -1)
        projected_x = torch.sqrt(2 * self.gamma) * (x @ self.random_directions)
        # Don't normalize by sqrt(self.n_components), it makes the weights too small.
        return torch.cat((torch.sin(projected_x), torch.cos(projected_x)), -1)

    def output_from_features(self, feat):
        return self.fc(feat)


class LeNet(MultinomialLogisticRegression):
    """LeNet for MNIST, with 2 convolution-max pooling layers and 2 fully connected
    layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4]

    def layer_1(self, x):
        feat = F.relu(self.conv1(x))
        return F.max_pool2d(feat, 2)

    def layer_2(self, x):
        feat = F.relu(self.conv2(x))
        return F.max_pool2d(feat, 2).view(feat.size(0), -1)

    def layer_3(self, x):
        return F.relu(self.fc1(x))

    def layer_4(self, x):
        return F.relu(self.fc2(x))

    def features(self, x):
        feat = x
        for layer in self.layers:
            feat = layer(feat)
        return feat

    def output_from_features(self, feat):
        return self.fc3(feat)


class LinearLogisticRegressionAug(MultinomialLogisticRegressionAug,
                                  LinearLogisticRegression):
    """Linear logistic regression model with augmented data.
    Input has size B x T x ..., where B is batch size and T is the number of
    transformations.
    Original i-th data point placed first among the transformed versions, which
    is input[i, 0].
    Output has size B x T.
    """

    def __init__(self,
                 n_features,
                 n_classes,
                 approx=True,
                 feature_avg=True,
                 regularization=False):
        """Parameters:
            n_features: number of input features.
            n_classes: number of classes.
            approx: whether to use approximation or train on augmented data points.
                    If False, ignore @feature_avg and @regularization.
            feature_avg: whether to average features or just use the features of the original data point.
            regularization: whether to add 2nd order term (variance regularization).
        """
        LinearLogisticRegression.__init__(self, n_features, n_classes)
        MultinomialLogisticRegressionAug.__init__(self, approx, feature_avg,
                                                  regularization)
        self.regularization_2nd_order = self.regularization_2nd_order_linear


class RBFLogisticRegressionAug(MultinomialLogisticRegressionAug,
                               RBFLogisticRegression):
    """Logistic regression model with RBF kernel and augmented data.
    Input has size B x T x ..., where B is batch size and T is the number of
    transformations.
    Original i-th data point placed first among the transformed versions, which
    is input[i, 0].
    Output has size B x T.
    """

    def __init__(self,
                 n_features,
                 n_classes,
                 gamma=1.0,
                 n_components=100,
                 approx=True,
                 feature_avg=True,
                 regularization=False):
        RBFLogisticRegression.__init__(self, n_features, n_classes, gamma,
                                       n_components)
        MultinomialLogisticRegressionAug.__init__(self, approx, feature_avg,
                                                  regularization)
        self.regularization_2nd_order = self.regularization_2nd_order_linear


class LeNetAug(MultinomialLogisticRegressionAug, LeNet):
    """LeNet for MNIST, with 2 convolution-max pooling layers and 2 fully connected
    layers.
    """

    def __init__(self, approx=True, feature_avg=True, regularization=False, layer_to_avg=4):
        LeNet.__init__(self)
        MultinomialLogisticRegressionAug.__init__(self, approx, feature_avg,
                                                  regularization)
        error_msg = "[!] layer_to_avg should be in the range [0, ..., 4]."
        assert (layer_to_avg in range(5)), error_msg
        self.layer_to_avg = layer_to_avg
        if layer_to_avg == 4:  # Not a linear model unless averaging at 4th layer
            self.regularization_2nd_order = self.regularization_2nd_order_linear

    def features(self, x):
        feat = x
        for layer in self.layers[:self.layer_to_avg]:
            feat = layer(feat)
        return feat

    def output_from_features(self, feat):
        for layer in self.layers[self.layer_to_avg:]:
            feat = layer(feat)
        return self.fc3(feat)


def combine_transformed_dimension(input):
    """Combine the minibatch and the transformation dimensions.
    Parameter:
        input: Tensor of shape B x T x ..., where B is the batch size and T is
               the number of transformations.
    Return:
        output: Same tensor, now of shape (B * T) x ....
    """
    return input.view(-1, *input.shape[2:])


def split_transformed_dimension(input, n_transforms):
    """Split the minibatch and the transformation dimensions.
    Parameter:
        input: Tensor of shape (B * T) x ..., where B is the batch size and T is
               the number of transformations.
    Return:
        output: Same tensor, now of shape B x T x ....
    """
    return input.view(-1, n_transforms, *input.shape[1:])
