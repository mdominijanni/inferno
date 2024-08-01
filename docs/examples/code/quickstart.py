import inferno
import torch
from inferno import functional, neural, learn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Lambda


# runtime settings
shape = (10, 10)
step_time = 1.0
batch_size = 20
epochs = 1
classify_interval = 10
train_cutoff = None
test_cutoff = None


# set device automatically
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"using device: {device}\n")


# retrieve and load the data
train_set = MNIST(
    root="data",
    train=True,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.squeeze(0)),
        ],
    ),
    download=True,
)

test_set = MNIST(
    root="data",
    train=False,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Lambda(lambda x: x.squeeze(0)),
        ],
    ),
    download=True,
)

train_data = DataLoader(train_set, batch_size=batch_size)
test_data = DataLoader(test_set, batch_size=batch_size)


# construct the neurons
exc = neural.ALIF(
    shape,
    step_time,
    rest_v=-65.0,
    reset_v=-60.0,
    thresh_eq_v=-52.0,
    refrac_t=5.0,
    tc_membrane=100.0,
    tc_adaptation=1e7,
    spike_increment=0.05,
    resistance=1.0,
    batch_size=batch_size,
)

inh = neural.LIF(
    shape,
    step_time,
    rest_v=-60.0,
    reset_v=-45.0,
    thresh_v=-40.0,
    refrac_t=2.0,
    time_constant=75.0,
    resistance=1.0,
    batch_size=batch_size,
)


# construct the connections
enc2exc = neural.LinearDense(
    (28, 28),
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(100.0),
    weight_init=lambda x: inferno.rescale(inferno.uniform(x), 0.0, 0.3),
    batch_size=batch_size,
)

inh2exc = neural.LinearLateral(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(100.0),
    weight_init=lambda x: inferno.full(x, -180.0),
    batch_size=batch_size,
)

exc2inh = neural.LinearDirect(
    shape,
    step_time,
    synapse=neural.DeltaCurrent.partialconstructor(75.0),
    weight_init=lambda x: inferno.full(x, 22.5),
    batch_size=batch_size,
)


# build the model class
class Model(inferno.Module):

    def __init__(self, exc, inh, enc2exc, inh2exc, exc2inh):

        # call superclass constructor
        inferno.Module.__init__(self)

        # construct the layers
        self.feedfwd = neural.Serial(enc2exc, exc)
        self.inhibit = neural.Serial(inh2exc, exc)
        self.trigger = neural.Serial(exc2inh, inh)

    def forward(self, inputs, trainer=None):
        # clears the model state
        def clear(m):
            if isinstance(m, neural.Neuron | neural.Connection):
                m.clear()

        # compute for each time step
        def step(x):

            # inference
            res = self.feedfwd(x)
            _ = self.inhibit(self.trigger(res))

            # training
            if self.training and trainer:
                trainer()
                self.feedfwd.connection.update()

            return res

        res = torch.stack([step(x) for x in inputs], dim=0)
        self.apply(clear)
        if trainer:
            trainer.clear()

        return res


model = Model(exc, inh, enc2exc, inh2exc, exc2inh)
model.to(device=device)


# add weight normalization
norm_hook = neural.Normalization(
    model,
    "feedfwd.connection.weight",
    order=1,
    scale=78.4,
    dim=-1,
    eval_update=False,
)
norm_hook.register()
norm_hook()


# create the trainer and updater, then connect them
trainer = learn.STDP(
    step_time=step_time,
    lr_post=5e-4,
    lr_pre=-5e-6,
    tc_post=30.0,
    tc_pre=30.0,
)

updater = model.feedfwd.connection.defaultupdater()
model.feedfwd.connection.updater = updater

trainer.register_cell("feedfwd", model.feedfwd.cell)
trainer.to(device=device)


# add weight bounding
clamp_hook = neural.Clamping(
    updater,
    "parent.weight",
    min=0.0,
)
clamp_hook.register()

updater.weight.upperbound(
    functional.bound_upper_multiplicative,
    1.0,
)


# build the classifier
classifier = learn.MaxRateClassifier(
    shape,
    num_classes=10,
    decay=1e-6,
)
classifier.to(device=device)


# build the encoder
encoder = neural.HomogeneousPoissonEncoder(
    250,
    step_time,
    frequency=128.0,
)


# create and run the training/testing loop
def train(data, encoder, model, trainer, classifier):
    size = len(data.dataset)
    rates, labels = [], []
    correct, current = 0, 0

    for batch, (X, y) in enumerate(data, start=1):
        X, y = X.to(device=device), y.to(device=device)

        rates.append(model(encoder(X), trainer).float().mean(dim=0))
        labels.append(y)

        if batch % classify_interval == 0:
            rates = torch.cat(rates, dim=0)
            labels = torch.cat(labels, dim=0)

            pred = classifier(rates, labels)
            nc = (pred == labels).sum().item()
            correct += nc
            current += rates.size(0)

            print(f"acc: {(nc / rates.size(0)):.4f} [{current:>5d}/{size:>5d}]")
            rates, labels = [], []

        if train_cutoff is not None and current >= train_cutoff:
            break

    print(f"Training Accuracy:\n    {(correct / current):.4f}")


def test(data, encoder, model, classifier):
    correct, current = 0, 0

    for batch, (X, y) in enumerate(data, start=1):
        X, y = X.to(device=device), y.to(device=device)

        rates = model(encoder(X), None).float().mean(dim=0)
        pred = classifier(rates, None)

        correct += (pred == y).sum().item()
        current += rates.size(0)

        if test_cutoff is not None and current >= test_cutoff:
            break

    print(f"Testing Accuracy:\n    {(correct / current):.4f}\n")


with torch.no_grad():
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------")
        model.train(), trainer.train()
        train(train_data, encoder, model, trainer, classifier)
        model.eval(), trainer.eval()
        test(test_data, encoder, model, classifier)
    print("Completed")
