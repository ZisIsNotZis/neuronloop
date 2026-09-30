import torch
from torch.nn import *


class ctx:
    pass


def pad(x):
    return functional.pad(x, [0, 0] * (x.ndim - 2) + [0, x.size(1)])


class inv(torch.autograd.Function):
    @staticmethod
    def forward(s, x, f, F, *_):
        s.f = f
        s.F = F
        s._ = len(_) + 2
        with torch.no_grad():
            return pad(f(x[:, : x.size(1) // 2]))

    @staticmethod
    def backward(s, y):
        dy, y = y.chunk(2, 1)
        with torch.no_grad():
            x = s.F(y)
        x.requires_grad = True
        with torch.enable_grad():
            y = s.f(x)
        y.backward(dy)
        return torch.cat([x.grad, x], 1), *[None] * s._


class rev(torch.autograd.Function):
    @staticmethod
    def forward(s, x, f, g1, G1, g2, G2, k, *_):
        s.f = f
        s.g1 = g1
        s.G1 = G1
        s.g2 = g2
        s.G2 = G2
        s._ = len(_) + 6
        k = round(x.size(1) * k / 2)
        with torch.no_grad():
            return pad(torch.cat(rev.f(s, *x.split([k, x.size(1) // 2 - k, x.size(1) // 2], 1)[:2]), 1))

    @staticmethod
    def backward(s, y):
        dy1, dy2, y1, y2 = y.split([y.size(1) // 2 - s.k, s.k, y.size(1) // 2 - s.k, s.k], 1)
        with torch.no_grad():
            x1, x2 = rev.F(s, y1, y2)
        x1.requires_grad = x2.requires_grad = True
        with torch.enable_grad():
            y1, y2 = rev.f(s, x1, x2)
        y1.backward(dy1)
        y2.backward(dy2)
        return torch.cat([x1.grad, x2.grad, x1, x2], 1), *[None] * s._

    @staticmethod
    def f(s, x1, x2):
        y2 = s.g1(x1)
        s.k = y2.size(1)
        return s.g2(x2) + s.f(x1), y2

    @staticmethod
    def F(s, y1, y2):
        x1 = s.G1(y2)
        return x1, s.G2(y1 - s.f(x1))

    @staticmethod
    def inv(f, G1, G2, k, y):
        s = Ctx()
        s.f = f
        s.G1 = G1
        s.G2 = G2
        k = round(x.size(1) * k)
        return torch.cat(rev.F(s, *y.split([y.size(1) - k, k], 1)), 1)


class end(torch.autograd.Function):
    @staticmethod
    def forward(s, x):
        x = x[:, : x.size(1) // 2]
        s.save_for_backward(x)
        return x

    @staticmethod
    def backward(s, dy):
        return torch.cat([dy, *s.saved_tensors], 1)


class InvPool1d(Module):
    def forward(s, x):
        return x.unflatten(2, (-1, 2)).transpose(2, 3).flatten(1, 2)

    def inv(s, y):
        return y.unflatten(1, (-1, 2)).transpose(2, 3).flatten(2)


class InvUnpool1d(InvPool1d):
    def forward(s, x):
        return super().inv(x)

    def inv(s, y):
        return super().forward(y)


class InvConv1d(Conv1d):
    def forward(s, x):
        return torch.fft.ifft(torch.fft.fft(x)[:, None] * torch.fft.fft(s.weight.flip(2), x.size(2))).real.sum(2).roll(-1, 2) + s.bias[:, None]

    def inv(s, y):
        return torch.fft.ifft(torch.fft.fft((b - s.bias[:, None]).roll(1, 2) / s.weight.size(1))[:, :, None] / torch.fft.fft(s.weight.flip(2), b.size(2))).real.mean(1)


class InvConvSpace1d(Conv1d):
    def forward(s, x):
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(s.weight[None, :, 0].flip(2), x.size(2))).real.roll(-1, 2)

    def inv(s, x):
        return torch.fft.ifft(torch.fft.fft(x.roll(1, 2)) / torch.fft.fft(s.weight[None, :, 0].flip(2), x.size(2))).real


class InvConvFeature1d(Conv1d):
    def forward(s, x):
        return s.weight[..., 0] @ x

    def inv(s, y):
        return torch.linalg.lstsq(s.weight[None, ..., 0], y)[0]


class InvLinear(Linear):
    def inv(s, y):
        return torch.linalg.lstsq(s.weight.T, y.T)[0].T


class InvIdentity(Identity):
    def inv(s, x):
        return x


class Rev(Sequential):
    def forward(s, x):
        x = pad(x)
        for s in s:
            if hasattr(s, "inv"):
                x = inv.apply(x, s, s.inv, *s.parameters())
            else:
                if not isinstance(s, RevPair):
                    s = RevPair(s)
                x = rev.apply(x, s.f, s.g1, s.g1.inv, s.g2, s.g2.inv, s.k, *s.parameters())
        return end.apply(x)

    def inv(s, x):
        for s in s[::-1]:
            if hasattr(s, "inv"):
                x = s.inv(x)
            else:
                if not isinstance(s, RevPair):
                    s = RevPair(s)
                x = rev.inv(x, s.f, s.g1.inv, s.g2.inv, s.k, *s.parameters())
        return x


class RevPair(Module):
    def __init__(s, f, g1=InvIdentity(), g2=None, k=0.5):
        super().__init__()
        s.f = f
        s.g1 = g1
        s.g2 = g2 or g1
        s.k = k


class F(Module):
    def __init__(s, f):
        super().__init__()
        s.f = f

    def forward(s, x):
        return s.f(x)

    def extra_repr(s):
        return repr(s.f)