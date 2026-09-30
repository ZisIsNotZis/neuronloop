import torch
from torch.nn import *


# def padd(a, b, c=None):
#     if a.size(1) < b.size(1):
#         a, b = b, a
#     if a.size(1) > b.size(1):
#         a, c = a.split((b.size(1), a.size(1) - b.size(1)), 1)
#     a = a + b
#     return a if c is None else torch.cat((a, c), 1)


# def psub(a, b, s, c=None):
#     if a.size(1) > b.size(1):
#         a, c = a.split((b.size(1), a.size(1) - b.size(1)), 1)
#     a = a[:, : s[1]] - b[:, : s[1]]
#     return a if c is None else torch.cat((a, c), 1)


def padd(a, b):
    return a + b


def psub(a, b, s):
    return a - b


class inv(torch.autograd.Function):
    @staticmethod
    def forward(s, x, _1, f, F, dF, *_):
        s.f = f
        s.F = F
        s.dF = dF
        s._ = 3 + len(_)
        return (f(x),) * 2

    @staticmethod
    def backward(s, dy, y):
        x = s.F(y)
        if s.dF:
            x.grad = s.dF(dy)
        else:
            x.requires_grad = True
            with torch.enable_grad():
                y = s.f(x)
            y.backward(dy)
        return x.grad, x, *[None] * s._


class rev(torch.autograd.Function):
    @staticmethod
    def forward(s, x1, x2, _1, _2, f, g, G, *_):
        s.f = f
        s.g = g
        s.G = G
        s._ = 3 + len(_)
        y1, y2, s.x2s = rev.f(x1, x2, f, g)
        return (y1, y2) * 2

    @staticmethod
    def backward(s, dy1, dy2, y1, y2):
        x1, x2 = rev.F(y1, y2, s.f, s.G, s.x2s)
        x1.requires_grad = True
        if x2 is not None:
            x2.requires_grad = True
        with torch.enable_grad():
            y1, y2 = rev.f(x1, x2, s.f, s.g)
        y1.backward(dy1)
        y2.backward(dy2)
        return x1.grad, x2 if x2 is None else x2.grad, x1, x2, *[None] * s._

    @staticmethod
    def f(x1, x2, f, g):
        if x2 is None:
            return f(x1), g(x1), None
        x2 = g(x2)
        return padd(x2, f(x1)), g(x1), x2.shape

    @staticmethod
    def F(y1, y2, f, G, x2s):
        x1 = G(y2)
        if not x2s:
            return x1, None
        return x1, G(psub(y1, f(x1), x2s))


class end(torch.autograd.Function):
    @staticmethod
    def forward(s, x1, x2, *_):
        s.save_for_backward(x1.detach(), x2.detach())
        return x1

    @staticmethod
    def backward(s, dx1):
        return dx1, None, *s.saved_tensors


class Inv(Module):
    dforward = dinv = None

    @property
    def T(s):
        s.forward, s.inv = s.inv, s.forward
        s.dforward, s.dinv = s.dinv, s.dforward
        return s


class InvConv1d(Conv1d, Inv):
    def inv(s, y):
        # torch.fft.ifft(torch.fft.fft(x)[:, None] * torch.fft.fft(s.weight.flip(2), x.size(2))).real.sum(2).roll(-1, 2) + s.bias[:, None]
        return torch.fft.ifft(torch.fft.fft((b - s.bias[:, None]).roll(1, 2) / s.weight.size(1))[:, :, None] / torch.fft.fft(s.weight.flip(2), b.size(2))).real.mean(1)


class InvConvSpace1d(Conv1d, Inv):
    def inv(s, x):
        # torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(s.weight[None, :, 0].flip(2), x.size(2))).real.roll(-1, 2)
        return torch.fft.ifft(torch.fft.fft(x.roll(1, 2)) / torch.fft.fft(s.weight[None, :, 0].flip(2), x.size(2))).real


class InvConvFeature1d(Conv1d, Inv):
    def inv(s, y):
        # s.weight[..., 0] @ x
        return torch.linalg.lstsq(s.weight[None, ..., 0], y)[0]


class InvLinear(Linear, Inv):
    def inv(s, y):
        return torch.linalg.lstsq(s.weight.T, y.T)[0].T


class InvPool1d(Inv):
    def forward(s, x):
        return x.unflatten(2, (-1, 2)).transpose(2, 3).flatten(1, 2)

    def inv(s, x):
        return x.unflatten(1, (-1, 2)).transpose(2, 3).flatten(2)

    dforward = forward
    dinv = inv


class InvIdentity(Identity, Inv):
    def inv(s, x):
        return x

    dforward = dinv = inv


class Rev(Sequential):
    def __init__(s, *_, fast=2):
        super().__init__(*_)
        s.fast = fast

    def forward(s, x1):
        if not s.fast:
            _1 = torch.empty_like(x1)
        x2 = _2 = None
        for i in s:
            if isinstance(i, Inv):
                if s.fast:
                    x1 = i(x1)
                    if s.fast > 1:
                        x2 = None
                    elif x2 is not None:
                        x2 = i(x2)
                else:
                    x1, _1 = inv.apply(x1, _1, i, i.inv, i.dinv, *i.parameters())
                    if x2 is not None:
                        x2, _2 = inv.apply(x2, _2, i, i.inv, i.dinv, *i.parameters())
            else:
                if not isinstance(i, RevPair):
                    i = i, InvIdentity()
                if s.fast:
                    x1, x2 = rev.f(x1, x2, *i)[:2]
                else:
                    x1, x2, _1, _2 = rev.apply(x1, x2, _1, _2, *i, i[1].inv, *i.parameters())
        if not s.fast:
            x1 = end.apply(x1, x2, _1, _2)
        return x1


class RevPair(Sequential):
    pass


class F(Module):
    def __init__(s, f):
        super().__init__()
        s.f = f

    def forward(s, x):
        return s.f(x)

    def extra_repr(s):
        return repr(s.f)