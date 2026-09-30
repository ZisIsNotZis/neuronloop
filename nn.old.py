from torch.nn import *
import numpy as np
import torch
import inspect
import functools
import inspect
import math


def Repr(_):
    t = repr(_)
    if t[0] == "<":
        t = getattr(_, "__name__", "<")
        if t[0] == "<":
            t = ""
    return t


def Try(f, *_, **__):
    try:
        return f(*_, **__)
    except:
        pass


class F(Module):
    X = object()

    def __init__(s, f, *_, **__):
        super().__init__()
        s.f = f
        s._ = list(_)
        s.__ = __
        s.i = Try(_.index, F.X) or -99
        s.__name__ = Repr(f) or inspect.stack()[1].function
        for _ in [["f", f]], enumerate(_), __.items():
            for _ in _:
                s.reg(*_)

    def reg(s, _, __):
        if isinstance(__, (list, tuple, dict)):
            for i, __ in __.items() if isinstance(__, dict) else enumerate(__):
                s.reg(f"{_}{i}", __)
        elif isinstance(__, Module):
            s.add_module(str(_), __)
        elif isinstance(__, Parameter):
            s.register_parameter(str(_), __)
        elif hasattr(__, "__call__"):
            try:
                for i, __ in inspect.signature(__).parameters.items():
                    s.reg(f"{_}{i}", __.default)
            except ValueError:
                pass

    def forward(s, *_, **__):
        try:
            return s.f(*s._[: s.i], *_, *s._[s.i + 1 :], **__, **s.__)
        except Exception as e:
            s.__name__ += f"\33[31m{e}\33[0m"
            raise

    def __len__(s):
        return len(s._)

    def __iter__(s):
        return iter(s._)

    def __getitem__(s, _):
        if isinstance(_, slice):
            return F(s.f, *s._[_], **s.__)
        return (s._ if isinstance(_, int) else s.__)[_]

    def __setitem__(s, _, __):
        (s._ if isinstance(_, (int, slice)) else s.__)[_] = __
        s.reg(_, __)

    def __repr__(s):
        t = {}, {}
        for _ in enumerate(s._), s.__.items():
            for _, __ in _:
                t[isinstance(__, Module)][_] = __
        return s.__name__ + "(" + ", ".join(f"{_}:{Repr(__)}" for _, __ in t[0].items()) + "".join(f"\n{_}: {__}" for _, __ in t[1].items()).replace("\n", "\n  ") + ")"

    @staticmethod
    def lift(_):
        return F(F, _, F.X)


@F.lift
def Const(x, _=0):
    return _


@F.lift
def Peek(x, _):
    _(x)
    return x


def Seq(*_):
    return F(seq, *_) if len(_ := [_ for _ in _ if not isinstance(_, Identity) for _ in (_ if isinstance(_, F) and getattr(_, "f", None) == seq else [_])]) > 1 else _[0] if _ else Identity()


def seq(x, *_):
    for _ in _:
        x = _(x)
    return x


@F.lift
def Map(x, _):
    return [_(x) for x in x]


def MapF(*_, end=Identity()):
    return Seq(F(mapF, *_), end) if len(_ := [_ for _ in _ if _]) > 1 else _[0]


def mapF(x, *_):
    return [_(x) for _ in _]


@F.lift
def Zip(x, *_):
    return [_(x) for _, x in zip(_, x)]


def Add(*_):
    return MapF(*_, end=sum)


def Res(*_):
    return Add(Seq(*_), Identity())


def Mul(*_):
    return MapF(*_, end=math.prod)


def Stack(_):
    return Seq(Map(_), torch.stack)


@F.lift
def LiftSz(x, _):
    return _(x).view(x.shape)


def LiftT(*_):
    return Seq(T, *_, T)


def linspace(l):
    return torch.linspace(0, 1, l)[:, None]


def Pos(h):
    return Seq(F(torch.Tensor.size, -2), linspace, Linear(1, h, False))


def trispace(l):
    return torch.stack([torch.sin(l := torch.linspace(0, 2 * torch.pi, l)), torch.cos(l)], 1)


def TriPos(h):
    return Seq(F(torch.Tensor.size, -2), trispace, Linear(2, h, False))


def Nn(i, h, o=0, bias=True, pos=lambda h: None, bn=Identity, act=ReLU(True)):
    return Seq(
        Add(
            (Linear if i else Identity)(i, h, bn == Identity),
            pos(h),
        ),
        bn(h),
        act,
        (Linear if o else Identity)(h, o, bias),
    )


def SwiGLU(i, h, o=0, bias=True, pos=lambda h: None, bn=Identity, act=Hardswish(True)):
    return Seq(
        Mul(
            Seq(
                Add(
                    (Linear if i else Identity)(i, h, bn == Identity),
                    pos(h),
                ),
                bn(h),
                act,
            ),
            Linear(i, h),
        ),
        (Linear if o else Identity)(h, o, bias),
    )


def AttentionMin(x):
    x = (x if isinstance(x, list) else [x]) * 3
    return (x[0] @ x[1].transpose(-2, -1)).softmax(-1) @ x[2]


def Attention(i, h, o=0, nn=[Nn] * 4, bias=False):
    return Seq(
        MapF(
            nn[0](i, h, h, bias),
            nn[1](i, h, h if bias else 0),
            nn[2](i, h),
        ),
        AttentionMin,
        (nn[3] if o else Identity)(0, h, o),
    )


def MultiheadAttentionMin(x):
    x = (x if isinstance(x, list) else [x]) * 3
    return torch.einsum("...Llh,...lhk->...Lhk", torch.einsum("...Lhk,...lhk->...Llh", x[0], x[1]).softmax(-2), x[2])


def MultiheadAttention(i, h, o=0, head=1, nn=[Nn] * 4, bias=False):
    return Seq(
        MapF(
            Seq(
                nn[0](i, head * h, head * h, bias),
                Unflatten(-1, [head, h]),
            ),
            Seq(
                nn[1](i, head * h, head * h, bias),
                Unflatten(-1, [head, h]),
            ),
            Seq(
                nn[2](i, head * h, head * h),
                Unflatten(-1, [head, h]),
            ),
        ),
        MultiheadAttentionMin,
        Flatten(-2),
        (nn[3] if o else Identity)(0, head * h, o),
    )


@F.lift
def RandHot(x, y, p=0.2):
    y = torch.rand(*x.shape, *[] if x.dtype == torch.float32 else [y], device=x.device)
    p = torch.rand_like(y[..., :1]) ** p
    y *= p / y.sum(-1, True)
    if x.dtype == torch.float32:
        y += (1 - p) * x
    else:
        y[torch.meshgrid(*[torch.arange(i, device=x.device) for i in y.shape[:-1]]) + (x,)] += 1 - p[..., 0]
    return y


def BatchNorm(*_, **__):
    return LiftSz(Seq(Flatten(1, -2), T, BatchNorm1d(*_, **__), T))


@F.lift
def If(x, f, _, __=Identity()):
    return _(x) if f(x) else __(x)


T = F(torch.transpose, 1, 2)