import torch
from torch.nn import *
import functools


class partial:
    def __init__(s, f, *_, _pre=[], _unpak=False, **__):
        s.f = f
        s._ = _
        s.__ = __
        s.pre = _pre
        s.unpak = _unpak

    def __call__(s, *_, **__):
        return s.f(*s.pre, *_[0] if s.unpak else _, *s._, **__, **s.__)

    def __str__(s):
        return (s.f.__name__ if getattr(s.f, "__name__", "<lambda>") != "<lambda>" else str(s.f)) + "(" + ", ".join(map(str, (*s.pre, "**" if s.unpak else "*", *s._, *(f"{i}={j}" for i, j in s.__.items())))) + ")"

    def __repr__(s):
        return str(s)


class F(partial, Module):
    def __init__(s, *_, **__):
        partial.__init__(s, *_, **__)
        Module.__init__(s)
        for _ in enumerate(s._), enumerate(s.pre, len(s._)), s.__.items():
            for __, _ in _:
                if isinstance(_, Module):
                    s.register_module(str(__), _)


class Map(Sequential):
    def __init__(s, f, *_, name="", unpak=False, zip=False):
        super().__init__(*[_ for _ in _ if _])
        s.f = f
        s.name = name
        s.unpak = unpak
        s.zip = zip

    def forward(s, x):
        x = [s(x) for s, x in zip(s, x)] if s.zip else [s(x) for s in s]
        return s.f(*x if s.unpak else [x])

    def _get_name(s):
        return s.name or str(s.f)


class Const(Module):
    def __init__(s, _):
        super().__init__()
        s._ = _

    def forward(s, _):
        return s._

    def extra_repr(s):
        return str(s._.shape if hasattr(s._, "shape") else s._)


def Weight(*_):
    return Seq(Const(Parameter(init.kaiming_uniform_(torch.empty(_[:-2] + _[:-3:-1]), a=5**0.5))), T)


def Bias(*_):
    return Const(Parameter(torch.zeros(_)))


def Seq(*_):
    _ = [_ for _ in _ for _ in (_ if type(_) == Sequential else [_]) if _ and not isinstance(_, Identity)]
    return Sequential(*_) if len(_) > 1 else _[0] if _ else Identity()


def Add(*_, zip=False):
    _ = [_ for _ in _ for _ in (_ if isinstance(_, Map) and _.name == "Add" else [_]) if _]
    return Map(partial(functools.reduce, _pre=[torch.add]), *_, name="Add", zip=zip) if len(_) > 1 or zip else _[0]


def Mul(*_, zip=False):
    _ = [_ for _ in _ for _ in (_ if isinstance(_, Map) and _.name == "Mul" else [_]) if _]
    return Map(partial(functools.reduce, _pre=[torch.mul]), *_, name="Mul", zip=zip) if len(_) > 1 or zip else _[0]


def Dot(*_, zip=False):
    _ = [_ for _ in _ for _ in (_ if isinstance(_, Map) and _.name == "Dot" else [_]) if _]
    return Map(partial(functools.reduce, _pre=[torch.matmul]), *_, name="Dot", zip=zip) if len(_) > 1 or zip else _[0]


def Cat(*_, dim=-1, zip=False):
    _ = [_ for _ in _ for _ in (_ if isinstance(_, Map) and _.name == "Cat" else [_]) if _]
    return Map(partial(torch.cat, dim), *_, name="Cat", zip=zip) if len(_) > 1 or zip else _[0]


def Einsum(f, *_, zip=False):
    return Map(partial(torch.einsum, _pre=[f], _unpak=True), *_, zip=zip)


def Res(*_):
    return Add(Identity(), Seq(*_))


def Nn(i, h, o=0, bias=False, bn=Identity, act=ReLU(True)):
    return Seq(
        Linear(i, h, bn == Identity) if i else Identity(),
        bn(h),
        act,
        Linear(h, o, bias) if o else Identity(),
    )


def SwiGLU(i, h, o=0, bias=False, bn=Identity, act=ReLU(True)):
    return Seq(
        Mul(
            Seq(
                Linear(i, h, bn == Identity) if i else Identity(),
                bn(h),
                act,
            ),
            Linear(i, h) if i else Identity(),
        ),
        Linear(h, o, bias) if o else Identity(),
    )


def Attention(i, h, o=0, pos=Identity(), nn=[Nn] * 4, bilin=False, bias=False):
    i += getattr(pos, "n", 0)
    return Seq(
        pos,
        Add(
            Dot(
                Seq(
                    Dot(
                        nn[0](i, h[0], h[4], bilin),
                        Seq(
                            nn[1](i, h[1] if bilin else h[4], h[4] if bilin else 0, True),
                            T,
                        ),
                    ),
                    Softmax(-1),
                ),
                nn[2](i, h[2]),
            ),
            Linear(i, h[2], o),
        ),
        nn[3](h[2], h[3], o, bias) if o else Identity(),
    )


def MultiheadAttention(i, h, o=0, pos=Identity(), head=1, nn=[Nn] * 4, bilin=False, bias=False):
    i += getattr(pos, "n", 0)
    return Seq(
        pos,
        Add(
            Seq(
                Einsum(
                    "...Llh,...lhk->...Lhk",
                    Einsum(
                        "...Lhk,...lhk->...Llh",
                        Seq(nn[0](i, head * h[0], head * h[4], bilin), Unflatten(-1, (head, h[4]))),
                        Seq(nn[1](i, head * h[1], head * h[4], bilin), Unflatten(-1, (head, h[4]))),
                    ),
                    Seq(nn[2](i, head * h[2], head * h[5]), Unflatten(-1, (head, h[5]))),
                ),
                Flatten(-2),
            ),
            Linear(i, head * h[5], o),
        ),
        nn[3](head * h[5], head * h[3], o) if o else Identity(),
    )


T = F(torch.transpose, -2, -1)
T2 = F(torch.transpose, -3, -2)
T13 = F(torch.transpose, -3, -1)