"""Minimal axis-aligned rectangle used by the simulation core.

Mirrors the pygame.Rect semantics the project relies on (collidepoint,
inflate-around-center, center properties) so the core package has no
pygame dependency and can run headless on any machine or Kaggle kernel.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def centerx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def centery(self) -> float:
        return self.y + self.h / 2.0

    def collidepoint(self, px: float, py: float) -> bool:
        return self.x <= px < self.right and self.y <= py < self.bottom

    def inflate(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x - dx / 2.0, self.y - dy / 2.0, self.w + dx, self.h + dy)

    def colliderect(self, other: "Rect") -> bool:
        return (self.x < other.right and other.x < self.right
                and self.y < other.bottom and other.y < self.bottom)

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.w, self.h)
