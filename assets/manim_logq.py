"""The logQ correction: why in-batch softmax punishes the popular.

    manim -qh --format=mp4 assets/manim_logq.py LogQCorrection
"""

import numpy as np
from manim import (
    config, Scene, VGroup, Square, Text, MathTex, FadeIn, FadeOut, Create,
    Flash, Transform, RIGHT, LEFT, UP, DOWN,
)

BG = "#171019"
INK = "#efe8f4"
FAINT = "#94849e"
GRID = "#392c40"
ROSE = "#fb7185"
GOLD = "#facc15"
TEAL = "#2dd4bf"
RED = "#f87171"

config.background_color = BG
MONO = "Menlo"


class LogQCorrection(Scene):
    def construct(self):
        title = Text("in-batch negatives punish the popular, unless you fix it",
                     font=MONO, font_size=22, color=INK).to_edge(UP, buff=0.4)
        self.play(FadeIn(title, shift=DOWN * 0.2), run_time=0.8)

        # a batch: users x items grid, diagonal = positives
        n = 5
        cell = 0.62
        grid = VGroup()
        origin = LEFT * 4.4 + UP * 1.15
        for r in range(n):
            for c in range(n):
                sq = Square(cell, stroke_color=GRID, stroke_width=1.6,
                            fill_color=(GOLD if r == c else BG),
                            fill_opacity=(0.7 if r == c else 1.0))
                sq.move_to(origin + RIGHT * c * cell + DOWN * r * cell)
                grid.add(sq)
        gl = Text("one batch: rows are users, the diagonal is their true item",
                  font=MONO, font_size=14, color=FAINT).next_to(grid, DOWN, buff=0.3)
        self.play(FadeIn(grid), FadeIn(gl), run_time=1.0)

        # a popular item's column: it shows up in EVERY row as a negative
        col = 2
        colcells = VGroup(*[grid[r * n + col] for r in range(n) if r != col])
        pop_label = Text("a popular movie lands in many batches...", font=MONO,
                         font_size=15, color=ROSE).shift(RIGHT * 3.3 + UP * 1.9)
        self.play(*[c.animate.set_fill(ROSE, opacity=0.55) for c in colcells],
                  FadeIn(pop_label), run_time=1.0)
        push = Text("...and every appearance off the diagonal\npushes its score DOWN",
                    font=MONO, font_size=15, color=RED).shift(RIGHT * 3.3 + UP * 1.0)
        self.play(FadeIn(push), run_time=0.8)
        verdict = Text("result here: recall 0.018, worse than counting",
                       font=MONO, font_size=14, color=RED).shift(RIGHT * 3.3 + UP * 0.25)
        self.play(FadeIn(verdict), run_time=0.7)
        self.wait(0.8)

        # the correction
        eq = MathTex(r"\ell_{ij} = \frac{u_i \cdot v_j}{\tau} - \log q_j",
                     font_size=40, color=GOLD).shift(RIGHT * 3.3 + DOWN * 0.9)
        eqnote = Text("subtract each item's log sampling probability:\n"
                      "being popular is no longer a crime",
                      font=MONO, font_size=14, color=FAINT).shift(RIGHT * 3.3 + DOWN * 1.85)
        self.play(FadeIn(eq), run_time=0.9)
        self.play(*[c.animate.set_fill(BG, opacity=1.0) for c in colcells],
                  FadeIn(eqnote), run_time=1.0)
        after = Text("same model, one line later: recall 0.085,\n46% above the counter",
                     font=MONO, font_size=15, color=GOLD).shift(RIGHT * 3.3 + DOWN * 2.75)
        self.play(FadeOut(verdict), FadeIn(after),
                  Flash(eq, color=GOLD, line_length=0.3), run_time=1.0)
        self.wait(2.2)
