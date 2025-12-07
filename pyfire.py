##
##
##   8888888b.            .d888 d8b
##   888   Y88b          d88P"  Y8P
##   888    888          888
##   888   d88P 888  888 888888 888 888d888  .d88b.
##   8888888P"  888  888 888    888 888P"   d8P  Y8b
##   888        888  888 888    888 888     88888888
##   888        Y88b 888 888    888 888     Y8b.
##   888         "Y88888 888    888 888      "Y8888
##                   888
##              Y8b d88P
##               "Y88P"             github.com/rykrr
##
################################################################################
################################################################################

from argparse import ArgumentParser
from time import sleep
import asyncio
import time
import os

from scipy.signal import convolve2d as conv2d
from numpy.random import rand
import numpy as np


# Argument Parser
################################################################################

arg_parser = ArgumentParser()
arg_parser.add_argument(
    "-t", "--timing", type=float, default=0.03, help="Frame time in seconds"
)
arg_parser.add_argument(
    "-s", "--strength", type=float, default=1.0, help="Maximum fire strength"
)
arg_parser.add_argument(
    "-a",
    "--alpha",
    type=float,
    default=1.0,
    help="Constant applied to kernel for decay",
)
arg_parser.add_argument(
    "-b",
    "--beta",
    type=float,
    default=1.0,
    help="Constant applied to the entire heatmap",
)
arg_parser.add_argument(
    "-e",
    "--epsilon",
    type=float,
    default=1.0,
    help="Balance between (-ab) and (-d) coefficients (default 100% -d)",
)
arg_parser.add_argument(
    "-d",
    "--delta",
    type=float,
    default=5,
    help="Cooling coefficient (make cold regions colder)",
)
arg_parser.add_argument(
    "--clip",
    type=int,
    default=3,
    help="Remove the bottom n rows (they're too chaotic)",
)
arg_parser.add_argument(
    "--margin", type=int, default=5, help="Margins from side"
)
arg_parser.add_argument(
    "--mode", type=str, default="fade", help="Choose charset"
)
arg_parser.add_argument(
    "--buffer", type=int, default=10, help="Render buffer size"
)
arg_parser.add_argument(
    "--cache", type=int, default=0, help="Use a fixed set of ignitions"
)
args = arg_parser.parse_args()

assert args.clip > 0, "--clip must be at least 1"


# Terminal Constants and ANSI Codes
################################################################################

TERMSIZE = os.get_terminal_size()
WIDTH = TERMSIZE[0]
HEIGHT = TERMSIZE[1] + args.clip
MARGIN = args.margin

HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR = "\x1b[H"

ESC = "\x1b[38;5;{}m"
RST = "\x1b[0m"


# Style settings
################################################################################

CHARSETS = {
    "numbers": "0123456789",
    "dither": " ░░░░░░░░░",
    "fade": " ░▒▓██████",
    "chars": " .,;:!!!!!",
    "compat": " ░▒▓█▓▒░██",
}

if args.mode not in CHARSETS:
    raise KeyError(f"{args.mode} not foun. Available: {CHARSETS.keys()}")

CHARS = CHARSETS[args.mode]
COLOURS = [0, 196, 202, 208, 214, 226, 226, 226, 226, 226, 226]

if args.mode == "compat":
    COLOURS = [0, 196, 196, 214, 214, 214, 214, 214, 226, 226, 226]

CHARS = [ESC.format(colour) + char for colour, char in zip(COLOURS, CHARS)]

to_chars = np.vectorize(lambda i: CHARS[int(i)], otypes=[str])


# Kernel / Filter
################################################################################

KERNEL = np.array(
    [
        [1, 2, 3, 2, 1],
        [1, 3, 4, 3, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

KERNEL = (KERNEL / KERNEL.sum()) * args.alpha


# Main
################################################################################


async def generate(queue: asyncio.Queue):

    heatmap = np.zeros((HEIGHT, WIDTH))

    ignitions = None
    hashes = {}
    cache = []

    start_time = time.perf_counter()

    if args.cache:
        # cache = np.zeros((int(args.cache * 1.5), HEIGHT, WIDTH))
        seeds = args.strength * abs(rand(args.cache, WIDTH - (2 * MARGIN)))
        seed_idx = 0

    while True:
        heatmap[:-1] = conv2d(
            heatmap, KERNEL, mode="same", boundary="fill", fillvalue=0.0
        )[:-1]
        heatmap *= args.beta

        heatmap = (1 - args.epsilon) * heatmap + (
            args.epsilon * heatmap * np.tanh(args.delta * heatmap)
        )

        if not args.cache:
            seed = args.strength * abs(rand(1, WIDTH - (2 * MARGIN)))
        else:
            seed = seeds[seed_idx]
            seed_idx += 1
            if seed_idx == args.cache:
                seed_idx = 0

        heatmap[-1][MARGIN:-MARGIN] = seed

        chars = to_chars(np.floor(heatmap * 10))
        frame = "\n".join(["".join(d) for d in chars.tolist()[: -args.clip]])

        if args.cache:
            h = hash(frame)
            if h in hashes and len(hashes) > 5:
                cache = cache[hashes[h] + 1 :]
                assert cache[0] == frame
                break

            cache.append(frame)
            hashes[h] = len(hashes)

        await queue.put(frame)

    if not cache:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return

    while True:
        for frame in cache:
            await queue.put(frame)


async def amain():
    queue = asyncio.Queue(maxsize=args.buffer)
    generator = asyncio.create_task(generate(queue))

    while True:
        frame = await queue.get()
        print(CLEAR + frame, end="")
        queue.task_done()
        await asyncio.sleep(args.timing)


print(HIDE_CURSOR)
try:
    asyncio.run(amain())
except:
    pass
print(SHOW_CURSOR + RST)
