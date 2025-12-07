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
import signal
import os

from scipy.signal import convolve2d as conv2d
from numpy.random import rand, normal
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
    "-B",
    "--bias",
    type=float,
    default=0.0,
    help="",
)
arg_parser.add_argument(
    "--bias-dist",
    type=float,
    default=0.2,
    help="",
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
    "--mode", type=str, default="dither", help="Choose charset"
)
arg_parser.add_argument(
    "--buffer", type=int, default=10, help="Render buffer size"
)
arg_parser.add_argument(
    "--cache", type=int, default=0, help="Use a fixed set of ignitions"
)
args = arg_parser.parse_args()

assert 0 <= args.strength and args.strength <= 1, "--strength must be in [0,1]"
assert args.clip > 0, "--clip must be at least 1"


# Terminal Constants and ANSI Codes
################################################################################

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
        [1, 0, 4, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
)

KERNEL = (KERNEL / KERNEL.sum()) * args.alpha


# Main
################################################################################

SIGWINCH = False


def sigwinch():
    global SIGWINCH
    SIGWINCH = True


async def run_generate(queue: asyncio.Queue):
    global SIGWINCH

    while True:
        SIGWINCH = False
        width, height = os.get_terminal_size()
        await generate(width, height, queue)


async def generate(width: int, height: int, queue: asyncio.Queue):

    heatmap = np.zeros((height + args.clip, width))

    ignitions = None
    hashes = {}
    cache = []

    # Playing around with density curves to shape the flame
    x = np.linspace(-1, 1, width - (2 * args.margin))
    bias = (1 - args.bias_dist * x ** 2) ** 3

    if args.cache:
        seeds = abs(rand(args.cache, width - (2 * args.margin)))
        seeds = (args.bias * bias) + (1 - args.bias) * seeds
        seeds *= args.strength
        seed_idx = 0

    while not SIGWINCH:
        heatmap[:-1] = conv2d(
            heatmap, KERNEL, mode="same", boundary="fill", fillvalue=0.0
        )[:-1]
        heatmap *= args.beta

        heatmap = (1 - args.epsilon) * heatmap + (
            args.epsilon * heatmap * np.tanh(args.delta * heatmap)
        )

        if not args.cache:
            # Default noise
            seed = abs(rand(1, width - (2 * args.margin)))

            # Add shape flame
            #seed = ((1 - args.bias) * seed) + args.bias * (0.25 + (0.75 * bias) * seed )
            seed = (args.bias * bias) + (1 - args.bias) * seed

            # Adjust flame strength
            seed *= args.strength
            seed = np.clip(seed, 0, 0.9999)
        else:
            seed = seeds[seed_idx]
            seed_idx += 1
            if seed_idx == args.cache:
                seed_idx = 0

        heatmap[-1][args.margin : -args.margin] = seed

        chars = to_chars(np.floor(heatmap * 10))
        frame = "\n".join(["".join(d) for d in chars.tolist()[: -args.clip]])

        if args.cache:
            h = hash(frame)
            if h in hashes and len(hashes) > args.clip:
                index = cache.index(frame)
                cache = cache[index:]
                assert cache[0] == frame
                break

            cache.append(frame)
            hashes[h] = len(hashes)

        await queue.put(frame)

    if not cache:
        return

    while not SIGWINCH:
        for frame in cache:
            await queue.put(frame)


async def amain():
    queue = asyncio.Queue(maxsize=args.buffer)
    generator = asyncio.create_task(run_generate(queue))

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGWINCH, lambda: sigwinch())

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
