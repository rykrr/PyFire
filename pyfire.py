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
import os

from scipy.signal import convolve2d as conv2d
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
    "--compat", action="store_true", help="Force compatibility set for vTTY"
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

CHARS = "0123456789A"
CHARS = " .,;:!!!!!!"
CHARS = " ░▒▓███████"
CHARS = " ░░░░░░░░░░"
COLOURS = [0, 196, 202, 208, 214, 226, 226, 226, 226, 226, 226]

if args.compat:
    CHARS = " ░▒▓█▓▒░███"
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

# Kernel / Filter
################################################################################


async def framegen(queue: asyncio.Queue):

    heatmap = np.zeros((HEIGHT, WIDTH))

    while True:
        heatmap[:-1] = conv2d(
            heatmap, KERNEL, mode="same", boundary="fill", fillvalue=0.0
        )[:-1]
        heatmap *= args.beta

        heatmap = (1 - args.epsilon) * heatmap + (
            args.epsilon * heatmap * np.tanh(args.delta * heatmap)
        )

        row = args.strength * abs(np.random.rand(1, WIDTH - (2 * MARGIN)))
        heatmap[-1][MARGIN:-MARGIN] = row

        chars = to_chars(np.floor(heatmap * 10))

        frame = "\n".join(["".join(d) for d in chars.tolist()[: -args.clip]])
        await queue.put(frame)


async def amain():
    queue = asyncio.Queue(maxsize=5)
    generator = asyncio.create_task(framegen(queue))

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
