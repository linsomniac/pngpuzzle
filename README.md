# PNG Puzzle Generator

A Python tool that generates jigsaw puzzle pieces from any image with realistic interlocking tabs and blanks.

## Features

- **Realistic puzzle pieces**: Creates pieces with smooth Bézier curve tabs and blanks that interlock perfectly
- **Customizable grid**: Specify any number of rows and columns
- **Transparent backgrounds**: Puzzle pieces saved as PNG files with proper transparency
- **Verification**: Automatically generates a reassembled image to verify pieces fit correctly
- **Debug mode**: Includes edge matching verification for development

## Requirements

- Python 3.6+
- PIL (Pillow)
- NumPy

Install dependencies:
```bash
pip install pillow numpy
```

## Usage

### Basic usage:
```bash
python jigsaw_puzzle.py image.png
```

### Custom grid size:
```bash
python jigsaw_puzzle.py image.png -r 3 -c 3    # 3x3 puzzle
python jigsaw_puzzle.py image.png -r 6 -c 4    # 6x4 puzzle
```

### Specify output directory:
```bash
python jigsaw_puzzle.py image.png -o my_puzzle
```

### Skip verification image:
```bash
python jigsaw_puzzle.py image.png --no-verify
```

### Debug mode (shows edge matching info):
```bash
python jigsaw_puzzle.py image.png --debug
```

## Command Line Options

- `image`: Path to input image file
- `-r, --rows`: Number of puzzle rows (default: 4)
- `-c, --cols`: Number of puzzle columns (default: 4)
- `-o, --output`: Output directory name (default: puzzle_pieces)
- `--no-verify`: Skip creating verification image
- `--debug`: Show edge matching debug information

## Output

The tool generates:
- Individual puzzle piece PNG files: `piece_row{N}_col{M}.png`
- `verification.png`: Reassembled puzzle to verify correctness
- Console output showing generation progress

## Technical Details

- Uses cubic Bézier curves for smooth, realistic puzzle piece edges
- Implements proper tab/blank complementarity between adjacent pieces
- Handles edge cases for border pieces (straight edges)
- Includes padding in piece images to accommodate protruding tabs
- Uses alpha compositing for proper transparency handling

## Example

Input: `lena.png` (any image format supported by PIL)
Output: Individual puzzle pieces + verification image showing the reassembled puzzle
