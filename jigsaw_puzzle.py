import numpy as np
from PIL import Image, ImageDraw
import os
from typing import List, Tuple, Dict
import random
import math
import sys
import argparse


class JigsawPuzzle:
    def __init__(self, image_path: str, rows: int = 4, cols: int = 4):
        """Initialize jigsaw puzzle with image and grid dimensions."""
        self.image = Image.open(image_path).convert("RGBA")
        self.rows = rows
        self.cols = cols
        self.width, self.height = self.image.size
        self.piece_width = self.width // cols
        self.piece_height = self.height // rows
        
        # AIDEV-NOTE: Edge connections matrix stores tab/blank info for each piece
        # 0 = straight edge, 1 = tab out, -1 = blank in
        self.h_edges = np.zeros((rows, cols + 1))  # Horizontal edges
        self.v_edges = np.zeros((rows + 1, cols))  # Vertical edges
        
        # Tab size relative to piece dimension
        self.tab_size = 0.2
        self.tab_neck_ratio = 0.3  # Neck width relative to tab diameter
        
    def generate_edge_pattern(self):
        """Generate interlocking edge pattern ensuring pieces fit together."""
        # AIDEV-NOTE: For each internal edge, randomly assign tab/blank
        # The same edge value is shared by adjacent pieces but interpreted oppositely
        
        # Generate horizontal edges (vertical connections)
        for r in range(self.rows):
            for c in range(1, self.cols):
                # Randomly assign tab (1) or blank (-1)
                self.h_edges[r, c] = random.choice([1, -1])
        
        # Generate vertical edges (horizontal connections)
        for r in range(1, self.rows):
            for c in range(self.cols):
                # Randomly assign tab (1) or blank (-1)
                self.v_edges[r, c] = random.choice([1, -1])
    
    def debug_edge_matching(self):
        """Debug function to verify adjacent pieces have complementary edges."""
        print("\nEdge Matching Debug:")
        print("=" * 50)
        
        # Let's trace a specific example
        print("\nDetailed trace for pieces [0,0] and [0,1]:")
        print(f"h_edges[0,1] = {self.h_edges[0,1]}")
        print(f"Piece [0,0] right edge value: {self.h_edges[0,1]}")
        print(f"Piece [0,1] left edge value: {-self.h_edges[0,1]}")
        
        # Check horizontal adjacencies
        for row in range(self.rows):
            for col in range(self.cols - 1):
                # Right edge of current piece
                right_edge_current = self.h_edges[row, col + 1]
                # Left edge of piece to the right (inverted)
                left_edge_next = -self.h_edges[row, col + 1]
                
                print(f"Piece [{row},{col}] right edge: {right_edge_current:+.0f} | "
                      f"Piece [{row},{col+1}] left edge: {left_edge_next:+.0f}")
                
                if right_edge_current == left_edge_next:
                    print("  ✗ ERROR: Same edge type!")
                else:
                    print("  ✓ OK: Complementary edges")
        
        # Check vertical adjacencies
        for row in range(self.rows - 1):
            for col in range(self.cols):
                # Bottom edge of current piece
                bottom_edge_current = -self.v_edges[row + 1, col]
                # Top edge of piece below
                top_edge_next = self.v_edges[row + 1, col]
                
                print(f"Piece [{row},{col}] bottom edge: {bottom_edge_current:+.0f} | "
                      f"Piece [{row+1},{col}] top edge: {top_edge_next:+.0f}")
                
                if bottom_edge_current == top_edge_next:
                    print("  ✗ ERROR: Same edge type!")
                else:
                    print("  ✓ OK: Complementary edges")
    
    def bezier_curve(self, t: float, p0: Tuple[float, float], p1: Tuple[float, float], 
                     p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate point on cubic Bezier curve at parameter t."""
        # AIDEV-NOTE: Cubic Bezier formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
        u = 1 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        
        x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
        y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
        
        return (x, y)
    
    def create_tab_path(self, start_x: float, start_y: float, end_x: float, end_y: float, 
                       tab_direction: int, is_horizontal: bool, edge_name: str) -> List[Tuple[float, float]]:
        """Create a smooth path for a single edge with optional tab/blank using Bezier curves."""
        if tab_direction == 0:  # Straight edge
            return [(start_x, start_y), (end_x, end_y)]
        
        # Calculate midpoint and tab parameters
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        path = []
        
        if is_horizontal:
            edge_length = abs(end_x - start_x)
            # AIDEV-NOTE: Adjusted tab parameters for smoother, bulbous shape
            tab_height = edge_length * self.tab_size * 0.8  # Height of the tab
            neck_width = edge_length * 0.15  # Width of the neck (narrower than bulb)
            bulb_width = edge_length * 0.25  # Width of the bulb (wider than neck)
            
            # Starting point
            path.append((start_x, start_y))
            
            # Define control points for smooth curves
            neck_start_x = mid_x - neck_width / 2
            neck_end_x = mid_x + neck_width / 2
            bulb_start_x = mid_x - bulb_width / 2
            bulb_end_x = mid_x + bulb_width / 2
            
            # Add points along the straight edge before the tab
            straight_end = neck_start_x - edge_length * 0.05
            path.append((straight_end, start_y))
            
            # AIDEV-NOTE: For horizontal edges, direction depends on which edge
            # Top edge: positive = tab up (negative Y)
            # Bottom edge: positive = tab down (positive Y)
            
            if tab_direction == 1:  # Tab out
                # Determine direction based on edge
                if edge_name == 'top':
                    y_direction = -1  # Tab goes up
                else:  # edge_name == 'bottom'
                    y_direction = 1  # Tab goes down
                    
                # Bezier curve from straight edge to neck start
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (straight_end, start_y),
                        (neck_start_x - edge_length * 0.03, start_y),
                        (neck_start_x, start_y + y_direction * tab_height * 0.1),
                        (neck_start_x, start_y + y_direction * tab_height * 0.3))
                    path.append(pt)
                
                # Bezier curve for neck to bulb transition
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_start_x, start_y + y_direction * tab_height * 0.3),
                        (neck_start_x, start_y + y_direction * tab_height * 0.5),
                        (bulb_start_x, start_y + y_direction * tab_height * 0.7),
                        (bulb_start_x, start_y + y_direction * tab_height * 0.85))
                    path.append(pt)
                
                # Bezier curve for bulb top (rounded)
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_start_x, start_y + y_direction * tab_height * 0.85),
                        (bulb_start_x, start_y + y_direction * tab_height),
                        (bulb_end_x, start_y + y_direction * tab_height),
                        (bulb_end_x, start_y + y_direction * tab_height * 0.85))
                    path.append(pt)
                
                # Bezier curve for bulb to neck transition (other side)
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_end_x, start_y + y_direction * tab_height * 0.85),
                        (bulb_end_x, start_y + y_direction * tab_height * 0.7),
                        (neck_end_x, start_y + y_direction * tab_height * 0.5),
                        (neck_end_x, start_y + y_direction * tab_height * 0.3))
                    path.append(pt)
                
                # Bezier curve from neck end back to straight edge
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_end_x, start_y + y_direction * tab_height * 0.3),
                        (neck_end_x, start_y + y_direction * tab_height * 0.1),
                        (neck_end_x + edge_length * 0.03, start_y),
                        (neck_end_x + edge_length * 0.05, start_y))
                    path.append(pt)
                    
            else:  # Blank in
                # Determine direction based on edge (opposite of tab)
                if edge_name == 'top':
                    y_direction = 1  # Groove goes down (inward)
                else:  # edge_name == 'bottom'
                    y_direction = -1  # Groove goes up (inward)
                    
                # Similar curves but going inward
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (straight_end, start_y),
                        (neck_start_x - edge_length * 0.03, start_y),
                        (neck_start_x, start_y + y_direction * tab_height * 0.1),
                        (neck_start_x, start_y + y_direction * tab_height * 0.3))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_start_x, start_y + y_direction * tab_height * 0.3),
                        (neck_start_x, start_y + y_direction * tab_height * 0.5),
                        (bulb_start_x, start_y + y_direction * tab_height * 0.7),
                        (bulb_start_x, start_y + y_direction * tab_height * 0.85))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_start_x, start_y + y_direction * tab_height * 0.85),
                        (bulb_start_x, start_y + y_direction * tab_height),
                        (bulb_end_x, start_y + y_direction * tab_height),
                        (bulb_end_x, start_y + y_direction * tab_height * 0.85))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_end_x, start_y + y_direction * tab_height * 0.85),
                        (bulb_end_x, start_y + y_direction * tab_height * 0.7),
                        (neck_end_x, start_y + y_direction * tab_height * 0.5),
                        (neck_end_x, start_y + y_direction * tab_height * 0.3))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_end_x, start_y + y_direction * tab_height * 0.3),
                        (neck_end_x, start_y + y_direction * tab_height * 0.1),
                        (neck_end_x + edge_length * 0.03, start_y),
                        (neck_end_x + edge_length * 0.05, start_y))
                    path.append(pt)
            
            # Complete the edge
            path.append((end_x, end_y))
            
        else:  # Vertical edge
            edge_length = abs(end_y - start_y)
            tab_width = edge_length * self.tab_size * 0.8
            neck_height = edge_length * 0.15
            bulb_height = edge_length * 0.25
            
            path.append((start_x, start_y))
            
            neck_start_y = mid_y - neck_height / 2
            neck_end_y = mid_y + neck_height / 2
            bulb_start_y = mid_y - bulb_height / 2
            bulb_end_y = mid_y + bulb_height / 2
            
            straight_end = neck_start_y - edge_length * 0.05
            path.append((start_x, straight_end))
            
            # AIDEV-NOTE: For vertical edges, direction depends on which edge
            # Right edge: positive = tab right (positive X)
            # Left edge: positive = tab left (negative X)
            
            if tab_direction == 1:  # Tab out
                # Determine direction based on edge
                if edge_name == 'right':
                    x_direction = 1  # Tab goes right
                else:  # edge_name == 'left'
                    x_direction = -1  # Tab goes left
                # Similar Bezier curves but for vertical orientation
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (start_x, straight_end),
                        (start_x, neck_start_y - edge_length * 0.03),
                        (start_x + x_direction * tab_width * 0.1, neck_start_y),
                        (start_x + x_direction * tab_width * 0.3, neck_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.3, neck_start_y),
                        (start_x + x_direction * tab_width * 0.5, neck_start_y),
                        (start_x + x_direction * tab_width * 0.7, bulb_start_y),
                        (start_x + x_direction * tab_width * 0.85, bulb_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.85, bulb_start_y),
                        (start_x + x_direction * tab_width, bulb_start_y),
                        (start_x + x_direction * tab_width, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.85, bulb_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.85, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.7, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.5, neck_end_y),
                        (start_x + x_direction * tab_width * 0.3, neck_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.3, neck_end_y),
                        (start_x + x_direction * tab_width * 0.1, neck_end_y),
                        (start_x, neck_end_y + edge_length * 0.03),
                        (start_x, neck_end_y + edge_length * 0.05))
                    path.append(pt)
                    
            else:  # Blank in
                # Determine direction based on edge (opposite of tab)
                if edge_name == 'right':
                    x_direction = -1  # Groove goes left (inward)
                else:  # edge_name == 'left'
                    x_direction = 1  # Groove goes right (inward)
                    
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (start_x, straight_end),
                        (start_x, neck_start_y - edge_length * 0.03),
                        (start_x + x_direction * tab_width * 0.1, neck_start_y),
                        (start_x + x_direction * tab_width * 0.3, neck_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.3, neck_start_y),
                        (start_x + x_direction * tab_width * 0.5, neck_start_y),
                        (start_x + x_direction * tab_width * 0.7, bulb_start_y),
                        (start_x + x_direction * tab_width * 0.85, bulb_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.85, bulb_start_y),
                        (start_x + x_direction * tab_width, bulb_start_y),
                        (start_x + x_direction * tab_width, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.85, bulb_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.85, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.7, bulb_end_y),
                        (start_x + x_direction * tab_width * 0.5, neck_end_y),
                        (start_x + x_direction * tab_width * 0.3, neck_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + x_direction * tab_width * 0.3, neck_end_y),
                        (start_x + x_direction * tab_width * 0.1, neck_end_y),
                        (start_x, neck_end_y + edge_length * 0.03),
                        (start_x, neck_end_y + edge_length * 0.05))
                    path.append(pt)
            
            path.append((end_x, end_y))
        
        return path
    
    def create_piece_mask(self, row: int, col: int) -> Image.Image:
        """Create a mask for a single puzzle piece."""
        # AIDEV-NOTE: Create larger canvas to accommodate tabs extending beyond piece boundaries
        padding = int(max(self.piece_width, self.piece_height) * self.tab_size)
        mask_width = self.piece_width + 2 * padding
        mask_height = self.piece_height + 2 * padding
        
        mask = Image.new('L', (mask_width, mask_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Calculate piece corners in the padded canvas
        left = padding
        top = padding
        right = left + self.piece_width
        bottom = top + self.piece_height
        
        # Build piece outline
        outline = []
        
        # AIDEV-NOTE: Edge direction convention:
        # - Positive value = tab/bulge outward from the piece
        # - Negative value = groove/indent inward into the piece
        # - Adjacent pieces share the same edge value in the matrix
        # - But they interpret it oppositely (one sees tab out, other sees groove in)
        
        # Top edge - use value as-is (positive = tab up)
        top_edge = self.v_edges[row, col]
        top_path = self.create_tab_path(left, top, right, top, top_edge, True, 'top')
        outline.extend(top_path)
        
        # Right edge - use value as-is (positive = tab right)
        right_edge = self.h_edges[row, col + 1]
        right_path = self.create_tab_path(right, top, right, bottom, right_edge, False, 'right')
        outline.extend(right_path[1:])  # Skip duplicate corner point
        
        # Bottom edge - invert because piece below sees it oppositely
        bottom_edge = -self.v_edges[row + 1, col]
        bottom_path = self.create_tab_path(right, bottom, left, bottom, bottom_edge, True, 'bottom')
        outline.extend(bottom_path[1:])
        
        # Left edge - invert because piece to left sees it oppositely
        left_edge = -self.h_edges[row, col]
        left_path = self.create_tab_path(left, bottom, left, top, left_edge, False, 'left')
        outline.extend(left_path[1:-1])  # Skip duplicate corner points
        
        # Draw filled polygon
        draw.polygon(outline, fill=255)
        
        return mask, padding
    
    def extract_piece(self, row: int, col: int) -> Image.Image:
        """Extract a single puzzle piece with transparent background."""
        # Create piece mask
        mask, padding = self.create_piece_mask(row, col)
        
        # Calculate source rectangle from original image
        src_left = col * self.piece_width
        src_top = row * self.piece_height
        src_right = min(src_left + self.piece_width, self.width)
        src_bottom = min(src_top + self.piece_height, self.height)
        
        # AIDEV-NOTE: Extract larger region to include tab areas
        extract_padding = int(max(self.piece_width, self.piece_height) * self.tab_size)
        extract_left = max(0, src_left - extract_padding)
        extract_top = max(0, src_top - extract_padding)
        extract_right = min(self.width, src_right + extract_padding)
        extract_bottom = min(self.height, src_bottom + extract_padding)
        
        # Extract region from original image
        piece_region = self.image.crop((extract_left, extract_top, extract_right, extract_bottom))
        
        # Create output image with transparency
        output_width = mask.width
        output_height = mask.height
        piece = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))
        
        # Calculate paste position
        paste_x = extract_padding - (src_left - extract_left)
        paste_y = extract_padding - (src_top - extract_top)
        
        # Paste the extracted region
        piece.paste(piece_region, (paste_x, paste_y))
        
        # Apply mask for transparency
        piece.putalpha(mask)
        
        return piece
    
    def generate_puzzle(self, output_dir: str = "puzzle_pieces"):
        """Generate all puzzle pieces and save them."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate edge pattern
        self.generate_edge_pattern()
        
        # Generate each piece
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.extract_piece(row, col)
                filename = f"piece_row{row}_col{col}.png"
                piece.save(os.path.join(output_dir, filename))
                print(f"Saved {filename}")
    
    def verify_puzzle(self, output_dir: str = "puzzle_pieces") -> Image.Image:
        """Verify that puzzle pieces fit together by reassembling them."""
        # AIDEV-NOTE: Reassemble pieces to verify they fit correctly
        # Account for overlapping tabs when positioning pieces
        
        # Calculate canvas size with overlap consideration
        canvas_width = self.width
        canvas_height = self.height
        verification = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
        
        # Load and place each piece
        for row in range(self.rows):
            for col in range(self.cols):
                filename = f"piece_row{row}_col{col}.png"
                piece_path = os.path.join(output_dir, filename)
                piece = Image.open(piece_path)
                
                # Calculate placement position
                # Account for padding in piece images
                padding = int(max(self.piece_width, self.piece_height) * self.tab_size)
                paste_x = col * self.piece_width - padding
                paste_y = row * self.piece_height - padding
                
                # Paste piece with alpha compositing
                verification.paste(piece, (paste_x, paste_y), piece)
        
        # Save verification image
        verification_path = os.path.join(output_dir, "verification.png")
        verification.save(verification_path)
        print(f"Saved verification image: {verification_path}")
        
        return verification


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a jigsaw puzzle from an image')
    parser.add_argument('image', help='Path to the input image file')
    parser.add_argument('-r', '--rows', type=int, default=4, 
                        help='Number of rows in the puzzle (default: 4)')
    parser.add_argument('-c', '--cols', type=int, default=4,
                        help='Number of columns in the puzzle (default: 4)')
    parser.add_argument('-o', '--output', default='puzzle_pieces',
                        help='Output directory for puzzle pieces (default: puzzle_pieces)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip creating verification image')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug information about edge matching')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"Error: Input file '{args.image}' not found!")
        sys.exit(1)
    
    # Check if file is an image
    try:
        # Try to open the image
        test_img = Image.open(args.image)
        test_img.close()
    except Exception as e:
        print(f"Error: Unable to open '{args.image}' as an image: {e}")
        sys.exit(1)
    
    # Create jigsaw puzzle
    print(f"Creating {args.rows}x{args.cols} jigsaw puzzle from '{args.image}'...")
    puzzle = JigsawPuzzle(args.image, rows=args.rows, cols=args.cols)
    
    # Generate puzzle pieces
    puzzle.generate_puzzle(args.output)
    
    # Debug edge matching if requested
    if args.debug:
        puzzle.debug_edge_matching()
    
    # Verify pieces fit together
    if not args.no_verify:
        puzzle.verify_puzzle(args.output)
    
    print(f"\nPuzzle generation complete!")
    print(f"Pieces saved in '{args.output}' directory")
    if not args.no_verify:
        print(f"Check '{args.output}/verification.png' to see pieces reassembled")


if __name__ == "__main__":
    main()