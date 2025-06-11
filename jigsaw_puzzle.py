import numpy as np
from PIL import Image, ImageDraw
import os
from typing import List, Tuple, Dict
import random
import math


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
        # ensuring matching edges have opposite values
        
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
                       tab_direction: int, is_horizontal: bool) -> List[Tuple[float, float]]:
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
            
            if tab_direction == 1:  # Tab out (up)
                # Bezier curve from straight edge to neck start
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (straight_end, start_y),
                        (neck_start_x - edge_length * 0.03, start_y),
                        (neck_start_x, start_y - tab_height * 0.1),
                        (neck_start_x, start_y - tab_height * 0.3))
                    path.append(pt)
                
                # Bezier curve for neck to bulb transition
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_start_x, start_y - tab_height * 0.3),
                        (neck_start_x, start_y - tab_height * 0.5),
                        (bulb_start_x, start_y - tab_height * 0.7),
                        (bulb_start_x, start_y - tab_height * 0.85))
                    path.append(pt)
                
                # Bezier curve for bulb top (rounded)
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_start_x, start_y - tab_height * 0.85),
                        (bulb_start_x, start_y - tab_height),
                        (bulb_end_x, start_y - tab_height),
                        (bulb_end_x, start_y - tab_height * 0.85))
                    path.append(pt)
                
                # Bezier curve for bulb to neck transition (other side)
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_end_x, start_y - tab_height * 0.85),
                        (bulb_end_x, start_y - tab_height * 0.7),
                        (neck_end_x, start_y - tab_height * 0.5),
                        (neck_end_x, start_y - tab_height * 0.3))
                    path.append(pt)
                
                # Bezier curve from neck end back to straight edge
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_end_x, start_y - tab_height * 0.3),
                        (neck_end_x, start_y - tab_height * 0.1),
                        (neck_end_x + edge_length * 0.03, start_y),
                        (neck_end_x + edge_length * 0.05, start_y))
                    path.append(pt)
                    
            else:  # Blank in (down) - mirror of tab
                # Similar curves but going down instead of up
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (straight_end, start_y),
                        (neck_start_x - edge_length * 0.03, start_y),
                        (neck_start_x, start_y + tab_height * 0.1),
                        (neck_start_x, start_y + tab_height * 0.3))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_start_x, start_y + tab_height * 0.3),
                        (neck_start_x, start_y + tab_height * 0.5),
                        (bulb_start_x, start_y + tab_height * 0.7),
                        (bulb_start_x, start_y + tab_height * 0.85))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_start_x, start_y + tab_height * 0.85),
                        (bulb_start_x, start_y + tab_height),
                        (bulb_end_x, start_y + tab_height),
                        (bulb_end_x, start_y + tab_height * 0.85))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (bulb_end_x, start_y + tab_height * 0.85),
                        (bulb_end_x, start_y + tab_height * 0.7),
                        (neck_end_x, start_y + tab_height * 0.5),
                        (neck_end_x, start_y + tab_height * 0.3))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (neck_end_x, start_y + tab_height * 0.3),
                        (neck_end_x, start_y + tab_height * 0.1),
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
            
            if tab_direction == 1:  # Tab out (left)
                # Similar Bezier curves but for vertical orientation
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (start_x, straight_end),
                        (start_x, neck_start_y - edge_length * 0.03),
                        (start_x - tab_width * 0.1, neck_start_y),
                        (start_x - tab_width * 0.3, neck_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x - tab_width * 0.3, neck_start_y),
                        (start_x - tab_width * 0.5, neck_start_y),
                        (start_x - tab_width * 0.7, bulb_start_y),
                        (start_x - tab_width * 0.85, bulb_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x - tab_width * 0.85, bulb_start_y),
                        (start_x - tab_width, bulb_start_y),
                        (start_x - tab_width, bulb_end_y),
                        (start_x - tab_width * 0.85, bulb_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x - tab_width * 0.85, bulb_end_y),
                        (start_x - tab_width * 0.7, bulb_end_y),
                        (start_x - tab_width * 0.5, neck_end_y),
                        (start_x - tab_width * 0.3, neck_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x - tab_width * 0.3, neck_end_y),
                        (start_x - tab_width * 0.1, neck_end_y),
                        (start_x, neck_end_y + edge_length * 0.03),
                        (start_x, neck_end_y + edge_length * 0.05))
                    path.append(pt)
                    
            else:  # Blank in (right)
                for t in np.linspace(0, 1, 10):
                    pt = self.bezier_curve(t,
                        (start_x, straight_end),
                        (start_x, neck_start_y - edge_length * 0.03),
                        (start_x + tab_width * 0.1, neck_start_y),
                        (start_x + tab_width * 0.3, neck_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + tab_width * 0.3, neck_start_y),
                        (start_x + tab_width * 0.5, neck_start_y),
                        (start_x + tab_width * 0.7, bulb_start_y),
                        (start_x + tab_width * 0.85, bulb_start_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 15)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + tab_width * 0.85, bulb_start_y),
                        (start_x + tab_width, bulb_start_y),
                        (start_x + tab_width, bulb_end_y),
                        (start_x + tab_width * 0.85, bulb_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + tab_width * 0.85, bulb_end_y),
                        (start_x + tab_width * 0.7, bulb_end_y),
                        (start_x + tab_width * 0.5, neck_end_y),
                        (start_x + tab_width * 0.3, neck_end_y))
                    path.append(pt)
                
                for t in np.linspace(0, 1, 10)[1:]:
                    pt = self.bezier_curve(t,
                        (start_x + tab_width * 0.3, neck_end_y),
                        (start_x + tab_width * 0.1, neck_end_y),
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
        
        # Top edge
        top_edge = self.v_edges[row, col]
        top_path = self.create_tab_path(left, top, right, top, top_edge, True)
        outline.extend(top_path)
        
        # Right edge
        right_edge = self.h_edges[row, col + 1]
        right_path = self.create_tab_path(right, top, right, bottom, right_edge, False)
        outline.extend(right_path[1:])  # Skip duplicate corner point
        
        # Bottom edge
        bottom_edge = -self.v_edges[row + 1, col]  # Invert for matching
        bottom_path = self.create_tab_path(right, bottom, left, bottom, bottom_edge, True)
        outline.extend(bottom_path[1:])
        
        # Left edge
        left_edge = -self.h_edges[row, col]  # Invert for matching
        left_path = self.create_tab_path(left, bottom, left, top, left_edge, False)
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
    # Create jigsaw puzzle from lena.png
    puzzle = JigsawPuzzle("lena.png", rows=4, cols=4)
    
    # Generate puzzle pieces
    puzzle.generate_puzzle()
    
    # Verify pieces fit together
    puzzle.verify_puzzle()
    
    print("\nPuzzle generation complete!")
    print("Pieces saved in 'puzzle_pieces' directory")
    print("Check 'verification.png' to see pieces reassembled")


if __name__ == "__main__":
    main()