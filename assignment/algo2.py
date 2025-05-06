import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional
import copy


class Item:
    def __init__(self, width: int, height: int, name: str = None):
        self.width = width
        self.height = height
        self.name = name
        self.placed = False
        self.x = None
        self.y = None

    def rotate(self) -> None:
        self.width, self.height = self.height, self.width

    def place(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.placed = True

    @property
    def area(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return f"Item({self.width}x{self.height}, '{self.name}')"


class Container:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items = []
        self.occupied = [[False for _ in range(width)] for _ in range(height)]
        self.skyline = []  # For skyline algorithm
        self.reset()

    def reset(self) -> None:
        """Reset the container for a new packing attempt"""
        self.items = []
        self.occupied = [[False for _ in range(self.width)] for _ in range(self.height)]
        self.skyline = [{'x': 0, 'y': 0, 'width': self.width}]

    def can_place(self, item: Item, x: int, y: int) -> bool:
        """Check if item fits at (x,y) without overlapping"""
        if x + item.width > self.width or y + item.height > self.height:
            return False

        for i in range(x, x + item.width):
            for j in range(y, y + item.height):
                if self.occupied[j][i]:
                    return False
        return True

    def place_item(self, item: Item, x: int, y: int) -> bool:
        """Place item at specified position if possible"""
        if not self.can_place(item, x, y):
            return False

        # Mark the space as occupied
        for i in range(x, x + item.width):
            for j in range(y, y + item.height):
                self.occupied[j][i] = True

        item.place(x, y)
        self.items.append(item)
        return True

    def find_best_spot(self, item: Item) -> Optional[Tuple[int, int]]:
        """Find best spot using bottom-left heuristic"""
        best_score = None
        best_position = None
        best_rotation = False

        for rotation in [False, True]:
            if rotation:
                item.rotate()

            for y in range(self.height - item.height + 1):
                for x in range(self.width - item.width + 1):
                    if self.can_place(item, x, y):
                        # Score based on how "tight" the fit is
                        score = (x + item.width) * (y + item.height)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_position = (x, y)
                            best_rotation = rotation

            if rotation:
                item.rotate()  # Rotate back

        return best_position

    def pack_bottom_left(self, items: List[Item]) -> None:
        """Basic bottom-left packing algorithm"""
        self.reset()
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)

        for item in sorted_items:
            position = self.find_best_spot(item)
            if position:
                x, y = position
                self.place_item(item, x, y)

    def pack_maximal_rectangles(self, items: List[Item]) -> None:
        """Maximal rectangles algorithm"""
        self.reset()
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)
        free_rectangles = [{'x': 0, 'y': 0, 'width': self.width, 'height': self.height}]

        for item in sorted_items:
            # Try both orientations
            for rotation in [False, True]:
                if rotation:
                    item.rotate()

                best_rect = None
                best_score = float('inf')

                for rect in free_rectangles:
                    if rect['width'] >= item.width and rect['height'] >= item.height:
                        # Score based on remaining space
                        leftover_horiz = rect['width'] - item.width
                        leftover_vert = rect['height'] - item.height
                        score = min(leftover_horiz, leftover_vert)

                        if score < best_score:
                            best_score = score
                            best_rect = rect

                if best_rect:
                    # Place the item
                    x, y = best_rect['x'], best_rect['y']
                    self.place_item(item, x, y)

                    # Update free rectangles
                    new_free_rectangles = []
                    for rect in free_rectangles:
                        if rect == best_rect:
                            # Split the remaining space
                            if item.width < rect['width']:
                                new_free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': rect['width'] - item.width,
                                    'height': item.height
                                })
                            if item.height < rect['height']:
                                new_free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': rect['width'],
                                    'height': rect['height'] - item.height
                                })
                            if item.width < rect['width'] and item.height < rect['height']:
                                new_free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y + item.height,
                                    'width': rect['width'] - item.width,
                                    'height': rect['height'] - item.height
                                })
                        else:
                            new_free_rectangles.append(rect)

                    free_rectangles = new_free_rectangles
                    break

                if rotation:
                    item.rotate()

    def pack_skyline(self, items: List[Item]) -> None:
        """Skyline packing algorithm"""
        self.reset()
        sorted_items = sorted(items, key=lambda x: x.height, reverse=True)

        for item in sorted_items:
            best_x = None
            best_y = float('inf')
            best_rotation = False

            # Try both orientations
            for rotation in [False, True]:
                if rotation:
                    item.rotate()

                # Find the minimal y position where the item fits
                for i in range(len(self.skyline)):
                    x = self.skyline[i]['x']
                    width = self.skyline[i]['width']
                    y = self.skyline[i]['y']

                    if width >= item.width:
                        # Check if we can place it here
                        current_y = y
                        valid = True

                        # Check if the item fits in the remaining skyline segments
                        remaining_width = item.width - width
                        j = i + 1

                        while remaining_width > 0 and j < len(self.skyline):
                            if self.skyline[j]['y'] != y:
                                valid = False
                                break
                            remaining_width -= self.skyline[j]['width']
                            j += 1

                        if valid and remaining_width <= 0 and current_y + item.height <= self.height:
                            if current_y < best_y:
                                best_y = current_y
                                best_x = x
                                best_rotation = rotation

                if rotation:
                    item.rotate()

            if best_x is not None:
                if best_rotation:
                    item.rotate()

                # Place the item
                self.place_item(item, best_x, best_y)

                # Update the skyline
                new_skyline = []
                i = 0
                skyline_length = len(self.skyline)

                while i < skyline_length:
                    segment = self.skyline[i]

                    if segment['x'] < best_x and segment['x'] + segment['width'] > best_x:
                        # Split the segment
                        left_width = best_x - segment['x']
                        if left_width > 0:
                            new_skyline.append({
                                'x': segment['x'],
                                'y': segment['y'],
                                'width': left_width
                            })

                        # Middle part (occupied by the item)
                        new_skyline.append({
                            'x': best_x,
                            'y': best_y + item.height,
                            'width': item.width
                        })

                        # Right part
                        right_width = segment['x'] + segment['width'] - (best_x + item.width)
                        if right_width > 0:
                            new_skyline.append({
                                'x': best_x + item.width,
                                'y': segment['y'],
                                'width': right_width
                            })

                        i += 1
                        # Skip segments covered by the item
                        remaining_width = item.width - segment['width']
                        while remaining_width > 0 and i < skyline_length:
                            remaining_width -= self.skyline[i]['width']
                            i += 1
                    else:
                        new_skyline.append(segment)
                        i += 1

                self.skyline = new_skyline

    def pack_guillotine(self, items: List[Item], split_method='shorter_axis') -> None:
        """Guillotine packing algorithm with different split methods"""
        self.reset()
        sorted_items = sorted(items, key=lambda x: x.area, reverse=True)
        free_rectangles = [{'x': 0, 'y': 0, 'width': self.width, 'height': self.height}]

        for item in sorted_items:
            # Try both orientations
            for rotation in [False, True]:
                if rotation:
                    item.rotate()

                best_rect = None
                best_fit = None

                for rect in free_rectangles:
                    if rect['width'] >= item.width and rect['height'] >= item.height:
                        # Calculate different fit metrics
                        area_fit = rect['width'] * rect['height'] - item.area
                        short_side_fit = min(rect['width'] - item.width, rect['height'] - item.height)
                        long_side_fit = max(rect['width'] - item.width, rect['height'] - item.height)

                        # Choose the best fit based on metrics
                        if best_fit is None or area_fit < best_fit[0]:
                            best_fit = (area_fit, short_side_fit, long_side_fit)
                            best_rect = rect

                if best_rect:
                    # Place the item
                    x, y = best_rect['x'], best_rect['y']
                    self.place_item(item, x, y)

                    # Remove the used rectangle
                    free_rectangles.remove(best_rect)

                    # Split the remaining space
                    remaining_width = best_rect['width'] - item.width
                    remaining_height = best_rect['height'] - item.height

                    if split_method == 'shorter_axis':
                        # Split along the shorter remaining axis
                        if remaining_width <= remaining_height:
                            # Split horizontally
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': best_rect['height']
                                })
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': item.width,
                                    'height': remaining_height
                                })
                        else:
                            # Split vertically
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': best_rect['width'],
                                    'height': remaining_height
                                })
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': item.height
                                })
                    elif split_method == 'longer_axis':
                        # Split along the longer remaining axis
                        if remaining_width >= remaining_height:
                            # Split horizontally
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': best_rect['height']
                                })
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': item.width,
                                    'height': remaining_height
                                })
                        else:
                            # Split vertically
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': best_rect['width'],
                                    'height': remaining_height
                                })
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': item.height
                                })
                    elif split_method == 'min_area':
                        # Split to minimize the area of the remaining rectangles
                        horiz_split_area = remaining_width * best_rect['height'] + item.width * remaining_height
                        vert_split_area = remaining_height * best_rect['width'] + item.height * remaining_width

                        if horiz_split_area <= vert_split_area:
                            # Split horizontally
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': best_rect['height']
                                })
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': item.width,
                                    'height': remaining_height
                                })
                        else:
                            # Split vertically
                            if remaining_height > 0:
                                free_rectangles.append({
                                    'x': x,
                                    'y': y + item.height,
                                    'width': best_rect['width'],
                                    'height': remaining_height
                                })
                            if remaining_width > 0:
                                free_rectangles.append({
                                    'x': x + item.width,
                                    'y': y,
                                    'width': remaining_width,
                                    'height': item.height
                                })

                    break

                if rotation:
                    item.rotate()

    def calculate_packing_efficiency(self) -> float:
        """Calculate what percentage of the container is filled"""
        total_area = self.width * self.height
        used_area = sum(item.area for item in self.items)
        return (used_area / total_area) * 100

    def visualize(self, title: str = "Packing Visualization") -> None:
        """Visualize the packed items in the container"""
        fig, ax = plt.subplots(figsize=(10, 10))
        container = patches.Rectangle((0, 0), self.width, self.height,
                                      linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(container)

        colors = plt.cm.get_cmap('tab20', len(self.items))

        for i, item in enumerate(self.items):
            rect = patches.Rectangle((item.x, item.y), item.width, item.height,
                                     linewidth=1, edgecolor='black',
                                     facecolor=colors(i), alpha=0.6)
            ax.add_patch(rect)
            plt.text(item.x + item.width / 2, item.y + item.height / 2,
                     item.name or f"{item.width}x{item.height}",
                     ha='center', va='center')

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"{title}\nEfficiency: {self.calculate_packing_efficiency():.2f}%")
        plt.show()


def compare_packing_algorithms(container_width: int, container_height: int, items: List[Item]) -> None:
    """Compare different packing algorithms on the same set of items"""
    original_items = copy.deepcopy(items)

    # Bottom-left packing
    container_bl = Container(container_width, container_height)
    container_bl.pack_bottom_left(copy.deepcopy(original_items))
    container_bl.visualize("Bottom-Left Packing")

    # Maximal rectangles packing
    container_mr = Container(container_width, container_height)
    container_mr.pack_maximal_rectangles(copy.deepcopy(original_items))
    container_mr.visualize("Maximal Rectangles Packing")

    # Skyline packing
    container_sky = Container(container_width, container_height)
    container_sky.pack_skyline(copy.deepcopy(original_items))
    container_sky.visualize("Skyline Packing")

    # Guillotine packing (shorter axis split)
    container_guillotine = Container(container_width, container_height)
    container_guillotine.pack_guillotine(copy.deepcopy(original_items), 'shorter_axis')
    container_guillotine.visualize("Guillotine Packing (Shorter Axis Split)")

    # Print summary
    print("\nPacking Efficiency Comparison:")
    print(f"- Bottom-Left: {container_bl.calculate_packing_efficiency():.2f}%")
    print(f"- Maximal Rectangles: {container_mr.calculate_packing_efficiency():.2f}%")
    print(f"- Skyline: {container_sky.calculate_packing_efficiency():.2f}%")
    print(f"- Guillotine: {container_guillotine.calculate_packing_efficiency():.2f}%")


# Example usage
if __name__ == "__main__":
    # Define container size
    container_width = 20
    container_height = 15

    # Create some items
    items = [
        Item(4, 6, "A"),
        Item(5, 3, "B"),
        Item(3, 4, "C"),
        Item(7, 2, "D"),
        Item(2, 5, "E"),
        Item(4, 4, "F"),
        Item(3, 3, "G"),
        Item(6, 2, "H"),
        Item(2, 7, "I"),
        Item(5, 2, "J"),
    ]

    # Compare different packing algorithms
    compare_packing_algorithms(container_width, container_height, items)