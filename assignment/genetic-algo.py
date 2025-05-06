# This code for generic algorithm to solve the knapsack problem
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Item:
    def __init__(self, width, height, name=None):
        self.width = width
        self.height = height
        self.name = name
        self.placed = False
        self.x = None
        self.y = None

    def rotate(self):
        self.width, self.height = self.height, self.width

    def place(self, x, y):
        self.x = x
        self.y = y
        self.placed = True


class Container:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.items = []
        self.occupied = [[False for _ in range(width)] for _ in range(height)]

    def can_place(self, item, x, y):
        # Check if item fits at (x,y) without overlapping
        if x + item.width > self.width or y + item.height > self.height:
            return False

        for i in range(x, x + item.width):
            for j in range(y, y + item.height):
                if self.occupied[j][i]:
                    return False
        return True

    def place_item(self, item, x, y):
        if not self.can_place(item, x, y):
            return False

        # Mark the space as occupied
        for i in range(x, x + item.width):
            for j in range(y, y + item.height):
                self.occupied[j][i] = True

        item.place(x, y)
        self.items.append(item)
        return True

    def find_best_spot(self, item):
        # Try all possible positions (bottom-left heuristic)
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

        if best_position:
            if best_rotation:
                item.rotate()
            return best_position
        return None

    def visualize(self):
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
        plt.title('Filling the box with items')
        plt.show()


def pack_items(container_width, container_height, items):
    container = Container(container_width, container_height)

    # Sort items by area in descending order (largest first)
    sorted_items = sorted(items, key=lambda x: x.width * x.height, reverse=True)

    for item in sorted_items:
        if item.placed:
            continue

        position = container.find_best_spot(item)
        if position:
            x, y = position
            container.place_item(item, x, y)

    return container


# Example usage
if __name__ == "__main__":
    # Define container size
    container_width = 10
    container_height = 10

    # Create some items
    items = [
        Item(4, 2, "A"),
        Item(3, 3, "B"),
        Item(2, 5, "C"),
        Item(3, 2, "D"),
        Item(2, 4, "E"),
        Item(1, 6, "F"),
        Item(5, 1, "G"),
        Item(4, 2, "H"),
        Item(8, 7, "I"),
    ]

    # Pack items into container
    container = pack_items(container_width, container_height, items)

    # Visualize the result
    container.visualize()

    # Print placement information
    print("Item placements:")
    for item in container.items:
        print(f"{item.name or 'Item'}: Position ({item.x}, {item.y}), Size {item.width}x{item.height}")