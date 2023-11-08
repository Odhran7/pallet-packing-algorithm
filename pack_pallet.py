import numpy as np
from scipy.stats import norm
import rpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from functools import reduce 

# Define the expansion rates
expansion_rates = {
    "English": 1.0,
    "French": 1.15,
    "German": 1.25,
    "Spanish": 1.1,
    "Italian": 0.9
}

# Convert distribution to kg
sample_weights_lbs = np.array([
    1.642, 1.552, 1.597, 1.576, 1.665, 1.549, 1.617, 1.577, 1.638,
    1.635, 1.543, 1.559, 1.634
])
sample_weights_kg = sample_weights_lbs * 0.453592
sample_weights_kg_std = sample_weights_kg.std()
sample_weights_kg_mean = sample_weights_kg.mean()

class Book():
    def __init__(self, language, orientation):
        self.language = language
        self.base_height = 20
        self.base_width = 15
        self.orientation = orientation  # 0 for portrait, 1 for landscape, 2 for side
        self.thickness = self.generate_thickness()
        self.weight = self.generate_weight()
    
    def generate_thickness(self):
        return np.random.normal(1.5, 0.02) * expansion_rates[self.language]

    def generate_weight(self):
        weight_distribution = norm(loc=sample_weights_kg_mean, scale=sample_weights_kg_std)
        return weight_distribution.rvs(1)[0] * expansion_rates[self.language]

class PalletLayer():
    def __init__(self, books, positions, orientation, layer_height, material='Wood'):
        self.books = books
        self.positions = positions
        self.orientation = orientation
        self.material = material
        self.layer_weight = sum(book.weight for book in books)
        self.layer_height = layer_height

    def get_price(self):
        return 2.5 if self.material == "Wood" else 10
    
class Pallet:
    def __init__(self, max_weight, width, length, height):
        self.max_weight = max_weight
        self.height = height
        self.width = width
        self.length = length
        self.current_weight = 0
        self.layers = []

    def add_layer(self, layer):
        if self.current_weight + layer.layer_weight <= self.max_weight:
            self.layers.append(layer)
            self.current_weight += layer.layer_weight
            return True
        else:
            return False
    

    def __str__(self):
        pallet_info = ""
        total_books = 0
        total_weight = 0.0
        total_height = 0.0

        for layerNum, layer in enumerate(self.layers):
            total_books += len(layer.books)
            total_weight += layer.layer_weight
            total_height += layer.layer_height

            # Assuming the layer's width and length are the same as the pallet's for simplicity
            layer_width = self.width
            layer_length = self.length

            pallet_info += f"Layer {layerNum + 1}\n"
            pallet_info += f"Orientation: {layer.orientation}\n"
            pallet_info += f"Layer dimensions (W x L): {layer_width} x {layer_length} cm\n"
            pallet_info += f"Amount of books: {len(layer.books)}\n"
            pallet_info += f"Total weight of layer: {layer.layer_weight:.2f} kg\n"
            pallet_info += f"Layer height: {layer.layer_height:.2f} cm\n"
            pallet_info += "-" * 30 + "\n"

        # Add overall stats
        pallet_info += "Overall Pallet Stats:\n"
        pallet_info += f"Total number of books: {total_books}\n"
        pallet_info += f"Total weight: {total_weight:.2f} kg\n"
        pallet_info += f"Total height: {total_height:.2f} cm\n"
        pallet_info += "-" * 30 + "\n"

        return pallet_info
    
    def visualize_pallet(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        total_height = 0
        for layer in self.layers:
            for book, position in zip(layer.books, layer.positions):
                if book.orientation == 0:  # Portrait
                    book_depth = book.base_width
                    book_width = book.thickness
                elif book.orientation == 1:  # Landscape
                    book_depth = book.thickness
                    book_width = book.base_height
                else:  # Side
                    book_depth = book.base_width
                    book_width = book.base_height
                self.draw_cuboid(position, (book_depth, book_width, book.thickness), ax, total_height)
            total_height += layer.layer_height
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('3D Visualization of Packed Pallet')
        ax.set_xlim(0, pallet.width)
        ax.set_ylim(0, pallet.length)
        ax.set_zlim(0, total_height)

        plt.show()

    def draw_cuboid(self, position, size, ax, total_height):
        x, y = position
        dx, dy, dz = size

        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]
        zz = [total_height, total_height, total_height, total_height, total_height]
        ax.plot3D(xx, yy, zz, 'gray')

        yy = [y+dy, y+dy, y+dy, y+dy, y+dy]
        zz = [total_height, total_height+dz, total_height+dz, total_height, total_height]
        ax.plot3D(xx, yy, zz, 'gray')

        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y, y, y, y]
        zz = [total_height+dz, total_height+dz, total_height+dz, total_height+dz, total_height+dz]
        ax.plot3D(xx, yy, zz, 'gray')


def generate_books(num, language):
    books = []
    for i in range(num):
        for orientation in range(3):  # 0 for portrait, 1 for landscape, 2 for side
            books.append(Book(language, orientation))
    return books


"""
Create a book object of type orientation
add to pallet 
compute rectangle score
repeat if not impossible to pack exception
calculate json obj
add to array -> return the one that maximises weight and abides by orientation condition

outer function create pallet layer obj and add to pallet obj
"""

"""
for each orientation and each increment of book compute rect score update metrics until catch
"""

"""
for each orientation calculate metrics and repeat until exception
"""

def generate_book(language, orientation):
    return Book(language, orientation)

def calculate_layer(pallet_weight, language, pallet_length, pallet_width, last_orientation=None):
    remaining_weight = pallet_weight
    best_layer = None
    best_layer_weight = 0
    books = generate_books(10000, language)
    orientations = [0, 1] if last_orientation == 2 else [0, 1, 2]

    for orientation in orientations:
        oriented_books = [book for book in books if book.orientation == orientation]

        layer_books = []
        dimensions = []
        positions = []
        layer_weight = 0
        layer_thicknesses = []  # To hold the thickness of each book

        for book in oriented_books:
            book_weight = book.weight
            if layer_weight + book_weight <= remaining_weight:
                book_dims = (book.base_width, book.base_height) if orientation in [0, 1] else (book.thickness, book.base_height)
                book_dims = (max(1, int(round(book_dims[0]))), max(1, int(round(book_dims[1]))))
                
                # Add book thickness to list
                layer_thicknesses.append(book.thickness)

                try:
                    new_positions = rpack.pack(dimensions + [book_dims], max_width=pallet_width, max_height=pallet_length)
                    dimensions.append(book_dims)
                    positions.append(new_positions[-1])
                    layer_books.append(book)
                    layer_weight += book_weight
                except rpack.PackingImpossibleError:
                    break

        if layer_books:
            if orientation == 2:  # If orientation is on the side, use max thickness
                layer_height = 15
            else:  # Otherwise, use average thickness
                layer_height = np.mean(layer_thicknesses)

            if layer_weight > best_layer_weight:
                best_layer = {
                    "orientation": orientation,
                    "books": layer_books,
                    "dimensions": dimensions,
                    "positions": positions,
                    "total_weight": layer_weight,
                    "number_of_books": len(layer_books),
                    "layer_height": layer_height
                }
                best_layer_weight = layer_weight

    return best_layer



max_layer_weight = 1500
pallet = Pallet(max_layer_weight, 100, 100, 15.24)
# layer_info = calculate_layer(pallet.max_weight, "English", pallet.length, pallet.width)
# print(layer_info)

def pack_pallet(pallet, language):
    last_orientation = None
    while pallet.current_weight < pallet.max_weight:
        layer_info = calculate_layer(pallet.max_weight - pallet.current_weight, language, pallet.length, pallet.width, last_orientation)
        if layer_info is not None:
            if last_orientation == 2 and layer_info['orientation'] == 2:
                continue
            new_layer = PalletLayer(layer_info['books'], layer_info['positions'], layer_info['orientation'], layer_info["layer_height"])
            successfully_added = pallet.add_layer(new_layer)
            if successfully_added:
                last_orientation = layer_info['orientation']
            else:
                break
        else:
            break
    return pallet

pack_pallet(pallet, 'English')
print(pallet)
pallet.visualize_pallet()
"""
$ python pallets.py
Layer 1
Orientation: 2
Amount of books: 331
Total weight of layer: 239.78 kg
Layer height: 15.00 cm
------------------------------
Layer 2
Orientation: 0
Amount of books: 30
Total weight of layer: 21.77 kg
Layer height: 1.50 cm
------------------------------
Layer 3
Orientation: 2
Amount of books: 338
Total weight of layer: 244.85 kg
Layer height: 15.00 cm
------------------------------
Layer 4
Orientation: 0
Amount of books: 30
Total weight of layer: 21.73 kg
Layer height: 1.49 cm
------------------------------
Layer 5
Orientation: 2
Amount of books: 327
Total weight of layer: 236.63 kg
Layer height: 15.00 cm
------------------------------
Layer 6
Orientation: 1
Amount of books: 30
Total weight of layer: 21.78 kg
Layer height: 1.50 cm
------------------------------
Layer 7
Orientation: 2
Amount of books: 331
Total weight of layer: 239.73 kg
Layer height: 15.00 cm
------------------------------
Layer 8
Orientation: 0
Amount of books: 30
Total weight of layer: 21.85 kg
Layer height: 1.50 cm
------------------------------
Layer 9
Orientation: 2
Amount of books: 324
Total weight of layer: 234.79 kg
Layer height: 15.00 cm
------------------------------
Layer 10
Orientation: 0
Amount of books: 30
Total weight of layer: 21.76 kg
Layer height: 1.50 cm
------------------------------
Layer 11
Orientation: 2
Amount of books: 269
Total weight of layer: 195.14 kg
Layer height: 15.00 cm
------------------------------
Overall Pallet Stats:
Total number of books: 2070
Total weight: 1499.82 kg
Total height: 97.49 cm


/ new one

$ python pallets.py
Layer 1
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 331
Total weight of layer: 240.41 kg
Layer height: 15.00 cm
------------------------------
Layer 2
Orientation: 1
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 30
Total weight of layer: 21.85 kg
Layer height: 1.50 cm
------------------------------
Layer 3
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 334
Total weight of layer: 242.21 kg
Layer height: 15.00 cm
------------------------------
Layer 4
Orientation: 0
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 30
Total weight of layer: 21.80 kg
Layer height: 1.50 cm
------------------------------
Layer 5
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 332
Total weight of layer: 241.03 kg
Layer height: 15.00 cm
------------------------------
Layer 6
Orientation: 0
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 30
Total weight of layer: 21.79 kg
Layer height: 1.50 cm
------------------------------
Layer 7
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 333
Total weight of layer: 241.47 kg
Layer height: 15.00 cm
------------------------------
Layer 8
Orientation: 0
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 30
Total weight of layer: 21.69 kg
Layer height: 1.49 cm
------------------------------
Layer 9
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 338
Total weight of layer: 244.97 kg
Layer height: 15.00 cm
------------------------------
Layer 10
Orientation: 0
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 30
Total weight of layer: 21.86 kg
Layer height: 1.50 cm
------------------------------
Layer 11
Orientation: 2
Layer dimensions (W x L): 100 x 100 cm
Amount of books: 249
Total weight of layer: 180.49 kg
Layer height: 15.00 cm
------------------------------
Overall Pallet Stats:
Total number of books: 2067
Total weight: 1499.57 kg
Total height: 97.49 cm
------------------------------
"""
