import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from sympy import symbols


# Функция для генерации координат
def generate_coordinates(shape, num_points):
    polygon = shape
    min_x, min_y, max_x, max_y = polygon.bounds
    points = []
    while len(points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = (x, y)
        if polygon.contains(Point(point)):
            points.append(point)
    return np.array(points)


# Функция для вычисления потенциалов двойного слоя
def calculate_double_layer_potential(coordinates, electrolyte_properties, boundary_conditions):
    num_points = len(coordinates)
    diel_constant = electrolyte_properties['diel_constant']
    ion_concentration = electrolyte_properties['ion_concentration']
    # temperature = electrolyte_properties['temperature']
    # potentials = np.zeros(num_points)

    stiffness_matrix = lil_matrix((num_points, num_points))
    rhs_vector = np.zeros(num_points)

    for i in range(num_points):
        xi, yi = coordinates[i]
        boundary_index = i % len(boundary_conditions)
        # boundary_type = boundary_conditions[boundary_index]['type']
        boundary_value = boundary_conditions[boundary_index]['value']

        for j in range(num_points):
            xj, yj = coordinates[j]
            distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            if i == j:
                stiffness_matrix[i, j] = 1.0
                rhs_vector[i] = boundary_value
            else:
                stiffness_matrix[i, j] = -1.0 / (4 * np.pi * diel_constant) * ion_concentration / distance
                rhs_vector[i] -= boundary_value / distance

    potentials = spsolve(stiffness_matrix.tocsr(), rhs_vector)
    return potentials


# Функция для обработки нажатия кнопки "Вычислить"
def calculate_potentials():
    global custom_formula_func, x, y
    num_points = int(num_points_entry.get())
    num_boundary_points = int(num_boundary_points_entry.get())
    num_interior_points = int(num_interior_points_entry.get())
    diel_constant = float(diel_constant_entry.get())
    ion_concentration = float(ion_concentration_entry.get())
    temperature = float(temperature_entry.get())
    shape_type = shape_var.get()

    if shape_type == 'custom':
        custom_formula_str = custom_formula_entry.get()
        x, y = symbols('x y')
        custom_formula_func = eval(custom_formula_str)
        x_values = np.linspace(-10, 10, 100)
        y_values = np.linspace(-10, 10, 100)
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        z_values = z(x_mesh, y_mesh)  # Пример функции z(x, y)
        contour = plt.contour(x_mesh, y_mesh, z_values, levels=[0], colors='black')
        shape = contour.collections[0].get_paths()[0].to_polygons()[0]
        shape = Polygon(shape)
    else:
        shape = generate_shape(shape_type)

    # Создаем форму образца на основе пользовательской формулы или выбранной геометрической формы
    if shape_type == 'custom':
        x_values = np.linspace(-10, 10, 100)
        y_values = np.linspace(-10, 10, 100)
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        z_values = z(x_mesh, y_mesh)  # Пример функции z(x, y)
        contour = plt.contour(x_values, y_values, z_values, levels=[0], colors='black')
        shape = contour.collections[0].get_paths()[0].to_polygons()[0]
        shape = Polygon(shape)

    # Генерируем координаты границы
    # boundary_coordinates = generate_coordinates(shape, num_boundary_points)

    # Генерируем координаты внутри и вне формы
    interior_coordinates = generate_coordinates(shape, num_interior_points)
    exterior_coordinates = generate_coordinates(shape, num_interior_points)

    # Генерируем случайные граничные условия
    boundary_conditions = []
    for _ in range(num_boundary_points):
        boundary_type = np.random.choice(['dirichlet', 'neumann'])
        boundary_value = np.random.uniform(-1.0, 1.0)
        boundary_conditions.append({'type': boundary_type, 'value': boundary_value})

    electrolyte_properties = {
        'diel_constant': diel_constant,
        'ion_concentration': ion_concentration,
        'temperature': temperature
    }

    # Вычисляем потенциалы внутри и вне формы
    interior_potentials = calculate_double_layer_potential(interior_coordinates, electrolyte_properties,
                                                           boundary_conditions)
    exterior_potentials = calculate_double_layer_potential(exterior_coordinates, electrolyte_properties,
                                                           boundary_conditions)

    # Выводим числовые значения ПДС
    print("Interior Double Layer Potentials:")
    print(interior_potentials)
    print("Exterior Double Layer Potentials:")
    print(exterior_potentials)

    # Визуализируем границу формы
    plt.plot(*shape.exterior.xy, color='black')
    plt.title('Shape Boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Визуализируем потенциалы внутри формы
    plt.scatter(interior_coordinates[:, 0], interior_coordinates[:, 1], c=interior_potentials, cmap='coolwarm', s=5)
    plt.title('Double Layer Potential Inside the Shape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Potential')
    plt.show()

    # Визуализируем потенциалы вне формы
    plt.scatter(exterior_coordinates[:, 0], exterior_coordinates[:, 1], c=exterior_potentials, cmap='coolwarm', s=5)
    plt.title('Double Layer Potential Outside the Shape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Potential')
    plt.show()


def z(x_values, y_values):
    return (x_values - 2)**2 + (y_values - 3)**2 - 4**2


# Функция для генерации геометрической формы
def generate_shape(shape_type, custom_formula_func=None):
    if shape_type == 'circle':
        return Point(0, 0).buffer(1)
    elif shape_type == 'ellipse':
        return Point(0, 0).buffer(1.5)
    elif shape_type == 'rectangle':
        return Polygon([(0, 0), (0, 2), (3, 2), (3, 0)])
    elif shape_type == 'custom' and custom_formula_func is not None:
        x_values = np.linspace(-1, 1, 100)
        y_values = np.linspace(-1, 1, 100)
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        z_values = np.array([[custom_formula_func(x_val, y_val) for x_val in x_values] for y_val in y_values])
        shape_indices = np.argwhere(z_values == 0)
        shape = [(x_mesh[i, j], y_mesh[i, j]) for i, j in shape_indices]
        return Polygon(shape)
    else:
        return None


# Создаем графический интерфейс
root = tk.Tk()
root.title("Double Layer Potential Calculator")

# Добавляем метку и поле для ввода количества точек
num_points_label = tk.Label(root, text="Number of Points:")
num_points_label.grid(row=0, column=0, sticky=tk.W)
num_points_entry = tk.Entry(root)
num_points_entry.insert(tk.END, "1000")
num_points_entry.grid(row=0, column=1)

# Добавляем метку и поле для ввода количества точек на границе
num_boundary_points_label = tk.Label(root, text="Number of Boundary Points:")
num_boundary_points_label.grid(row=1, column=0, sticky=tk.W)
num_boundary_points_entry = tk.Entry(root)
num_boundary_points_entry.insert(tk.END, "100")
num_boundary_points_entry.grid(row=1, column=1)

# Добавляем метку и поле для ввода количества точек внутри формы
num_interior_points_label = tk.Label(root, text="Number of Interior Points:")
num_interior_points_label.grid(row=2, column=0, sticky=tk.W)
num_interior_points_entry = tk.Entry(root)
num_interior_points_entry.insert(tk.END, "500")
num_interior_points_entry.grid(row=2, column=1)

# Добавляем метку и поле для ввода диэлектрической константы
diel_constant_label = tk.Label(root, text="Dielectric Constant:")
diel_constant_label.grid(row=3, column=0, sticky=tk.W)
diel_constant_entry = tk.Entry(root)
diel_constant_entry.insert(tk.END, "78.5")
diel_constant_entry.grid(row=3, column=1)

# Добавляем метку и поле для ввода концентрации ионов
ion_concentration_label = tk.Label(root, text="Ion Concentration:")
ion_concentration_label.grid(row=4, column=0, sticky=tk.W)
ion_concentration_entry = tk.Entry(root)
ion_concentration_entry.insert(tk.END, "0.1")
ion_concentration_entry.grid(row=4, column=1)

# Добавляем метку и поле для ввода температуры
temperature_label = tk.Label(root, text="Temperature:")
temperature_label.grid(row=5, column=0, sticky=tk.W)
temperature_entry = tk.Entry(root)
temperature_entry.insert(tk.END, "298.15")
temperature_entry.grid(row=5, column=1)

# Добавляем метку и выпадающий список для выбора формы
shape_label = tk.Label(root, text="Shape:")
shape_label.grid(row=6, column=0, sticky=tk.W)
shape_var = tk.StringVar(root)
shape_var.set('circle')
shape_option_menu = tk.OptionMenu(root, shape_var, 'circle', 'ellipse', 'rectangle', 'custom')
shape_option_menu.grid(row=6, column=1)

# Добавляем метку и поле для ввода пользовательской формулы
custom_formula_label = tk.Label(root, text="Custom Formula:")
custom_formula_label.grid(row=7, column=0, sticky=tk.W)
custom_formula_entry = tk.Entry(root)
custom_formula_entry.insert(tk.END, "x**2 + y**2 + 1")
custom_formula_entry.grid(row=7, column=1)

# Добавляем кнопку "Вычислить"
calculate_button = tk.Button(root, text="Calculate", command=calculate_potentials)
calculate_button.grid(row=8, column=0, columnspan=2)

# Запускаем главный цикл обработки событий
root.mainloop()
