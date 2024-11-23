import math
import numpy as np
import random
import matplotlib.pyplot as plt

cities = np.array([
    [0.30521241, 0.18630380], [0.5416322, 0.75636134], [0.9146427, 0.56843619], [0.23487191, 0.71608927],
    [0.23744324, 0.04432229], [0.98550312, 0.31416236], [0.06900982, 0.23309370], [0.18780658, 0.84952662],
    [0.65076716, 0.11224197], [0.81890889, 0.66712018], [0.64522359, 0.44572718], [0.04612005, 0.04930954],
    [0.40655274, 0.49232880], [0.50282918, 0.67212568], [0.99884629, 0.65680969], [0.93605003, 0.62967331],
    [0.99393103, 0.11286275], [0.69173572, 0.07072200], [0.26446284, 0.97047993], [0.29561705, 0.60990036],
    [0.96738685, 0.22562734], [0.3909346, 0.02425457], [0.15647212, 0.64319123], [0.28889163, 0.39333727],
    [0.12540389, 0.01490534], [0.00287364, 0.64631502], [0.69121243, 0.29458765], [0.70367699, 0.34631052],
    [0.14231427, 0.54070638], [0.05646348, 0.60758725], [0.83179647, 0.26361788], [0.80634361, 0.97518400],
    [0.06423712, 0.68769045], [0.03403532, 0.81497189], [0.49067572, 0.60675072], [0.57725681, 0.56723063],
    [0.46786624, 0.38511662], [0.53191193, 0.38229725], [0.20038992, 0.71712527], [0.85241791, 0.51498287]
])

#cities -> the list of cities travelled
#path -> list of city indices showing the order
def energy(path, cities):
    distance = 0.0
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]
        between = math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        distance += between
    return distance

#cities -> the list of cities travelled
# t -> initial temperature
# iterations_num -> total number of iterations
# a -> cooling factor
def metropolis(cities, t, iterations_num, a):
    path = list(range(len(cities)))
    random.shuffle(path)
    current_energy = energy(path, cities)
    energy_history = [current_energy]

    for i in range(iterations_num):
        new_path = path.copy()
        x, y = random.sample(range(len(path)), 2)
        temp = new_path[x]
        new_path[x] = new_path[y]
        new_path[y] = temp
        new_energy = energy(new_path, cities)
        if new_energy >= current_energy:
            prob = math.exp(-(new_energy - current_energy) / t)
            if random.random() < prob:
                path = new_path
                current_energy = new_energy
        else:
            path = new_path
            current_energy = new_energy

        energy_history.append(current_energy)
        t = a * t
    return path, current_energy, energy_history

#path -> list of city indices in the solution order
#cities -> array of city coordinates
#current_energy -> total distance of the best solution
def plot_path(path, cities, current_energy):
    plt.figure(figsize=(10, 6))
    ordered_cities = cities[path]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
    plt.plot([ordered_cities[0, 0], ordered_cities[1, 0]],
             [ordered_cities[0, 1], ordered_cities[1, 1]],
             color='cyan', lw=5)
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], '-o', label=f"Total Distance: {current_energy:.2f}")
    start_city = ordered_cities[0]
    plt.scatter(start_city[0], start_city[1], color='green', label='Starting City', zorder=5, s=100)
    plt.scatter(cities[:, 0], cities[:, 1], color='red', label='Cities', zorder=4)
    plt.title("TSP Solution")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

#energy_history -> list of total distances during the iterations
def plot_energy(energy_history):
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, label="Energy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Total Distance)")
    plt.title("Energy vs. Iterations")
    plt.legend()
    plt.show()

#all_energies -> stores the distances found
#least_energy -> tracks minimum distance found
#best_solution -> tracks the path with minimum distance
#best_energy_history -> tracks the energy history of the best solution

def main():
    initial_temperature = 10.0
    iterations = 10000
    cooling_factor = 0.999

    all_energies = []
    least_energy = float('inf')
    best_solution = None
    best_energy_history = None

    for _ in range(100):
        path, current_energy, energy_history = metropolis(cities, initial_temperature, iterations, cooling_factor)
        all_energies.append(current_energy)
        if current_energy < least_energy:
            least_energy = current_energy
            best_solution = path
            best_energy_history = energy_history

    plot_path(best_solution, cities, least_energy)
    plot_energy(best_energy_history)

    print(f"Minimum total distance: {least_energy:.4f}")
    print("All energies from 100 runs:", all_energies)


if __name__ == "__main__":
    main()
